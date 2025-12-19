"""
LLaMA-3 8B Production Training with NCCL-LSA
Streamlined for actual training workloads
"""

import os
import time
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from pathlib import Path

try:
    import tomli as tomllib  # Python < 3.11
except ImportError:
    try:
        import tomllib  # Python >= 3.11
    except ImportError:
        import toml as tomllib  # Fallback to toml package

# Import NCCL-LSA utilities
from nccl_lsa.dist_utils import init_distributed, allreduce_tensor


class LLaMA3Trainer:
    """Production trainer for LLaMA-3 8B with NCCL-LSA"""
    
    @classmethod
    def from_config(cls, config_path: str = "config.toml"):
        """Load trainer from TOML config file"""
        if Path(config_path).exists():
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            
            # Extract configuration
            model_config = config.get("model", {})
            training_config = config.get("training", {})
            optimizer_config = config.get("optimizer", {})
            data_config = config.get("data", {})
            checkpoint_config = config.get("checkpointing", {})
            logging_config = config.get("logging", {})
            system_config = config.get("system", {})
            
            # Create trainer instance
            trainer = cls(
                model_name=model_config.get("model_name", "meta-llama/Llama-3-8b-hf"),
                max_seq_length=model_config.get("max_seq_length", 2048),
                batch_size=training_config.get("batch_size", 1),
                gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
                learning_rate=training_config.get("learning_rate", 3e-4),
                warmup_steps=training_config.get("warmup_steps", 100),
                max_steps=training_config.get("max_steps", 10000),
                beta1=optimizer_config.get("beta1", 0.9),
                beta2=optimizer_config.get("beta2", 0.95),
                weight_decay=optimizer_config.get("weight_decay", 0.1),
                gradient_clip_norm=optimizer_config.get("gradient_clip_norm", 1.0),
                save_steps=checkpoint_config.get("save_steps", 1000),
                checkpoint_dir=checkpoint_config.get("checkpoint_dir", "checkpoints"),
                log_steps=logging_config.get("log_steps", 10),
                num_workers=system_config.get("num_workers", 4),
                use_bf16=system_config.get("use_bf16", True),
                use_gradient_checkpointing=system_config.get("use_gradient_checkpointing", True),
            )
            
            # Store additional configs for later use
            trainer.data_config = data_config
            trainer.logging_config = logging_config
            
            return trainer
        else:
            print(f"Warning: Config file {config_path} not found. Using default parameters.")
            return cls()
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8b-hf",
        max_seq_length: int = 2048,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 3e-4,
        warmup_steps: int = 100,
        max_steps: int = 10000,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 0.1,
        gradient_clip_norm: float = 1.0,
        save_steps: int = 1000,
        checkpoint_dir: str = "checkpoints",
        log_steps: int = 10,
        num_workers: int = 4,
        use_bf16: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        # Initialize distributed with NCCL-LSA
        init_distributed()
        
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device("cuda", self.local_rank)
        
        torch.cuda.set_device(self.device)
        
        # Training config
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Optimizer config
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        
        # Checkpointing config
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        
        # Logging config
        self.log_steps = log_steps
        
        # System config
        self.num_workers = num_workers
        self.use_bf16 = use_bf16
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Additional configs (set by from_config)
        self.data_config = {}
        self.logging_config = {}
        
        if self.rank == 0:
            print(f"🚀 Initialized LLaMA-3 8B Trainer")
            print(f"   World size: {self.world_size}")
            print(f"   Batch size per GPU: {self.batch_size}")
            print(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"   Effective batch size: {self.batch_size * self.gradient_accumulation_steps * self.world_size}")
            print(f"   Learning rate: {self.learning_rate}")
            print(f"   Max steps: {self.max_steps}")
            print(f"   Precision: {'BF16' if self.use_bf16 else 'FP32'}")
            print(f"   Gradient checkpointing: {self.use_gradient_checkpointing}")
            print(f"   Using NCCL-LSA for AllReduce")
    
    def load_model(self):
        """Load LLaMA-3 8B model"""
        if self.rank == 0:
            print(f"\n📦 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine dtype
        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map={"": self.device},
            torch_dtype=dtype,
            use_cache=False,
        )
        
        # Enable gradient checkpointing to save memory
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if self.rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            dtype_str = "BF16" if self.use_bf16 else "FP16"
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Model memory: ~{total_params * 2 / 1e9:.2f} GB ({dtype_str})")
            if self.use_gradient_checkpointing:
                print(f"   Gradient checkpointing: Enabled")
    
    def setup_optimizer(self):
        """Setup optimizer with linear warmup"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay
        )
        
        if self.rank == 0:
            print(f"\n⚙️  Optimizer: AdamW")
            print(f"   Learning rate: {self.learning_rate}")
            print(f"   Betas: ({self.beta1}, {self.beta2})")
            print(f"   Weight decay: {self.weight_decay}")
    
    def load_data(self, dataset_name: str = None, dataset_config: str = None):
        """Load and prepare training data"""
        # Use data_config if available, otherwise use parameters
        if not dataset_name and self.data_config:
            dataset_name = self.data_config.get("dataset_name", "wikitext")
            dataset_config = self.data_config.get("dataset_config", "wikitext-2-raw-v1")
        
        # Defaults if still not set
        dataset_name = dataset_name or "wikitext"
        dataset_config = dataset_config or "wikitext-2-raw-v1"
        
        if self.rank == 0:
            print(f"\n📚 Loading dataset: {dataset_name}/{dataset_config}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, dataset_config, split="train")
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        # Setup distributed sampler
        sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        self.train_loader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        if self.rank == 0:
            print(f"   Dataset size: {len(tokenized_dataset):,} examples")
            print(f"   Batches per epoch: {len(self.train_loader):,}")
            print(f"   Dataloader workers: {self.num_workers}")
    
    def allreduce_gradients(self):
        """Perform NCCL-LSA AllReduce on gradients"""
        for param in self.model.parameters():
            if param.grad is not None:
                allreduce_tensor(param.grad)
                param.grad.div_(self.world_size)
    
    def get_lr(self, step: int):
        """Get learning rate with linear warmup"""
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        return self.learning_rate
    
    def train_step(self, batch, step: int):
        """Single training step"""
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def train(self):
        """Main training loop"""
        if self.rank == 0:
            print(f"\n🏋️  Starting training...")
            print("="*70)
        
        self.model.train()
        global_step = 0
        epoch = 0
        total_loss = 0.0
        
        start_time = time.time()
        
        while global_step < self.max_steps:
            epoch += 1
            self.train_loader.sampler.set_epoch(epoch)
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Update learning rate
                current_lr = self.get_lr(global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # Training step
                loss = self.train_step(batch, global_step)
                total_loss += loss
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # AllReduce gradients with NCCL-LSA
                    allreduce_start = time.time()
                    self.allreduce_gradients()
                    allreduce_time = time.time() - allreduce_start
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % self.log_steps == 0 and self.rank == 0:
                        avg_loss = total_loss / self.log_steps
                        elapsed = time.time() - start_time
                        tokens_per_sec = (
                            self.batch_size * 
                            self.gradient_accumulation_steps * 
                            self.world_size * 
                            self.max_seq_length * 
                            self.log_steps / elapsed
                        )
                        
                        print(f"Step {global_step:5d} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {current_lr:.2e} | "
                              f"AllReduce: {allreduce_time*1000:.1f}ms | "
                              f"Tokens/s: {tokens_per_sec:.0f}")
                        
                        total_loss = 0.0
                        start_time = time.time()
                    
                    # Checkpointing
                    if global_step % self.save_steps == 0 and self.rank == 0:
                        self.save_checkpoint(global_step)
                    
                    # Early stopping
                    if global_step >= self.max_steps:
                        break
            
            if global_step >= self.max_steps:
                break
        
        if self.rank == 0:
            print("="*70)
            print(f"✅ Training completed! Final step: {global_step}")
            self.save_checkpoint(global_step, final=True)
    
    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint"""
        checkpoint_path = f"{self.checkpoint_dir}/step_{step}" if not final else f"{self.checkpoint_dir}/final"
        
        if self.rank == 0:
            print(f"\n💾 Saving checkpoint to {checkpoint_path}...")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
            
            # Save training state
            torch.save({
                'step': step,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': {
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'gradient_accumulation_steps': self.gradient_accumulation_steps,
                }
            }, f"{checkpoint_path}/training_state.pt")
            
            print(f"   ✓ Checkpoint saved")


def main():
    """Main execution"""
    
    # Try to load from config.toml, fall back to defaults
    config_path = "config.toml"
    
    if Path(config_path).exists():
        print(f"📋 Loading configuration from {config_path}")
        trainer = LLaMA3Trainer.from_config(config_path)
    else:
        print(f"⚠️  Config file {config_path} not found, using default parameters")
        trainer = LLaMA3Trainer()
    
    # Setup
    trainer.load_model()
    trainer.setup_optimizer()
    trainer.load_data()
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()