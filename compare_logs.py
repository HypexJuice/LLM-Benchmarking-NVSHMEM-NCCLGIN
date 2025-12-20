#!/usr/bin/env python3
"""
Simple Profile Comparison Script (works without nsys CLI)
Parses log files and creates comparison plots.
"""

import re
import os
import sys
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def parse_log_file(log_file):
    """Parse training log file for metrics."""
    
    metrics = {
        'tokens_per_sec': [],
        'loss': [],
        'steps': [],
        'tflops': [],
        'mfu': [],
        'grad_norm': [],
        'memory_gb': [],
        'nvtx_info': {}
    }
    
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} not found")
        return metrics
    
    print(f"Parsing {log_file}...")
    
    # Track which steps we've already seen to avoid duplicates (multi-rank logs)
    seen_steps = set()
    
    with open(log_file, 'r') as f:
        for line in f:
            # Skip if not a metric line
            if 'step:' not in line or 'loss:' not in line:
                continue
            
            # Parse step number
            step_match = re.search(r'step:\s*(\d+)', line)
            if not step_match:
                continue
            
            step = int(step_match.group(1))
            
            # Skip duplicates (same step from different ranks)
            if step in seen_steps:
                continue
            seen_steps.add(step)
            
            # Parse tps (tokens per second)
            tps_match = re.search(r'tps:\s*([0-9.]+)', line)
            if tps_match:
                metrics['tokens_per_sec'].append(float(tps_match.group(1)))
            else:
                # If no tps found, skip this line
                continue
            
            # Parse loss
            loss_match = re.search(r'loss:\s*([0-9.]+)', line)
            if loss_match:
                metrics['loss'].append(float(loss_match.group(1)))
            
            # Parse tflops
            tflops_match = re.search(r'tflops:\s*([0-9.]+)', line)
            if tflops_match:
                metrics['tflops'].append(float(tflops_match.group(1)))
            
            # Parse mfu (model FLOPs utilization)
            mfu_match = re.search(r'mfu:\s*([0-9.]+)%', line)
            if mfu_match:
                metrics['mfu'].append(float(mfu_match.group(1)))
            
            # Parse grad_norm
            grad_norm_match = re.search(r'grad_norm:\s*([0-9.]+)', line)
            if grad_norm_match:
                metrics['grad_norm'].append(float(grad_norm_match.group(1)))
            
            # Parse memory
            memory_match = re.search(r'memory:\s*([0-9.]+)GiB', line)
            if memory_match:
                metrics['memory_gb'].append(float(memory_match.group(1)))
            
            # Add step
            metrics['steps'].append(step)
    
    return metrics


def plot_comparison(nccl_metrics, nvshmem_metrics, output_dir='plots'):
    """Create comparison plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Throughput (TPS) over steps
    ax1 = fig.add_subplot(gs[0, :])
    if nccl_metrics['tokens_per_sec'] and nvshmem_metrics['tokens_per_sec']:
        ax1.plot(nccl_metrics['steps'], nccl_metrics['tokens_per_sec'], 
                marker='o', label='NCCL', linewidth=2.5, markersize=8, 
                color='#3498db', alpha=0.8)
        ax1.plot(nvshmem_metrics['steps'], nvshmem_metrics['tokens_per_sec'], 
                marker='s', label='NVSHMEM', linewidth=2.5, markersize=8,
                color='#e74c3c', alpha=0.8)
        
        ax1.set_xlabel('Training Step', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Throughput (tokens/sec)', fontsize=13, fontweight='bold')
        ax1.set_title('Training Throughput Comparison: NCCL vs NVSHMEM', 
                     fontsize=15, fontweight='bold', pad=20)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss over steps
    ax2 = fig.add_subplot(gs[1, :])
    if nccl_metrics['loss'] and nvshmem_metrics['loss']:
        ax2.plot(nccl_metrics['steps'], nccl_metrics['loss'], 
                marker='o', label='NCCL', linewidth=2.5, markersize=8,
                color='#3498db', alpha=0.7)
        ax2.plot(nvshmem_metrics['steps'], nvshmem_metrics['loss'], 
                marker='s', label='NVSHMEM', linewidth=2.5, markersize=8,
                color='#e74c3c', alpha=0.7)
        
        ax2.set_xlabel('Training Step', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax2.set_title('Training Loss Comparison', fontsize=15, fontweight='bold', pad=20)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: MFU (Model FLOPs Utilization) over steps
    ax3 = fig.add_subplot(gs[2, :])
    if nccl_metrics['mfu'] and nvshmem_metrics['mfu']:
        ax3.plot(nccl_metrics['steps'], nccl_metrics['mfu'], 
                marker='o', label='NCCL', linewidth=2.5, markersize=8,
                color='#3498db', alpha=0.8)
        ax3.plot(nvshmem_metrics['steps'], nvshmem_metrics['mfu'], 
                marker='s', label='NVSHMEM', linewidth=2.5, markersize=8,
                color='#e74c3c', alpha=0.8)
        
        ax3.set_xlabel('Training Step', fontsize=13, fontweight='bold')
        ax3.set_ylabel('MFU (%)', fontsize=13, fontweight='bold')
        ax3.set_title('Model FLOPs Utilization (MFU) Comparison', fontsize=15, fontweight='bold', pad=20)
        ax3.legend(fontsize=11, loc='best')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average throughput comparison (bar chart)
    ax4 = fig.add_subplot(gs[3, 0])
    if nccl_metrics['tokens_per_sec'] and nvshmem_metrics['tokens_per_sec']:
        nccl_avg = np.mean(nccl_metrics['tokens_per_sec'])
        nvshmem_avg = np.mean(nvshmem_metrics['tokens_per_sec'])
        
        bars = ax4.bar(['NCCL', 'NVSHMEM'], [nccl_avg, nvshmem_avg], 
                      color=['#3498db', '#e74c3c'], alpha=0.8, width=0.6)
        
        ax4.set_ylabel('Avg Throughput (tokens/sec)', fontsize=12, fontweight='bold')
        ax4.set_title('Average Throughput', fontsize=13, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 5: Speedup calculation
    ax5 = fig.add_subplot(gs[3, 1])
    if nccl_metrics['tokens_per_sec'] and nvshmem_metrics['tokens_per_sec']:
        nccl_avg = np.mean(nccl_metrics['tokens_per_sec'])
        nvshmem_avg = np.mean(nvshmem_metrics['tokens_per_sec'])
        
        if nccl_avg > 0:
            speedup_pct = ((nvshmem_avg / nccl_avg) - 1) * 100
            
            color = '#27ae60' if speedup_pct > 0 else '#e74c3c'
            bar = ax5.barh(['Speedup'], [speedup_pct], color=color, alpha=0.8, height=0.4)
            
            ax5.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
            ax5.set_title('NVSHMEM Speedup over NCCL', fontsize=13, fontweight='bold', pad=15)
            ax5.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add value label
            label_x = speedup_pct + (2 if speedup_pct > 0 else -2)
            ha = 'left' if speedup_pct > 0 else 'right'
            ax5.text(label_x, 0, f'{speedup_pct:+.1f}%',
                    ha=ha, va='center', fontsize=14, fontweight='bold')
    
    plt.savefig(f'{output_dir}/comparison_full.png', dpi=300, bbox_inches='tight')
    print(f"✓ Full comparison saved to {output_dir}/comparison_full.png")
    plt.close()
    
    # Create separate detailed speedup plot
    create_speedup_plot(nccl_metrics, nvshmem_metrics, output_dir)


def create_speedup_plot(nccl_metrics, nvshmem_metrics, output_dir):
    """Create detailed speedup analysis plot."""
    
    if not nccl_metrics['tokens_per_sec'] or not nvshmem_metrics['tokens_per_sec']:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Per-step speedup
    # Match steps that exist in both
    nccl_steps_dict = {step: tps for step, tps in zip(nccl_metrics['steps'], nccl_metrics['tokens_per_sec'])}
    nvshmem_steps_dict = {step: tps for step, tps in zip(nvshmem_metrics['steps'], nvshmem_metrics['tokens_per_sec'])}
    
    common_steps = sorted(set(nccl_steps_dict.keys()) & set(nvshmem_steps_dict.keys()))
    
    if common_steps:
        speedups = []
        for step in common_steps:
            if nccl_steps_dict[step] > 0:
                speedup = (nvshmem_steps_dict[step] / nccl_steps_dict[step] - 1) * 100
                speedups.append(speedup)
            else:
                speedups.append(0)
        
        colors = ['#27ae60' if s > 0 else '#e74c3c' for s in speedups]
        
        ax1.bar(common_steps, speedups, color=colors, alpha=0.7, width=max(1, common_steps[-1] * 0.05))
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax1.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speedup (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Per-Step Speedup: NVSHMEM vs NCCL', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Summary statistics
    nccl_avg = np.mean(nccl_metrics['tokens_per_sec'])
    nvshmem_avg = np.mean(nvshmem_metrics['tokens_per_sec'])
    nccl_std = np.std(nccl_metrics['tokens_per_sec'])
    nvshmem_std = np.std(nvshmem_metrics['tokens_per_sec'])
    
    x = [0, 1]
    means = [nccl_avg, nvshmem_avg]
    stds = [nccl_std, nvshmem_std]
    
    bars = ax2.bar(x, means, yerr=stds, capsize=10, 
                   color=['#3498db', '#e74c3c'], alpha=0.8, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['NCCL', 'NVSHMEM'])
    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Throughput ± Std Dev', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax2.text(bar.get_x() + bar.get_width()/2., mean + std,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Speedup analysis saved to {output_dir}/speedup_analysis.png")
    plt.close()


def print_summary(nccl_metrics, nvshmem_metrics):
    """Print summary statistics."""
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    if nccl_metrics['tokens_per_sec'] and nvshmem_metrics['tokens_per_sec']:
        nccl_avg = np.mean(nccl_metrics['tokens_per_sec'])
        nccl_std = np.std(nccl_metrics['tokens_per_sec'])
        nccl_min = np.min(nccl_metrics['tokens_per_sec'])
        nccl_max = np.max(nccl_metrics['tokens_per_sec'])
        
        nvshmem_avg = np.mean(nvshmem_metrics['tokens_per_sec'])
        nvshmem_std = np.std(nvshmem_metrics['tokens_per_sec'])
        nvshmem_min = np.min(nvshmem_metrics['tokens_per_sec'])
        nvshmem_max = np.max(nvshmem_metrics['tokens_per_sec'])
        
        print("\nThroughput Statistics (tokens/sec):")
        print("-" * 80)
        print(f"{'Backend':<15} {'Average':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
        print("-" * 80)
        print(f"{'NCCL':<15} {nccl_avg:<15.2f} {nccl_std:<15.2f} {nccl_min:<15.2f} {nccl_max:<15.2f}")
        print(f"{'NVSHMEM':<15} {nvshmem_avg:<15.2f} {nvshmem_std:<15.2f} {nvshmem_min:<15.2f} {nvshmem_max:<15.2f}")
        
        if nccl_avg > 0:
            speedup = nvshmem_avg / nccl_avg
            improvement = (speedup - 1) * 100
            
            print("\n" + "-" * 80)
            print(f"Speedup: {speedup:.4f}x ({improvement:+.2f}%)")
            
            if improvement > 10:
                print("Result: ✓✓ Excellent speedup!")
            elif improvement > 5:
                print("Result: ✓ Good speedup")
            elif improvement > 0:
                print("Result: ~ Small speedup")
            else:
                print("Result: ⚠ NVSHMEM slower (check configuration)")
    
    # MFU comparison
    if nccl_metrics['mfu'] and nvshmem_metrics['mfu']:
        nccl_mfu_avg = np.mean(nccl_metrics['mfu'])
        nvshmem_mfu_avg = np.mean(nvshmem_metrics['mfu'])
        
        print("\n" + "-" * 80)
        print("Model FLOPs Utilization (MFU):")
        print(f"  NCCL:    {nccl_mfu_avg:.2f}%")
        print(f"  NVSHMEM: {nvshmem_mfu_avg:.2f}%")
        
        if nccl_mfu_avg > 0:
            mfu_improvement = ((nvshmem_mfu_avg / nccl_mfu_avg) - 1) * 100
            print(f"  Improvement: {mfu_improvement:+.2f}%")
    
    # TFLOPS comparison
    if nccl_metrics['tflops'] and nvshmem_metrics['tflops']:
        nccl_tflops_avg = np.mean(nccl_metrics['tflops'])
        nvshmem_tflops_avg = np.mean(nvshmem_metrics['tflops'])
        
        print("\n" + "-" * 80)
        print("TFLOPS:")
        print(f"  NCCL:    {nccl_tflops_avg:.2f}")
        print(f"  NVSHMEM: {nvshmem_tflops_avg:.2f}")
    
    if nccl_metrics['loss'] and nvshmem_metrics['loss']:
        nccl_final_loss = nccl_metrics['loss'][-1] if nccl_metrics['loss'] else 0
        nvshmem_final_loss = nvshmem_metrics['loss'][-1] if nvshmem_metrics['loss'] else 0
        
        print("\n" + "-" * 80)
        print("Final Loss:")
        print(f"  NCCL:    {nccl_final_loss:.6f}")
        print(f"  NVSHMEM: {nvshmem_final_loss:.6f}")
        
        # Check if losses are similar (training correctness)
        loss_diff = abs(nccl_final_loss - nvshmem_final_loss)
        if loss_diff < 0.01:
            print(f"  ✓ Loss values match (both training correctly)")
        else:
            print(f"  ⚠ Loss difference: {loss_diff:.6f} (check training consistency)")
    
    # Memory comparison
    if nccl_metrics['memory_gb'] and nvshmem_metrics['memory_gb']:
        nccl_mem_avg = np.mean(nccl_metrics['memory_gb'])
        nvshmem_mem_avg = np.mean(nvshmem_metrics['memory_gb'])
        
        print("\n" + "-" * 80)
        print("GPU Memory Usage:")
        print(f"  NCCL:    {nccl_mem_avg:.2f} GiB")
        print(f"  NVSHMEM: {nvshmem_mem_avg:.2f} GiB")
        
        mem_diff = nvshmem_mem_avg - nccl_mem_avg
        if abs(mem_diff) > 0.5:
            print(f"  Difference: {mem_diff:+.2f} GiB")
    
    print("="*80 + "\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare NCCL vs NVSHMEM from training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare using log files
  python compare_logs.py \\
    --nccl logs/bench_nccl_*.log \\
    --nvshmem logs/bench_nvshmem_*.log

  # With custom output directory
  python compare_logs.py \\
    --nccl logs/bench_nccl_*.log \\
    --nvshmem logs/bench_nvshmem_*.log \\
    --output plots/my_comparison
        """
    )
    
    parser.add_argument('--nccl', required=True, help='NCCL log file')
    parser.add_argument('--nvshmem', required=True, help='NVSHMEM log file')
    parser.add_argument('--output', default='plots', help='Output directory')
    
    args = parser.parse_args()
    
    # Expand wildcards
    nccl_files = glob.glob(args.nccl)
    nvshmem_files = glob.glob(args.nvshmem)
    
    if not nccl_files:
        print(f"Error: No NCCL log files found matching: {args.nccl}")
        sys.exit(1)
    
    if not nvshmem_files:
        print(f"Error: No NVSHMEM log files found matching: {args.nvshmem}")
        sys.exit(1)
    
    # Use first matching file
    nccl_log = nccl_files[0]
    nvshmem_log = nvshmem_files[0]
    
    print("\n" + "="*80)
    print("TRAINING LOG COMPARISON TOOL")
    print("="*80)
    print(f"\nNCCL log:    {nccl_log}")
    print(f"NVSHMEM log: {nvshmem_log}\n")
    
    # Parse logs
    nccl_metrics = parse_log_file(nccl_log)
    print(nccl_metrics)
    nvshmem_metrics = parse_log_file(nvshmem_log)
    print(nvshmem_metrics)
    # Print summary
    print_summary(nccl_metrics, nvshmem_metrics)
    
    # Create plots
    print("Generating plots...")
    plot_comparison(nccl_metrics, nvshmem_metrics, args.output)
    
    print("\n" + "="*80)
    print("✓ COMPARISON COMPLETE")
    print("="*80)
    print(f"\nPlots saved to {args.output}/:")
    print("  - comparison_full.png: Complete comparison dashboard")
    print("  - speedup_analysis.png: Detailed speedup analysis")
    print()


if __name__ == "__main__":
    main()