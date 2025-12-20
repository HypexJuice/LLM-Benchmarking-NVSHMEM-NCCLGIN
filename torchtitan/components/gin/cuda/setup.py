from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

nccl_include = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/comm_libs/nccl/include"  

setup(
    name="gin_lsa_ext",
    ext_modules=[
        CUDAExtension(
            name="gin_lsa_ext",
            sources=[
                "gin_lsa.cpp",
                "gin_all_gather.cpp",
            ],
            include_dirs=[nccl_include],      
            extra_cuda_cflags=["--use_fast_math"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
