from setuptools import setup, find_packages

setup(
    name="nccl-lsa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    description="NCCL-LSA Python bindings for AllReduce",
    author="NCCL Team",
    python_requires=">=3.8",
)
