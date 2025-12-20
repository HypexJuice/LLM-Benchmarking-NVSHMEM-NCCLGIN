#!/usr/bin/env python3
"""
Find NVSHMEM installation and generate environment variable setup.
Run this script to get the exact export commands you need.
"""

import os
import sys
from pathlib import Path


def find_nvshmem_installation():
    """Search for NVSHMEM installation in various locations."""
    
    print("=" * 60)
    print("Searching for NVSHMEM Installation")
    print("=" * 60)
    print()
    
    results = {}
    
    # Method 1: Check if nvshmem4py is installed as Python package
    print("1. Checking Python package installation...")
    try:
        import nvshmem
        pkg_path = Path(nvshmem.__file__).parent
        results['python_package'] = pkg_path
        print(f"   ✓ Found NVSHMEM4Py package: {pkg_path}")
        
        # Check for library directories
        possible_lib_dirs = [
            pkg_path / "lib",
            pkg_path.parent / "nvidia" / "nvshmem" / "lib",
            pkg_path.parent.parent / "nvidia" / "nvshmem" / "lib",
        ]
        
        for lib_dir in possible_lib_dirs:
            if lib_dir.exists():
                lib_files = list(lib_dir.glob("libnvshmem*.so*"))
                if lib_files:
                    results['lib_dir'] = lib_dir
                    print(f"   ✓ Found library directory: {lib_dir}")
                    print(f"      Contains: {[f.name for f in lib_files[:3]]}")
                    break
                    
    except ImportError:
        print("   ✗ NVSHMEM4Py package not found via import")
    print()
    
    # Method 2: Check conda environment
    print("2. Checking conda environment...")
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_path = Path(conda_prefix)
        print(f"   Conda prefix: {conda_path}")
        
        # Search in conda prefix
        site_packages = list(conda_path.glob("lib/python*/site-packages"))
        for sp in site_packages:
            nvshmem_wheel = sp / "nvidia" / "nvshmem"
            if nvshmem_wheel.exists():
                results['conda_wheel'] = nvshmem_wheel
                print(f"   ✓ Found NVSHMEM wheel: {nvshmem_wheel}")
                
                lib_dir = nvshmem_wheel / "lib"
                if lib_dir.exists():
                    results['lib_dir'] = lib_dir
                    print(f"   ✓ Library directory: {lib_dir}")
                break
    else:
        print("   Not in a conda environment")
    print()
    
    # Method 3: Check system paths
    print("3. Checking system-wide installations...")
    system_paths = [
        Path("/usr/local"),
        Path("/opt/nvidia"),
        Path("/opt"),
        Path.home() / ".local",
    ]
    
    for base_path in system_paths:
        if not base_path.exists():
            continue
            
        # Look for nvshmem directories
        for item in base_path.glob("nvshmem*"):
            if item.is_dir():
                lib_dir = item / "lib"
                if lib_dir.exists() and list(lib_dir.glob("libnvshmem*.so*")):
                    results['system_install'] = item
                    results['lib_dir'] = lib_dir
                    print(f"   ✓ Found system installation: {item}")
                    break
    print()
    
    # Method 4: Check LD_LIBRARY_PATH
    print("4. Checking current LD_LIBRARY_PATH...")
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_library_path:
        for path_str in ld_library_path.split(':'):
            path = Path(path_str)
            if path.exists() and any(path.glob("libnvshmem*.so*")):
                results['ld_library_path'] = path
                print(f"   ✓ Found NVSHMEM libs in: {path}")
                break
    else:
        print("   LD_LIBRARY_PATH not set")
    print()
    
    return results


def generate_setup_commands(results):
    """Generate the environment variable setup commands."""
    
    print("=" * 60)
    print("Environment Setup Commands")
    print("=" * 60)
    print()
    
    if not results:
        print("❌ NVSHMEM not found on this system!")
        print()
        print("Installation options:")
        print()
        print("1. Install via pip (recommended):")
        print("   pip install nvshmem4py-cu12  # for CUDA 12.x")
        print("   pip install nvshmem4py-cu11  # for CUDA 11.x")
        print()
        print("2. Install via conda:")
        print("   conda install -c nvidia nvshmem4py")
        print()
        print("3. Download and build from source:")
        print("   https://developer.nvidia.com/nvshmem")
        print()
        return
    
    # Determine the best NVSHMEM_HOME and LD_LIBRARY_PATH
    lib_dir = results.get('lib_dir')
    
    if lib_dir:
        nvshmem_home = lib_dir.parent
        
        print("✓ NVSHMEM found! Add these to your environment:")
        print()
        print("# Option 1: Add to ~/.bashrc (persistent)")
        print("=" * 60)
        print(f"export NVSHMEM_HOME={nvshmem_home}")
        print(f"export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH")
        print()
        
        print("# Option 2: Add to your SLURM batch script")
        print("=" * 60)
        print(f"export NVSHMEM_HOME={nvshmem_home}")
        print(f"export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH")
        print()
        
        print("# Option 3: Source this in your shell (temporary)")
        print("=" * 60)
        setup_file = Path.home() / "nvshmem_env.sh"
        with open(setup_file, 'w') as f:
            f.write(f"export NVSHMEM_HOME={nvshmem_home}\n")
            f.write(f"export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH\n")
        
        print(f"# Setup script created: {setup_file}")
        print(f"source {setup_file}")
        print()
        
    else:
        # Fallback: use Python package location
        pkg_path = results.get('python_package')
        if pkg_path:
            print("✓ NVSHMEM installed as Python wheel (no separate NVSHMEM_HOME needed)")
            print()
            print("The libraries are bundled with the Python package.")
            print("No additional LD_LIBRARY_PATH setup required for basic usage.")
            print()
            print("If you need to set NVSHMEM_HOME for advanced features:")
            print(f"export NVSHMEM_HOME={pkg_path}")
            print()


def test_nvshmem_access():
    """Test if NVSHMEM can be imported and used."""
    
    print("=" * 60)
    print("Testing NVSHMEM Access")
    print("=" * 60)
    print()
    
    try:
        import nvshmem.core
        print("✓ nvshmem.core can be imported")
    except ImportError as e:
        print(f"✗ Cannot import nvshmem.core: {e}")
        return False
    
    try:
        import nvshmem.bindings.nvshmem
        print("✓ nvshmem.bindings.nvshmem can be imported")
    except ImportError as e:
        print(f"✗ Cannot import nvshmem.bindings: {e}")
        return False
    
    try:
        from mpi4py import MPI
        print("✓ mpi4py is available")
    except ImportError:
        print("⚠ mpi4py not found (needed for MPI-based initialization)")
        print("  Install with: pip install mpi4py")
    
    print()
    return True


def main():
    """Main function."""
    
    results = find_nvshmem_installation()
    generate_setup_commands(results)
    
    if results:
        test_nvshmem_access()
        
        print("=" * 60)
        print("Quick Start")
        print("=" * 60)
        print()
        print("To use these settings immediately:")
        print()
        print("1. Run this script and copy the export commands:")
        print("   python find_nvshmem.py")
        print()
        print("2. Paste them into your terminal:")
        print("   export NVSHMEM_HOME=...")
        print("   export LD_LIBRARY_PATH=...")
        print()
        print("3. Or add to ~/.bashrc for permanent setup:")
        print("   echo 'export NVSHMEM_HOME=...' >> ~/.bashrc")
        print("   echo 'export LD_LIBRARY_PATH=...' >> ~/.bashrc")
        print("   source ~/.bashrc")
        print()


if __name__ == "__main__":
    main()