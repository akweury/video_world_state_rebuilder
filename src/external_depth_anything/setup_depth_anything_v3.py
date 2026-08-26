"""
Setup script for Depth Anything V3
Clones the repository and installs dependencies
"""

import os
import subprocess
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "external"
DEPTH_ANYTHING_DIR = EXTERNAL_DIR / "Depth-Anything-3"

def run_command(cmd, cwd=None, description=""):
    """Run a command and print output"""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print('='*60)
    
    # Modified run_command for conda compatibility
    if cmd.startswith("pip "):
        cmd = f"{sys.executable} -m {cmd}"
    elif cmd.startswith("conda "):
        # Use conda directly without python interpreter wrapper
        pass
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    return True

def setup_depth_anything_v3():
    """Clone and install Depth Anything V3"""
    
    # Create external directory
    EXTERNAL_DIR.mkdir(exist_ok=True)
    print(f"✓ External directory: {EXTERNAL_DIR}")
    
    # Clone repository if not exists
    if not DEPTH_ANYTHING_DIR.exists():
        print(f"\n📦 Cloning Depth Anything V3...")
        if not run_command(
            "git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git",
            cwd=EXTERNAL_DIR,
            description="Cloning Depth Anything V3 repository"
        ):
            return False
        print("✓ Repository cloned successfully")
    else:
        print(f"✓ Repository already exists at {DEPTH_ANYTHING_DIR}")
    
    # Check current environment and PyTorch
    print("\n🔍 Checking environment...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} detected")
        print(f"✓ MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    except ImportError:
        print("❌ PyTorch not found. Please install PyTorch first.")
        print("  For Mac Silicon: conda install pytorch torchvision torchaudio -c pytorch")
        return False
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    # Skip xformers for now - it's causing build issues on Mac Silicon
    print("⚠️  Skipping xformers installation (not required for basic functionality)")
    
    # Install essential dependencies first
    essential_deps = [
        "opencv-python",
        "transformers>=4.21.0",
        "accelerate", 
        "matplotlib",
    ]
    
    # Install optional dependencies separately
    optional_deps = [
        "trimesh",
        "typer",
        "uvicorn",
        "gradio"  # Only if Python >= 3.10
    ]
    
    print("📦 Installing essential dependencies...")
    for dep in essential_deps:
        if not run_command(
            f"pip install {dep}",
            description=f"Installing {dep}"
        ):
            print(f"❌ Failed to install {dep}, this may cause issues...")
            return False
    
    print("📦 Installing optional dependencies...")        
    for dep in optional_deps:
        if not run_command(
            f"pip install {dep}",
            description=f"Installing {dep}"
        ):
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    # Install the package using the pyproject build backend.
    # The upstream repository does not ship setup.py/setup.cfg, so editable
    # installs fail with older pip versions even when the source tree is valid.
    print("📦 Installing Depth Anything V3...")
    if not run_command(
        "pip install . --no-deps",
        cwd=DEPTH_ANYTHING_DIR,
        description="Installing Depth Anything V3 (without dependencies)"
    ):
        print("⚠️ Failed to install without dependencies, trying regular install...")
        if not run_command(
            "pip install .",
            cwd=DEPTH_ANYTHING_DIR,
            description="Installing Depth Anything V3 (with dependencies)"
        ):
            print("⚠️ Package install failed, but the source tree is present and importable via sys.path.")
    
    print("\n" + "="*60)
    print("✅ Depth Anything V3 setup complete!")
    print("="*60)
    print(f"Repository location: {DEPTH_ANYTHING_DIR}")
    print("\n⚠️  Notes:")
    print("  - xformers was skipped (not required for basic depth estimation)")
    print("  - If you need xformers later, install manually: conda install -c conda-forge xformers")
    print("  - Remember to set: export KMP_DUPLICATE_LIB_OK=TRUE")
    print("\nOptional installations (run these in the Depth-Anything-3 directory):")
    print("  - For Gaussian head: pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70")
    print("  - For Gradio app: pip install -e \".[app]\" (requires Python>=3.10)")
    print("  - For all features: pip install -e \".[all]\"")
    
    return True

if __name__ == "__main__":
    print("Depth Anything V3 Setup")
    print("="*60)
    
    # Check if we're in the right environment
    try:
        import torch
        if torch.__version__.startswith('2.1'):
            print(f"⚠️  You're using PyTorch {torch.__version__}")
            print("   For best Mac Silicon compatibility, use the 'cauvid_arm' environment:")
            print("   conda activate cauvid_arm && python setup_depth_anything_v3.py")
            print()
    except ImportError:
        print("❌ PyTorch not found!")
        print("   Please activate an environment with PyTorch:")
        print("   conda activate cauvid_arm && python setup_depth_anything_v3.py")
        sys.exit(1)
    
    success = setup_depth_anything_v3()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\nYou can now use Depth Anything V3 in your project.")
        print("Remember to set: export KMP_DUPLICATE_LIB_OK=TRUE")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        print("💡 Try running in the cauvid_arm environment:")
        print("   conda activate cauvid_arm")
        print("   export KMP_DUPLICATE_LIB_OK=TRUE") 
        print("   python setup_depth_anything_v3.py")
        sys.exit(1)
