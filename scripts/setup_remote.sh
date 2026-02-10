#!/bin/bash

################################################################################
# Remote Server Environment Setup Script
# 
# This script automates the setup of the Python environment for the
# Dual-Adapter Federated Learning project on a remote server.
#
# Usage:
#   bash scripts/setup_remote.sh [--token YOUR_HF_TOKEN]
#
# Requirements:
#   - Ubuntu/Debian-based system (or compatible)
#   - NVIDIA GPU with CUDA support
#   - Python 3.10 or higher
#   - Internet connection for downloading packages and models
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
HF_TOKEN=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--token YOUR_HF_TOKEN]"
            echo ""
            echo "Options:"
            echo "  --token TOKEN    HuggingFace authentication token"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Step 1: System Information
################################################################################

log_info "=== System Information ==="
echo "Hostname: $(hostname)"
echo "OS: $(uname -s) $(uname -r)"
echo "Python version: $(python3 --version 2>/dev/null || echo 'Not found')"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log_warning "nvidia-smi not found. GPU may not be available."
fi

echo ""

################################################################################
# Step 2: Check System Dependencies
################################################################################

log_info "=== Checking System Dependencies ==="

# Check Python version
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    log_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

log_success "Python $PYTHON_VERSION detected"

# Check for pip
if ! command -v pip3 &> /dev/null; then
    log_warning "pip3 not found. Installing pip..."
    python3 -m ensurepip --upgrade || {
        log_error "Failed to install pip"
        exit 1
    }
fi

log_success "pip3 is available"

# Check for venv module
if ! python3 -c "import venv" &> /dev/null; then
    log_warning "venv module not found. Installing python3-venv..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3-venv
    else
        log_error "Cannot install python3-venv automatically. Please install it manually."
        exit 1
    fi
fi

log_success "venv module is available"

echo ""

################################################################################
# Step 3: Create Virtual Environment
################################################################################

log_info "=== Creating Virtual Environment ==="

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    log_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        log_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR" || {
        log_error "Failed to create virtual environment"
        exit 1
    }
    log_success "Virtual environment created"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate" || {
    log_error "Failed to activate virtual environment"
    exit 1
}

log_success "Virtual environment activated"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel || {
    log_error "Failed to upgrade pip"
    exit 1
}

echo ""

################################################################################
# Step 4: Install Python Dependencies
################################################################################

log_info "=== Installing Python Dependencies ==="

if [ ! -f "requirements.txt" ]; then
    log_error "requirements.txt not found in current directory"
    exit 1
fi

log_info "Installing packages from requirements.txt..."
pip install -r requirements.txt || {
    log_error "Failed to install dependencies"
    exit 1
}

log_success "All dependencies installed successfully"

echo ""

################################################################################
# Step 5: Verify PyTorch and CUDA
################################################################################

log_info "=== Verifying PyTorch and CUDA ==="

python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: CUDA is not available. Training will be very slow on CPU.")
EOF

echo ""

################################################################################
# Step 6: Configure HuggingFace Token
################################################################################

log_info "=== Configuring HuggingFace Token ==="

# Check if token is provided via command line
if [ -n "$HF_TOKEN" ]; then
    log_info "Using HuggingFace token from command line argument"
    export HF_TOKEN="$HF_TOKEN"
    echo "export HF_TOKEN=\"$HF_TOKEN\"" >> "$VENV_DIR/bin/activate"
    log_success "HuggingFace token configured"
# Check if token is already in environment
elif [ -n "$HF_TOKEN" ]; then
    log_success "HuggingFace token already set in environment"
# Check if token is in .env file
elif [ -f ".env" ] && grep -q "HF_TOKEN" .env; then
    log_info "Loading HuggingFace token from .env file"
    export $(grep HF_TOKEN .env | xargs)
    log_success "HuggingFace token loaded from .env"
else
    log_warning "HuggingFace token not found"
    echo ""
    echo "To download Qwen2.5-7B-Instruct, you need a HuggingFace token."
    echo "You can:"
    echo "  1. Set it now (will be saved to venv/bin/activate)"
    echo "  2. Set it later via: export HF_TOKEN='your_token'"
    echo "  3. Create a .env file with: HF_TOKEN=your_token"
    echo ""
    read -p "Enter your HuggingFace token (or press Enter to skip): " token_input
    
    if [ -n "$token_input" ]; then
        export HF_TOKEN="$token_input"
        echo "export HF_TOKEN=\"$token_input\"" >> "$VENV_DIR/bin/activate"
        log_success "HuggingFace token configured"
    else
        log_warning "Skipping HuggingFace token configuration"
        log_warning "You will need to set HF_TOKEN before running experiments"
    fi
fi

# Test HuggingFace authentication
if [ -n "$HF_TOKEN" ]; then
    log_info "Testing HuggingFace authentication..."
    python3 << EOF
from huggingface_hub import HfApi
import os

try:
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    user = api.whoami()
    print(f"✓ Authenticated as: {user['name']}")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
    print("Please check your token and try again")
EOF
fi

echo ""

################################################################################
# Step 7: Create Results Directory
################################################################################

log_info "=== Creating Results Directory ==="

if [ ! -d "results" ]; then
    mkdir -p results
    log_success "Created results/ directory"
else
    log_info "results/ directory already exists"
fi

echo ""

################################################################################
# Step 8: Verify Installation
################################################################################

log_info "=== Verifying Installation ==="

log_info "Checking installed packages..."
python3 << EOF
import sys

packages = [
    'torch',
    'transformers',
    'peft',
    'bitsandbytes',
    'accelerate',
    'scipy',
    'yaml',
    'tqdm',
    'matplotlib',
    'seaborn'
]

all_ok = True
for pkg in packages:
    try:
        if pkg == 'yaml':
            __import__('yaml')
        else:
            __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg} - NOT FOUND")
        all_ok = False

if all_ok:
    print("\n✓ All required packages are installed")
    sys.exit(0)
else:
    print("\n✗ Some packages are missing")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    log_success "All packages verified"
else
    log_error "Some packages are missing. Please check the output above."
    exit 1
fi

echo ""

################################################################################
# Step 9: Display Summary
################################################################################

log_success "=== Setup Complete ==="
echo ""
echo "Environment setup completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Verify your training data exists:"
echo "     python scripts/validate_data.py"
echo ""
echo "  3. Run the experiment:"
echo "     python experiments/exp001_dual_adapter_fl/train.py"
echo ""
echo "  4. Or run the complete pipeline:"
echo "     bash scripts/run_experiment.sh"
echo ""

if [ -z "$HF_TOKEN" ]; then
    log_warning "Remember to set your HuggingFace token before running experiments:"
    echo "  export HF_TOKEN='your_token_here'"
    echo ""
fi

log_info "For more information, see README.md and DEPLOYMENT.md"
echo ""
