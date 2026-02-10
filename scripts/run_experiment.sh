#!/bin/bash

################################################################################
# Experiment Execution Script
# 
# This script runs the complete Dual-Adapter Federated Learning experiment
# pipeline on a remote server.
#
# Usage:
#   bash scripts/run_experiment.sh [--skip-validation] [--config CONFIG_PATH]
#
# The script will:
#   1. Activate the virtual environment
#   2. Validate data (optional)
#   3. Run the training experiment
#   4. Handle errors and logging
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

# Default values
SKIP_VALIDATION=false
CONFIG_PATH="experiments/exp001_dual_adapter_fl/config.yaml"
VENV_DIR="venv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-validation    Skip data validation step"
            echo "  --config PATH        Path to config file (default: experiments/exp001_dual_adapter_fl/config.yaml)"
            echo "  -h, --help           Show this help message"
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
# Step 1: Check Environment
################################################################################

log_info "=== Checking Environment ==="

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    log_error "Please run scripts/setup_remote.sh first"
    exit 1
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate" || {
    log_error "Failed to activate virtual environment"
    exit 1
}

log_success "Virtual environment activated"

# Check HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    log_warning "HF_TOKEN environment variable is not set"
    log_warning "Model download may fail if the model requires authentication"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Exiting. Please set HF_TOKEN and try again:"
        echo "  export HF_TOKEN='your_token_here'"
        exit 1
    fi
else
    log_success "HF_TOKEN is set"
fi

# Check GPU availability
log_info "Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
else:
    print("⚠ WARNING: No GPU detected. Training will be very slow on CPU.")
EOF

echo ""

################################################################################
# Step 2: Validate Data (Optional)
################################################################################

if [ "$SKIP_VALIDATION" = false ]; then
    log_info "=== Validating Data ==="
    
    if [ ! -f "scripts/validate_data.py" ]; then
        log_warning "Data validation script not found. Skipping validation."
    else
        python3 scripts/validate_data.py || {
            log_error "Data validation failed"
            echo ""
            read -p "Do you want to continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Exiting. Please fix data issues and try again."
                exit 1
            fi
        }
    fi
    
    echo ""
else
    log_info "Skipping data validation (--skip-validation flag set)"
    echo ""
fi

################################################################################
# Step 3: Check Config File
################################################################################

log_info "=== Checking Configuration ==="

if [ ! -f "$CONFIG_PATH" ]; then
    log_error "Config file not found: $CONFIG_PATH"
    exit 1
fi

log_success "Config file found: $CONFIG_PATH"
echo ""

################################################################################
# Step 4: Create Results Directory
################################################################################

log_info "=== Preparing Output Directory ==="

# Extract output directory from config
OUTPUT_DIR=$(python3 -c "
import yaml
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('experiment', {}).get('output_dir', 'results/exp001_dual_adapter_fl'))
" 2>/dev/null || echo "results/exp001_dual_adapter_fl")

log_info "Output directory: $OUTPUT_DIR"

if [ -d "$OUTPUT_DIR" ]; then
    log_warning "Output directory already exists"
    log_warning "Existing results may be overwritten"
else
    mkdir -p "$OUTPUT_DIR"
    log_success "Created output directory"
fi

# Create subdirectories
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/metrics"

echo ""

################################################################################
# Step 5: Run Training
################################################################################

log_info "=== Starting Training ==="
echo ""

TRAIN_SCRIPT="experiments/exp001_dual_adapter_fl/train.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    log_error "Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

# Set up logging
LOG_FILE="$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

log_info "Training log will be saved to: $LOG_FILE"
log_info "Starting training... (this may take several hours)"
echo ""

# Run training with output to both console and log file
python3 "$TRAIN_SCRIPT" --config "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log_success "=== Training Completed Successfully ==="
    echo ""
    log_info "Results saved to: $OUTPUT_DIR"
    log_info "Training log: $LOG_FILE"
    echo ""
    log_info "Next steps:"
    echo "  1. Review training logs: cat $LOG_FILE"
    echo "  2. Run evaluation: python experiments/exp001_dual_adapter_fl/eval.py"
    echo "  3. Check results: ls -lh $OUTPUT_DIR"
    echo ""
else
    log_error "=== Training Failed ==="
    echo ""
    log_error "Training exited with code: $TRAIN_EXIT_CODE"
    log_error "Check the log file for details: $LOG_FILE"
    echo ""
    log_info "Common issues:"
    echo "  - Out of memory: Try reducing batch_size in config.yaml"
    echo "  - Model download failed: Check HF_TOKEN and internet connection"
    echo "  - Data not found: Run scripts/validate_data.py"
    echo ""
    exit $TRAIN_EXIT_CODE
fi

################################################################################
# Step 6: Summary
################################################################################

log_info "=== Experiment Summary ==="
echo ""

# Count checkpoints
if [ -d "$OUTPUT_DIR/checkpoints" ]; then
    CHECKPOINT_COUNT=$(find "$OUTPUT_DIR/checkpoints" -type d -name "round_*" | wc -l)
    log_info "Checkpoints saved: $CHECKPOINT_COUNT rounds"
fi

# Show disk usage
if [ -d "$OUTPUT_DIR" ]; then
    DISK_USAGE=$(du -sh "$OUTPUT_DIR" | cut -f1)
    log_info "Total disk usage: $DISK_USAGE"
fi

echo ""
log_success "Experiment execution complete!"
echo ""
