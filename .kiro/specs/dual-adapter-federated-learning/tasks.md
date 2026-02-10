# Tasks: Dual-Adapter Federated Learning System

## Phase 1: Core Infrastructure Setup

### 1.1 Project Structure Setup
- [x] Create directory structure following architecture.md guidelines
  - [x] Create `src/` with subdirectories: models, federated, data, utils
  - [x] Create `tools/` with subdirectories: runners, evaluators, visualizers
  - [x] Create `experiments/exp001_dual_adapter_fl/`
  - [x] Ensure `results/` is in .gitignore (will be created on remote)

### 1.2 Dependencies and Environment
- [x] Create `requirements.txt` with all necessary dependencies
  - [x] PyTorch, transformers, peft, bitsandbytes, accelerate
  - [x] scipy, pyyaml, tqdm
  - [x] matplotlib, seaborn (for visualization)
- [x] Create environment setup script for remote server
- [x] Document hardware requirements and HuggingFace token setup

### 1.3 Configuration Management
- [x] Implement `src/utils/config.py`
  - [x] `load_config()` function for YAML loading
  - [x] `merge_config_with_args()` for CLI override
  - [x] `validate_config()` for config validation
  - [x] `get_adaptive_quantization_config()` for automatic VRAM detection and quantization decision
- [x] Create default config template `experiments/exp001_dual_adapter_fl/config.yaml`

### 1.4 Logging System
- [x] Implement `src/utils/logger.py`
  - [x] `setup_logger()` with file and console handlers
  - [x] `log_metrics()` for structured metric logging
  - [x] Support for different log levels
- [x] Ensure logs are saved to results directory

## Phase 2: Model Components

### 2.1 Base Model Management
- [x] Implement `src/models/base_model.py`
  - [x] `get_adaptive_quantization_config()` to auto-detect GPU VRAM and decide quantization
    - [x] If VRAM < 16GB: Enable 4-bit quantization, log warning
    - [x] If VRAM >= 16GB: Use bfloat16 native precision, log info
  - [x] `load_base_model()` with automatic download from HuggingFace Hub
  - [x] Support for HuggingFace authentication token (via environment variable or config)
  - [x] `freeze_base_model()` to freeze parameters
  - [x] Proper error handling for download failures
- [x] Document model download process and cache location
- [x] Add retry logic for network failures

### 2.2 Dual-Adapter Architecture
- [x] Implement `src/models/dual_adapter.py`
  - [x] `DualAdapterModel` class
  - [x] `add_global_adapter()` method
  - [x] `add_local_adapter()` method
  - [x] `set_active_adapters()` method
  - [x] `save_adapter()` and `load_adapter()` methods
- [x] Define LoRA configuration (r=16, alpha=32, target_modules="all-linear")
- [x] Implement adapter state management

## Phase 3: Data Processing

### 3.1 Dataset Implementation
- [x] Implement `src/data/dataset.py`
  - [x] `AlpacaDataset` class with Qwen Chat Template support
  - [x] `apply_qwen_chat_template()` function to wrap data in <|im_start|> format
  - [x] Support for system prompt injection (for city identity)
  - [x] Tokenization with max_length=1024
  - [x] Proper padding and truncation

### 3.2 Data Preprocessing
- [x] Implement `src/data/preprocessor.py`
  - [x] `load_json_data()` function
  - [x] `merge_datasets()` for combining global and local data
  - [x] `validate_data_format()` for Alpaca format validation

### 3.3 Test Data Preparation
- [x] Document expected test data format in README
  - [x] Test-G format: `data/test/global_test.json`
  - [x] Test-A format: `data/test/strict_test.json`
  - [x] Test-B format: `data/test/service_test.json`
  - [x] Conflict cases format: `data/test/conflict_cases.json`
- [x] Add data validation checks that run before training
- [x] Note: Data is already in standard Alpaca format (input field is empty), no cleaning needed

## Phase 4: Federated Learning Core

### 4.1 Client Trainer
- [x] Implement `src/federated/client.py`
  - [x] `ClientTrainer` class
  - [x] `train_round()` method
  - [x] `_setup_model()` helper
  - [x] `_prepare_data()` helper
  - [x] `_train()` helper with HuggingFace Trainer
- [x] Configure training arguments (epochs=2, lr=2e-4, batch_size=4)

### 4.2 Server Aggregator
- [x] Implement `src/federated/server.py`
  - [x] `FederatedServer` class
  - [x] `aggregate_global_adapters()` method
  - [x] `distribute_global_adapter()` method

### 4.3 Aggregation Algorithms
- [x] Implement `src/federated/aggregators.py`
  - [x] `fedavg()` function for FedAvg algorithm
  - [x] `weighted_average()` helper function

## Phase 5: Experiment Orchestration

### 5.1 Federated Learning Runner
- [x] Implement `tools/runners/fl_runner.py`
  - [x] `FLRunner` class
  - [x] `run_experiment()` method for full experiment
  - [x] `run_round()` method for single round
  - [x] `_train_clients()` helper
  - [x] `_aggregate_global()` helper
  - [x] `_save_checkpoints()` helper

### 5.2 Baseline Trainers
- [x] Implement `tools/runners/baseline_trainer.py`
  - [x] `train_local_only()` for Local Only baseline
  - [x] `train_standard_fedavg()` for Standard FedAvg baseline

## Phase 6: Evaluation and Testing

### 6.1 Conflict Tester
- [x] Implement `tools/evaluators/conflict_tester.py`
  - [x] `ConflictTester` class
  - [x] `test_conflict_case()` method with system prompt injection support
  - [x] Automatic city identity prompt injection when loading different adapters
    - [x] Strict adapter: "你是上海市公安局的政务助手，请根据上海市的政策回答问题。"
    - [x] Service adapter: "你是石家庄市公安局的政务助手，请根据石家庄市的政策回答问题。"
  - [x] `run_test_suite()` method
- [x] Define conflict test case format

### 6.2 Metrics Calculator
- [x] Implement `tools/evaluators/metrics.py`
  - [x] `calculate_accuracy()` function
  - [ ] `calculate_perplexity()` function (optional)
  - [x] `calculate_conflict_resolution_rate()` function

### 6.3 Experiment Evaluator
- [x] Implement `tools/evaluators/experiment_evaluator.py`
  - [x] `evaluate_method()` function for single method
  - [x] `compare_methods()` function for comparison
  - [x] Support for Test-G, Test-A, Test-B evaluation

## Phase 7: Visualization and Reporting

### 7.1 Visualization Tools
- [~] Implement `tools/visualizers/plot_results.py`
  - [x] `plot_training_curve()` for loss curves
  - [x] `plot_accuracy_comparison()` for bar charts
  - [x] `plot_conflict_examples()` for case studies

### 7.2 Report Generator
- [~] Implement `tools/visualizers/report_generator.py`
  - [x] `generate_report()` function
  - [x] Markdown report template
  - [x] JSON results export
  - [x] Include all required sections (overview, methods, results, conclusion)

## Phase 8: Experiment 1 Implementation

### 8.1 Experiment Entry Points
- [x] Create `experiments/exp001_dual_adapter_fl/train.py`
  - [x] Load configuration
  - [x] Initialize FLRunner
  - [x] Execute training
  - [x] Save checkpoints
- [x] Create `experiments/exp001_dual_adapter_fl/eval.py`
  - [x] Load trained models
  - [x] Run evaluation
  - [x] Generate reports
- [x] Create `experiments/exp001_dual_adapter_fl/run_experiment.py`
  - [x] Complete pipeline: train all methods → evaluate → report

### 8.2 Experiment Configuration
- [x] Finalize `experiments/exp001_dual_adapter_fl/config.yaml`
  - [x] Model configuration (Qwen2.5-7B, 4-bit quantization)
  - [x] Training hyperparameters (lr=2e-4, epochs=2, batch_size=4)
  - [x] Federated settings (5 rounds, 2 clients)
  - [x] Data paths (training and test data)
  - [x] Output paths for results

### 8.3 Experiment Documentation
- [x] Create `experiments/exp001_dual_adapter_fl/README.md`
  - [x] Experiment objectives
  - [x] Data requirements
  - [x] How to run on remote server
  - [x] Expected results
- [x] Create setup script for remote server
  - [x] Environment setup
  - [x] HuggingFace token configuration
  - [x] Data validation checks

## Phase 9: Remote Deployment Preparation

### 9.1 Deployment Scripts
- [~] Create `scripts/setup_remote.sh`
  - [x] Install system dependencies
  - [x] Create Python virtual environment
  - [x] Install Python packages from requirements.txt
  - [x] Set up HuggingFace token
- [~] Create `scripts/run_experiment.sh`
  - [x] Activate environment
  - [x] Run complete experiment pipeline
  - [x] Handle errors and logging

### 9.2 Data Validation
- [~] Create `scripts/validate_data.py`
  - [x] Check training data exists and is valid
  - [x] Check test data exists and is valid
  - [x] Verify data format compliance
  - [x] Report missing or invalid data

### 9.3 Documentation
- [x] Create comprehensive README.md
  - [x] Project overview
  - [x] Remote server requirements
  - [x] Installation instructions
  - [x] How to run experiments
  - [x] Expected outputs
- [~] Create DEPLOYMENT.md
  - [x] Step-by-step deployment guide
  - [x] HuggingFace token setup
  - [x] Troubleshooting common issues
  - [x] GPU memory requirements

## Phase 10: Final Verification

### 10.1 Code Review
- [x] Review all code for clarity and correctness
- [x] Add docstrings to all functions and classes
- [x] Add inline comments for complex logic
- [x] Ensure consistent code style

### 10.2 Pre-deployment Checklist
- [x] Verify all required files are present
- [x] Check .gitignore excludes results/ and cache/
- [x] Ensure requirements.txt is complete
- [x] Verify config.yaml has correct paths
- [x] Test data validation script locally

### 10.3 Documentation Review
- [x] Review README.md for completeness
- [x] Review DEPLOYMENT.md for accuracy
- [x] Ensure all scripts have usage instructions
- [x] Document expected runtime and resource usage

## Notes for Remote Execution

### Critical Requirements
1. **HuggingFace Token**: Must be set as environment variable `HF_TOKEN` or in config
2. **Test Data**: Must exist in `data/test/` before running evaluation
3. **GPU Memory**: Minimum 24GB recommended, 8GB possible with 4-bit quantization
4. **Network**: Stable connection required for model download (~14GB for Qwen2.5-7B)

### Execution Flow
1. Upload code to remote server
2. Run `scripts/setup_remote.sh` to set up environment
3. Run `scripts/validate_data.py` to check data
4. Run `scripts/run_experiment.sh` to execute full pipeline
5. Download results from `results/exp001_dual_adapter_fl/`

### Expected Outputs
- `results/exp001_dual_adapter_fl/checkpoints/` - Model checkpoints
- `results/exp001_dual_adapter_fl/logs/` - Training logs
- `results/exp001_dual_adapter_fl/report/` - Experiment report and visualizations
