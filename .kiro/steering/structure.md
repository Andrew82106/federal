# Project Structure

## Architecture Principle
This project follows a strict separation of concerns for experimental research:

1. **Tools** - Reusable utilities and frameworks for running experiments
2. **Source Code** - Core implementation and models
3. **Experiments** - Experiment definitions and configurations
4. **Results** - Experimental outputs, data, and analysis

## Directory Structure

```
policeModel/
├── data/                           # Source data (version controlled)
│   ├── files/                      # Raw PDF documents (laws and regulations)
│   └── rule_data/                  # Training datasets in JSON format
│       ├── global_train.json       # National laws (条)
│       ├── client_strict.json      # Strict city policies (块-严管)
│       └── client_service.json     # Service-oriented city policies (块-服务)
├── src/                            # Core implementation code
│   ├── models/                     # Model wrappers and architectures
│   │   ├── __init__.py
│   │   ├── base_model.py           # Base model loader (Qwen2.5-7B)
│   │   └── dual_adapter.py         # Dual-adapter architecture
│   ├── federated/                  # Federated learning core algorithms
│   │   ├── __init__.py
│   │   ├── client.py               # Client training logic
│   │   ├── server.py               # Server aggregation logic
│   │   └── aggregators.py          # FedAvg and other aggregation methods
│   ├── data/                       # Data processing utilities
│   │   ├── __init__.py
│   │   ├── dataset.py              # Dataset classes for Alpaca format
│   │   └── preprocessor.py         # Data preprocessing utilities
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       └── logger.py               # Logging utilities
├── tools/                          # Experimental tools (reusable)
│   ├── runners/                    # Experiment execution frameworks
│   │   ├── __init__.py
│   │   └── fl_runner.py            # Federated learning runner
│   ├── evaluators/                 # Evaluation tools
│   │   ├── __init__.py
│   │   ├── conflict_tester.py      # Test conflict resolution
│   │   └── metrics.py              # Accuracy and performance metrics
│   └── visualizers/                # Visualization utilities
│       ├── __init__.py
│       └── plot_results.py         # Plot training curves and comparisons
├── experiments/                    # Experiment definitions
│   └── exp001_dual_adapter_fl/     # Dual-adapter federated learning
│       ├── README.md               # Experiment documentation
│       ├── config.yaml             # Hyperparameters and settings
│       ├── train.py                # Training entry point
│       └── eval.py                 # Evaluation script
├── results/                        # Experimental outputs (gitignored)
│   └── exp001_dual_adapter_fl/     # Results for experiment 001
│       ├── checkpoints/            # Model checkpoints by round
│       │   ├── round_1/
│       │   ├── round_2/
│       │   └── final_adapters/
│       │       ├── global/         # Global adapter (条)
│       │       ├── strict/         # Local adapter for strict city
│       │       └── service/        # Local adapter for service city
│       ├── logs/                   # Training logs
│       ├── metrics/                # Performance metrics
│       └── report.md               # Experiment report
├── docs/                           # Documentation
│   ├── experimentPlan.md           # Detailed experiment plan
│   ├── plan.md                     # Research proposal
│   └── verify.md                   # Literature review
└── requirements.txt                # Python dependencies
```

## Key Guidelines

- **Never mix**: Keep tools, source code, experiments, and results in separate directories
- **Tools should be generic**: Tools in `tools/` should work with any experiment
- **Code should be pure**: Core code in `src/` should not contain experiment-specific logic
- **Experiments are self-contained**: Each experiment folder contains only config and entry scripts
- **Results are ephemeral**: Results can be regenerated from code and should be gitignored
- **One experiment, one folder**: Each experiment should have its own folder with all related files
- **Mirror structure**: Results folder structure should mirror experiments folder structure

## Component Responsibilities

### Source Code (`src/`)
**Purpose**: Reusable, experiment-agnostic implementations

- `models/`: Model architectures and wrappers (base model loader, dual-adapter)
- `federated/`: Core federated learning algorithms (client, server, aggregators)
- `data/`: Data processing utilities (dataset classes, preprocessors)
- `utils/`: Shared utilities (config management, logging)

**Rules**:
- No hardcoded paths or experiment-specific parameters
- All configurations should be passed as arguments
- Functions should be pure and testable
- No direct file I/O except through utility functions

### Tools (`tools/`)
**Purpose**: Generic frameworks for running and analyzing experiments

- `runners/`: Execution frameworks (FL runner, training orchestrator)
- `evaluators/`: Evaluation tools (conflict testing, metrics calculation)
- `visualizers/`: Visualization utilities (plotting, reporting)

**Rules**:
- Tools should accept configuration objects, not hardcoded values
- Should work with any experiment that follows the standard structure
- Focus on orchestration, not implementation
- Can depend on `src/` but not on specific experiments

### Experiments (`experiments/`)
**Purpose**: Experiment definitions and entry points

Each experiment folder (`expXXX_name/`) contains:
- `README.md`: Experiment documentation and objectives
- `config.yaml`: All hyperparameters and settings
- `train.py`: Training entry point (calls tools and src)
- `eval.py`: Evaluation entry point

**Rules**:
- Minimal code, mostly configuration and orchestration
- Import from `src/` and `tools/`, never duplicate code
- All parameters in config.yaml, not hardcoded
- Use sequential numbering (exp001, exp002, etc.)

### Data (`data/`)
**Purpose**: Source data for experiments

- `files/`: Raw documents (PDFs, original sources)
- `rule_data/`: Processed training data in JSON format

**Rules**:
- Version controlled (committed to git)
- Read-only during experiments
- Document data sources and preprocessing steps

### Results (`results/`)
**Purpose**: Experimental outputs

Structure mirrors `experiments/`:
- `checkpoints/`: Model weights by round
- `logs/`: Training and evaluation logs
- `metrics/`: Performance metrics (JSON, CSV)
- `report.md`: Experiment summary

**Rules**:
- Always gitignored
- Can be completely regenerated
- Organized by experiment name
