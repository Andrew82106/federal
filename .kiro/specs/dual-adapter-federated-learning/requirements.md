# Requirements Document: Dual-Adapter Federated Learning System

## Introduction

This document specifies the requirements for a dual-adapter federated learning system designed for public security governance scenarios. The system addresses the "条块分割" (vertical-horizontal division) challenge in Chinese administrative systems, where national laws (条) must coexist with diverse local policies (块). The system uses a novel dual-adapter architecture with LoRA (Low-Rank Adaptation) to enable privacy-preserving federated learning while maintaining the ability to handle semantic conflicts between different jurisdictions.

## Glossary

- **Base_Model**: The frozen Qwen2.5-7B-Instruct foundation model
- **Global_Adapter**: LoRA adapter that learns universal laws and participates in federated aggregation (条适配器)
- **Local_Adapter**: LoRA adapter that learns jurisdiction-specific policies and remains private (块适配器)
- **Client**: A simulated local jurisdiction (city) that trains on its own data
- **Server**: The federated learning coordinator that aggregates global adapters
- **FedAvg**: Federated Averaging algorithm for aggregating model parameters
- **LoRA**: Low-Rank Adaptation, a parameter-efficient fine-tuning method
- **Round**: One complete cycle of client training and server aggregation
- **Conflict_Resolution**: The system's ability to provide different answers based on local context

## Requirements

### Requirement 1: Base Model Management

**User Story:** As a system developer, I want to load and configure the base language model with memory optimization, so that the system can run on remote GPU servers.

#### Acceptance Criteria

1. WHEN the system initializes, THE Base_Model SHALL automatically download and load Qwen2.5-7B-Instruct from HuggingFace Hub
2. WHEN GPU memory is less than 20GB, THE Base_Model SHALL apply 4-bit quantization using bitsandbytes
3. WHEN the Base_Model is loaded, THE System SHALL freeze all base model parameters to prevent modification
4. WHEN loading fails, THE System SHALL return a descriptive error message with memory requirements and download status
5. THE System SHALL cache downloaded models in HuggingFace default cache directory for reuse

### Requirement 2: Dual-Adapter Architecture

**User Story:** As a system architect, I want to implement a dual-adapter architecture with separate global and local LoRA adapters, so that the system can learn both universal laws and jurisdiction-specific policies.

#### Acceptance Criteria

1. THE System SHALL support simultaneous attachment of Global_Adapter and Local_Adapter to the Base_Model
2. WHEN creating adapters, THE System SHALL configure LoRA with rank=16, alpha=32, and target modules ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
3. WHEN training, THE System SHALL enable gradient updates for both Global_Adapter and Local_Adapter
4. WHEN saving adapters, THE System SHALL store Global_Adapter and Local_Adapter weights separately
5. WHEN loading adapters, THE System SHALL support loading either adapter independently or both simultaneously

### Requirement 3: Data Processing

**User Story:** As a data engineer, I want to process training data in Alpaca format with proper tokenization, so that the model can learn from structured instruction-response pairs.

#### Acceptance Criteria

1. WHEN loading training data, THE System SHALL parse JSON files in Alpaca format with fields "instruction", "input", and "output"
2. WHEN tokenizing data, THE System SHALL use the Qwen tokenizer with maximum sequence length of 1024 tokens
3. WHEN a sequence exceeds 1024 tokens, THE System SHALL truncate the sequence and log a warning
4. WHEN creating batches, THE System SHALL apply padding to align sequences within each batch
5. THE System SHALL support loading three data sources: global training data, strict client data, and service client data

### Requirement 4: Client Training

**User Story:** As a federated learning researcher, I want each client to train on mixed data (global + local), so that the adapters learn both universal and jurisdiction-specific knowledge.

#### Acceptance Criteria

1. WHEN a client begins training, THE Client SHALL load the latest Global_Adapter weights from the Server
2. WHEN a client begins training, THE Client SHALL load or initialize its Local_Adapter weights from local storage
3. WHEN Client A trains, THE Client SHALL use training data composed of global_train.json + client_strict.json
4. WHEN Client B trains, THE Client SHALL use training data composed of global_train.json + client_service.json
5. WHEN training completes, THE Client SHALL save both Global_Adapter and Local_Adapter weights to local storage
6. WHEN training for a round, THE Client SHALL train for exactly 2 epochs with learning rate 2e-4
7. WHEN training, THE Client SHALL use batch size 4 or 8 depending on available GPU memory
8. WHEN training, THE Client SHALL apply gradient accumulation with 4 steps

### Requirement 5: Server Aggregation

**User Story:** As a federated learning coordinator, I want to aggregate global adapters from all clients using FedAvg, so that the system learns universal knowledge while preserving local privacy.

#### Acceptance Criteria

1. WHEN a round completes, THE Server SHALL collect Global_Adapter weights from all clients
2. WHEN aggregating, THE Server SHALL apply FedAvg algorithm to compute the average of Global_Adapter weights
3. WHEN aggregating, THE Server SHALL NOT access or aggregate Local_Adapter weights
4. WHEN aggregation completes, THE Server SHALL save the new Global_Adapter weights for the next round
5. THE Server SHALL maintain a history of Global_Adapter weights for each round

### Requirement 6: Federated Learning Orchestration

**User Story:** As an experiment coordinator, I want to execute multiple rounds of federated learning with proper checkpointing, so that the system progressively improves through collaboration.

#### Acceptance Criteria

1. WHEN starting an experiment, THE System SHALL execute exactly 5 federated learning rounds
2. WHEN executing a round, THE System SHALL train Client A (strict city) followed by Client B (service city) in serial
3. WHEN a round completes, THE System SHALL perform server aggregation before starting the next round
4. WHEN each round completes, THE System SHALL save checkpoints for all adapters (global, strict local, service local)
5. WHEN all rounds complete, THE System SHALL save final adapter weights to designated output directories

### Requirement 7: Inference and Conflict Resolution

**User Story:** As a system evaluator, I want to test the model with different local adapters activated, so that I can verify the system correctly handles jurisdiction-specific policies.

#### Acceptance Criteria

1. WHEN performing inference, THE System SHALL load Base_Model with specified adapter combination
2. WHEN testing with Global_Adapter + Strict_Local_Adapter, THE System SHALL generate responses based on strict city policies
3. WHEN testing with Global_Adapter + Service_Local_Adapter, THE System SHALL generate responses based on service city policies
4. WHEN given the same question with different local adapters, THE System SHALL produce different answers reflecting local policies
5. WHEN testing universal law questions, THE System SHALL produce consistent answers regardless of which local adapter is active

### Requirement 8: Configuration Management

**User Story:** As an experiment designer, I want to configure all hyperparameters through YAML files, so that I can easily modify and track experimental settings.

#### Acceptance Criteria

1. THE System SHALL load configuration from YAML files with sections for model, training, federated, and data parameters
2. WHEN configuration is loaded, THE System SHALL validate all required fields are present
3. WHEN command-line arguments are provided, THE System SHALL override corresponding configuration file values
4. WHEN configuration is invalid, THE System SHALL return descriptive error messages indicating missing or incorrect fields
5. THE System SHALL log the complete configuration at the start of each experiment

### Requirement 9: Logging and Monitoring

**User Story:** As a researcher, I want comprehensive logging of training metrics and system events, so that I can analyze experiment progress and debug issues.

#### Acceptance Criteria

1. WHEN training begins, THE System SHALL create log files in the designated output directory
2. WHEN each training step completes, THE System SHALL log loss, learning rate, and step number
3. WHEN each round completes, THE System SHALL log round number, client metrics, and aggregation results
4. WHEN errors occur, THE System SHALL log error messages with timestamps and stack traces
5. THE System SHALL save training metrics in JSON format for post-experiment analysis

### Requirement 10: Memory Optimization

**User Story:** As a system operator, I want automatic memory management and optimization, so that the system can run efficiently on available hardware.

#### Acceptance Criteria

1. WHEN GPU memory is limited, THE System SHALL enable 4-bit quantization for the Base_Model
2. WHEN training between rounds, THE System SHALL clear CUDA cache to free memory
3. WHEN training, THE System SHALL use gradient accumulation to achieve larger effective batch sizes
4. WHEN memory errors occur, THE System SHALL provide recommendations for reducing memory usage
5. THE System SHALL log GPU memory usage at the start and end of each training round

### Requirement 11: Checkpoint Management

**User Story:** As an experiment manager, I want systematic checkpoint saving and loading, so that I can resume experiments and analyze intermediate results.

#### Acceptance Criteria

1. WHEN each round completes, THE System SHALL save checkpoints in directories named by round number
2. WHEN saving checkpoints, THE System SHALL store Global_Adapter, Strict_Local_Adapter, and Service_Local_Adapter separately
3. WHEN loading checkpoints, THE System SHALL verify checkpoint integrity before loading weights
4. WHEN final training completes, THE System SHALL save final adapters in a designated "final_adapters" directory
5. THE System SHALL maintain a checkpoint manifest file listing all saved checkpoints with metadata

### Requirement 12: Evaluation and Testing

**User Story:** As a quality assurance engineer, I want automated conflict testing and accuracy evaluation, so that I can verify the system meets correctness requirements.

#### Acceptance Criteria

1. WHEN evaluation runs, THE System SHALL execute predefined conflict test cases with different adapter combinations
2. WHEN testing conflict resolution, THE System SHALL verify that the same question produces different answers with different local adapters
3. WHEN calculating accuracy, THE System SHALL compute separate metrics for global laws, strict policies, and service policies
4. WHEN evaluation completes, THE System SHALL generate a report with test results and accuracy metrics
5. THE System SHALL support custom test cases provided in JSON format

### Requirement 13: Experiment 1 - Complete Training and Evaluation Pipeline

**User Story:** As a researcher, I want to execute a complete experimental pipeline that trains the dual-adapter federated model and evaluates its performance with publishable results, so that I can demonstrate the effectiveness of the approach in my paper.

#### Acceptance Criteria

1. WHEN Experiment 1 starts, THE System SHALL execute the complete federated learning training for 5 rounds with 2 clients (strict and service)
2. WHEN training completes, THE System SHALL automatically run comprehensive evaluation including:
   - Conflict resolution tests (same question, different local adapters produce different answers)
   - Accuracy tests on three test sets (Test-G for global laws, Test-A for strict policies, Test-B for service policies)
   - Baseline comparisons (Local Only, Standard FedAvg, Dual-Adapter)
3. WHEN evaluation completes, THE System SHALL generate a results report containing:
   - Performance comparison table with accuracy metrics for all three methods
   - Training curves showing loss and accuracy over rounds
   - Conflict test case results with example outputs
   - Statistical analysis of results
4. WHEN generating the report, THE System SHALL save results in both human-readable (Markdown) and machine-readable (JSON) formats
5. WHEN the experiment finishes, THE System SHALL save all artifacts including:
   - Final adapter checkpoints
   - Training logs
   - Evaluation metrics
   - Generated report
   - Visualization plots
6. THE System SHALL provide a single command to run the complete Experiment 1 pipeline from start to finish
7. THE Results SHALL be reproducible by setting random seeds and logging all hyperparameters
