---
inclusion: always
---

# Product Overview

## Project Name
policeModel - Dual-Adapter Federated Learning for Public Security Governance

## Purpose
An experimental research project exploring federated learning techniques for public security governance scenarios. The project addresses the "条块分割" (vertical-horizontal division) challenge in Chinese administrative systems, where national laws (条) must coexist with diverse local policies (块).

## Research Objective
Demonstrate that a **Dual-Adapter architecture** (Global Adapter + Local Adapter) outperforms traditional federated learning (FedAvg) in scenarios with:
- **Accuracy**: Correctly answer both national laws and local policies without confusion
- **Privacy**: Local policy data never leaves the local client
- **Conflict Resolution**: Handle semantic conflicts between different jurisdictions

## Core Innovation
**Dual-Adapter Federated Learning Architecture**:
1. **Global Adapter (条适配器)**: Learns universal laws, participates in federated aggregation
2. **Local Adapter (块适配器)**: Learns jurisdiction-specific policies, remains private

This architecture mirrors the administrative structure of Chinese public security systems, where:
- **条 (Vertical)**: Ministry-level unified laws and regulations
- **块 (Horizontal)**: City-level differentiated policies and implementation details

## Technical Approach
- **Base Model**: Qwen2.5-7B-Instruct (frozen parameters)
- **PEFT Method**: LoRA (Low-Rank Adaptation)
- **Federated Strategy**: Serial simulation with FedAvg aggregation
- **Conflict Handling**: Architectural separation prevents catastrophic forgetting

## Application Scenario
Public security governance with conflicting policies:
- **National Laws**: Traffic safety law, residence permit regulations (universal)
- **City A (Strict)**: Shanghai points-based settlement (7 years social security)
- **City B (Service)**: Shijiazhuang zero-threshold settlement (6 months social security)

The model must correctly answer "Can I settle without social security?" differently based on which local adapter is activated.

## Core Principles
- **Decoupling**: Strict separation between src/, tools/, experiments/, results/
- **Reproducibility**: Configuration-driven experiments with version control
- **Modularity**: Reusable components across different experiments
- **Traceability**: Clear connection between code, experiments, and results
- **Privacy-Preserving**: Local adapters never participate in aggregation
