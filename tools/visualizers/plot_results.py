"""
Results Visualization Tools

This module provides functions to visualize training curves, accuracy comparisons,
and conflict resolution examples for the Dual-Adapter Federated Learning project.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curve(
    metrics_path: str,
    output_path: str,
    metric_name: str = "loss",
    title: Optional[str] = None
):
    """
    Plot training curves (loss or accuracy over rounds/steps).
    
    Args:
        metrics_path: Path to training metrics JSON file
        output_path: Path to save the plot
        metric_name: Name of the metric to plot (e.g., "loss", "accuracy")
        title: Optional custom title for the plot
    """
    logger.info(f"Plotting training curve for {metric_name}...")
    
    # Load metrics
    if not os.path.exists(metrics_path):
        logger.error(f"Metrics file not found: {metrics_path}")
        return
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    # Extract data
    rounds = []
    values = []
    
    if isinstance(metrics, dict):
        # Format: {"round_1": {"loss": 0.5}, "round_2": {"loss": 0.4}, ...}
        for round_key in sorted(metrics.keys()):
            if round_key.startswith("round_"):
                round_num = int(round_key.split("_")[1])
                rounds.append(round_num)
                
                round_data = metrics[round_key]
                if metric_name in round_data:
                    values.append(round_data[metric_name])
                else:
                    logger.warning(f"Metric '{metric_name}' not found in {round_key}")
    elif isinstance(metrics, list):
        # Format: [{"round": 1, "loss": 0.5}, {"round": 2, "loss": 0.4}, ...]
        for item in metrics:
            if "round" in item and metric_name in item:
                rounds.append(item["round"])
                values.append(item[metric_name])
    
    if not rounds or not values:
        logger.error(f"No data found for metric '{metric_name}'")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rounds, values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    
    ax.set_xlabel('Federated Learning Round', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name.capitalize(), fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Training {metric_name.capitalize()} Over Rounds', 
                     fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training curve saved to {output_path}")
    
    plt.close()


def plot_accuracy_comparison(
    comparison_data: Dict,
    output_path: str,
    title: Optional[str] = None
):
    """
    Plot bar chart comparing accuracy across methods and test sets.
    
    Args:
        comparison_data: Dictionary containing comparison results from compare_methods()
        output_path: Path to save the plot
        title: Optional custom title for the plot
    """
    logger.info("Plotting accuracy comparison...")
    
    # Extract data
    test_sets = comparison_data.get("test_sets", {})
    methods = comparison_data.get("methods", [])
    
    if not test_sets or not methods:
        logger.error("No comparison data found")
        return
    
    # Prepare data for plotting
    test_names = list(test_sets.keys())
    method_names = [m["name"] for m in methods]
    
    # Create matrix of accuracies
    accuracy_matrix = []
    for method_name in method_names:
        method_accuracies = []
        for test_name in test_names:
            acc = test_sets[test_name].get(method_name)
            method_accuracies.append(acc if acc is not None else 0)
        accuracy_matrix.append(method_accuracies)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(test_names))
    width = 0.25
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (method_name, accuracies) in enumerate(zip(method_names, accuracy_matrix)):
        offset = width * (i - len(method_names) / 2 + 0.5)
        bars = ax.bar(x + offset, accuracies, width, label=method_name, 
                      color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.1%}',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Accuracy Comparison Across Methods and Test Sets', 
                     fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=15, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Accuracy comparison saved to {output_path}")
    
    plt.close()


def plot_conflict_examples(
    conflict_results: Dict,
    output_path: str,
    max_examples: int = 3
):
    """
    Create a markdown file with conflict resolution examples.
    
    Args:
        conflict_results: Dictionary containing conflict resolution results
        output_path: Path to save the markdown file
        max_examples: Maximum number of examples to include
    """
    logger.info("Creating conflict examples document...")
    
    cases = conflict_results.get("cases", [])
    if not cases:
        logger.error("No conflict cases found")
        return
    
    # Select examples (prioritize resolved cases)
    resolved_cases = [c for c in cases if c.get("resolved", False)]
    unresolved_cases = [c for c in cases if not c.get("resolved", False)]
    
    selected_cases = resolved_cases[:max_examples]
    if len(selected_cases) < max_examples:
        selected_cases.extend(unresolved_cases[:max_examples - len(selected_cases)])
    
    # Create markdown content
    md_content = "# Conflict Resolution Examples\n\n"
    md_content += f"**Conflict Resolution Rate**: {conflict_results.get('conflict_resolution_rate', 0):.2%}\n\n"
    md_content += f"**Total Cases**: {conflict_results.get('num_cases', 0)}\n\n"
    md_content += "---\n\n"
    
    for i, case in enumerate(selected_cases, 1):
        question = case.get("question", "")
        outputs = case.get("outputs", {})
        resolved = case.get("resolved", False)
        
        md_content += f"## Example {i}\n\n"
        md_content += f"**Question**: {question}\n\n"
        md_content += f"**Conflict Resolved**: {'✅ Yes' if resolved else '❌ No'}\n\n"
        
        # Add outputs from different adapters
        for adapter_name, output in outputs.items():
            md_content += f"### Response from {adapter_name.capitalize()} Adapter\n\n"
            md_content += f"```\n{output}\n```\n\n"
        
        # Add analysis
        if resolved:
            md_content += "**Analysis**: The model successfully provided different responses "
            md_content += "based on the local adapter, demonstrating effective conflict resolution.\n\n"
        else:
            md_content += "**Analysis**: The responses are too similar, indicating potential "
            md_content += "issues with adapter differentiation.\n\n"
        
        md_content += "---\n\n"
    
    # Save markdown file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Conflict examples saved to {output_path}")


def plot_overall_comparison(
    comparison_data: Dict,
    output_path: str,
    title: Optional[str] = None
):
    """
    Plot overall accuracy comparison across methods.
    
    Args:
        comparison_data: Dictionary containing comparison results
        output_path: Path to save the plot
        title: Optional custom title
    """
    logger.info("Plotting overall accuracy comparison...")
    
    methods = comparison_data.get("methods", [])
    if not methods:
        logger.error("No methods data found")
        return
    
    # Extract data
    method_names = [m["name"] for m in methods]
    accuracies = [m["overall_accuracy"] for m in methods]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(method_names, accuracies, color=colors[:len(method_names)], alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{height:.2%}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Accuracy', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Overall comparison saved to {output_path}")
    
    plt.close()


def create_all_visualizations(
    results_dir: str,
    output_dir: Optional[str] = None
):
    """
    Create all visualizations from results directory.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Optional output directory (defaults to results_dir/report)
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "report")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating visualizations from {results_dir}...")
    
    # Plot training curve
    metrics_path = os.path.join(results_dir, "metrics", "training_metrics.json")
    if os.path.exists(metrics_path):
        plot_training_curve(
            metrics_path=metrics_path,
            output_path=os.path.join(output_dir, "training_loss_curve.png"),
            metric_name="loss"
        )
    
    # Plot accuracy comparison
    comparison_path = os.path.join(results_dir, "metrics", "comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path, 'r') as f:
            comparison_data = json.load(f)
        
        plot_accuracy_comparison(
            comparison_data=comparison_data,
            output_path=os.path.join(output_dir, "accuracy_comparison.png")
        )
        
        plot_overall_comparison(
            comparison_data=comparison_data,
            output_path=os.path.join(output_dir, "overall_comparison.png")
        )
    
    # Create conflict examples
    conflict_path = os.path.join(results_dir, "metrics", "conflict_results.json")
    if os.path.exists(conflict_path):
        with open(conflict_path, 'r') as f:
            conflict_results = json.load(f)
        
        plot_conflict_examples(
            conflict_results=conflict_results,
            output_path=os.path.join(output_dir, "conflict_resolution_examples.md")
        )
    
    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations from experiment results")
    parser.add_argument("--results-dir", required=True, help="Directory containing results")
    parser.add_argument("--output-dir", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    create_all_visualizations(args.results_dir, args.output_dir)
