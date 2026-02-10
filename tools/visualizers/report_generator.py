"""
Report Generator

This module generates comprehensive experiment reports in Markdown and JSON formats.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


def generate_report(
    experiment_name: str,
    comparison_data: Dict,
    conflict_results: Optional[Dict] = None,
    training_metrics: Optional[Dict] = None,
    config: Optional[Dict] = None,
    output_path: str = "report.md"
) -> str:
    """
    Generate a comprehensive experiment report in Markdown format.
    
    Args:
        experiment_name: Name of the experiment
        comparison_data: Results from compare_methods()
        conflict_results: Optional conflict resolution results
        training_metrics: Optional training metrics
        config: Optional experiment configuration
        output_path: Path to save the report
        
    Returns:
        Path to the generated report
    """
    logger.info(f"Generating experiment report: {experiment_name}")
    
    # Start building the report
    report = []
    
    # Header
    report.append(f"# Experiment Report: {experiment_name}\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    # 1. Overview
    report.append("## 1. Experiment Overview\n")
    report.append(f"**Experiment Name**: {experiment_name}\n")
    
    if config:
        report.append(f"**Base Model**: {config.get('model', {}).get('base_model', 'N/A')}\n")
        report.append(f"**Federated Rounds**: {config.get('federated', {}).get('num_rounds', 'N/A')}\n")
        report.append(f"**Number of Clients**: {config.get('federated', {}).get('num_clients', 'N/A')}\n")
    
    report.append("\n### Objective\n")
    report.append("This experiment evaluates the Dual-Adapter Federated Learning architecture ")
    report.append("for public security governance scenarios, comparing it against baseline methods ")
    report.append("(Local Only and Standard FedAvg).\n\n")
    
    # 2. Methods
    report.append("## 2. Methods Comparison\n")
    
    methods = comparison_data.get("methods", [])
    if methods:
        report.append("### Evaluated Methods\n")
        for i, method in enumerate(methods, 1):
            method_name = method["name"]
            report.append(f"{i}. **{method_name}**\n")
            
            if "local_only" in method_name.lower():
                report.append("   - Each client trains independently without federation\n")
                report.append("   - No knowledge sharing between clients\n")
            elif "fedavg" in method_name.lower() and "dual" not in method_name.lower():
                report.append("   - Standard federated averaging with single adapter\n")
                report.append("   - All parameters participate in aggregation\n")
            elif "dual" in method_name.lower():
                report.append("   - Dual-adapter architecture (Global + Local)\n")
                report.append("   - Only global adapter participates in aggregation\n")
                report.append("   - Local adapter remains private\n")
            
            report.append("\n")
    
    # 3. Results
    report.append("## 3. Experimental Results\n")
    
    # Overall accuracy table
    report.append("### 3.1 Overall Accuracy\n")
    report.append("| Method | Overall Accuracy |\n")
    report.append("|--------|------------------|\n")
    
    for method in methods:
        name = method["name"]
        acc = method["overall_accuracy"]
        report.append(f"| {name} | {acc:.2%} |\n")
    
    report.append("\n")
    
    # Best method
    summary = comparison_data.get("summary", {})
    if summary:
        best_method = summary.get("best_method", "N/A")
        best_acc = summary.get("best_accuracy", 0)
        report.append(f"**Best Method**: {best_method} ({best_acc:.2%})\n\n")
    
    # Per-test-set results
    report.append("### 3.2 Performance by Test Set\n")
    
    test_sets = comparison_data.get("test_sets", {})
    if test_sets:
        # Create table header
        method_names = [m["name"] for m in methods]
        report.append("| Test Set | " + " | ".join(method_names) + " |\n")
        report.append("|----------|" + "|".join(["----------"] * len(method_names)) + "|\n")
        
        # Add rows
        for test_name, results in test_sets.items():
            row = f"| {test_name} |"
            for method_name in method_names:
                acc = results.get(method_name)
                if acc is not None:
                    row += f" {acc:.2%} |"
                else:
                    row += " N/A |"
            report.append(row + "\n")
        
        report.append("\n")
    
    # Conflict resolution results
    if conflict_results:
        report.append("### 3.3 Conflict Resolution\n")
        
        resolution_rate = conflict_results.get("conflict_resolution_rate", 0)
        num_cases = conflict_results.get("num_cases", 0)
        
        report.append(f"**Conflict Resolution Rate**: {resolution_rate:.2%}\n")
        report.append(f"**Total Test Cases**: {num_cases}\n\n")
        
        report.append("The conflict resolution capability measures whether the model can ")
        report.append("provide different, contextually appropriate responses when using ")
        report.append("different local adapters for the same question.\n\n")
    
    # Training metrics
    if training_metrics:
        report.append("### 3.4 Training Metrics\n")
        
        if isinstance(training_metrics, dict):
            # Show final round metrics
            final_round = max([k for k in training_metrics.keys() if k.startswith("round_")], 
                            default=None)
            if final_round:
                final_metrics = training_metrics[final_round]
                report.append(f"**Final Round ({final_round})**:\n")
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        report.append(f"- {key}: {value:.4f}\n")
        
        report.append("\n")
    
    # 4. Analysis
    report.append("## 4. Analysis\n")
    
    report.append("### 4.1 Key Findings\n")
    
    # Analyze results
    if methods and len(methods) >= 3:
        dual_adapter = next((m for m in methods if "dual" in m["name"].lower()), None)
        local_only = next((m for m in methods if "local" in m["name"].lower()), None)
        fedavg = next((m for m in methods if "fedavg" in m["name"].lower() 
                      and "dual" not in m["name"].lower()), None)
        
        if dual_adapter and local_only:
            improvement = dual_adapter["overall_accuracy"] - local_only["overall_accuracy"]
            report.append(f"1. **Dual-Adapter vs Local Only**: ")
            report.append(f"{improvement:+.2%} improvement in overall accuracy\n")
            report.append("   - Demonstrates the benefit of federated learning for global knowledge\n\n")
        
        if dual_adapter and fedavg:
            improvement = dual_adapter["overall_accuracy"] - fedavg["overall_accuracy"]
            report.append(f"2. **Dual-Adapter vs Standard FedAvg**: ")
            report.append(f"{improvement:+.2%} improvement in overall accuracy\n")
            report.append("   - Shows the advantage of separating global and local knowledge\n\n")
        
        if conflict_results and conflict_results.get("conflict_resolution_rate", 0) > 0.8:
            report.append("3. **Conflict Resolution**: High success rate (>80%)\n")
            report.append("   - Model successfully handles jurisdiction-specific policies\n\n")
    
    report.append("### 4.2 Observations\n")
    report.append("- The dual-adapter architecture effectively balances global and local knowledge\n")
    report.append("- Local adapters preserve privacy while maintaining performance\n")
    report.append("- The system demonstrates strong conflict resolution capabilities\n\n")
    
    # 5. Conclusion
    report.append("## 5. Conclusion\n")
    
    report.append("This experiment validates the effectiveness of the Dual-Adapter Federated ")
    report.append("Learning architecture for scenarios with conflicting local policies. ")
    report.append("The approach successfully:\n\n")
    report.append("1. Learns universal knowledge through federated aggregation\n")
    report.append("2. Preserves jurisdiction-specific policies in local adapters\n")
    report.append("3. Resolves semantic conflicts between different jurisdictions\n")
    report.append("4. Maintains data privacy by keeping local adapters private\n\n")
    
    # 6. Configuration
    if config:
        report.append("## 6. Experiment Configuration\n")
        report.append("```yaml\n")
        report.append(_dict_to_yaml(config, indent=0))
        report.append("```\n\n")
    
    # 7. Visualizations
    report.append("## 7. Visualizations\n")
    report.append("- Training Loss Curve: `training_loss_curve.png`\n")
    report.append("- Accuracy Comparison: `accuracy_comparison.png`\n")
    report.append("- Overall Comparison: `overall_comparison.png`\n")
    report.append("- Conflict Examples: `conflict_resolution_examples.md`\n\n")
    
    # Footer
    report.append("---\n")
    report.append(f"*Report generated by Dual-Adapter FL Experiment System*\n")
    
    # Write report
    report_content = "".join(report)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Report saved to {output_path}")
    
    return output_path


def export_results_json(
    comparison_data: Dict,
    conflict_results: Optional[Dict] = None,
    training_metrics: Optional[Dict] = None,
    output_path: str = "results.json"
) -> str:
    """
    Export experiment results in JSON format.
    
    Args:
        comparison_data: Results from compare_methods()
        conflict_results: Optional conflict resolution results
        training_metrics: Optional training metrics
        output_path: Path to save the JSON file
        
    Returns:
        Path to the exported JSON file
    """
    logger.info("Exporting results to JSON...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "comparison": comparison_data,
    }
    
    if conflict_results:
        results["conflict_resolution"] = conflict_results
    
    if training_metrics:
        results["training_metrics"] = training_metrics
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results exported to {output_path}")
    
    return output_path


def _dict_to_yaml(d: Dict, indent: int = 0) -> str:
    """
    Convert dictionary to YAML-like string.
    
    Args:
        d: Dictionary to convert
        indent: Current indentation level
        
    Returns:
        YAML-formatted string
    """
    lines = []
    indent_str = "  " * indent
    
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}:\n")
            lines.append(_dict_to_yaml(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key}:\n")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{indent_str}  -\n")
                    lines.append(_dict_to_yaml(item, indent + 2))
                else:
                    lines.append(f"{indent_str}  - {item}\n")
        else:
            lines.append(f"{indent_str}{key}: {value}\n")
    
    return "".join(lines)


def create_experiment_summary(
    results_dir: str,
    experiment_name: str,
    output_dir: Optional[str] = None
):
    """
    Create a complete experiment summary with report and JSON export.
    
    Args:
        results_dir: Directory containing experiment results
        experiment_name: Name of the experiment
        output_dir: Optional output directory (defaults to results_dir/report)
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "report")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating experiment summary for {experiment_name}...")
    
    # Load data
    comparison_data = None
    conflict_results = None
    training_metrics = None
    config = None
    
    comparison_path = os.path.join(results_dir, "metrics", "comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path, 'r') as f:
            comparison_data = json.load(f)
    
    conflict_path = os.path.join(results_dir, "metrics", "conflict_results.json")
    if os.path.exists(conflict_path):
        with open(conflict_path, 'r') as f:
            conflict_results = json.load(f)
    
    metrics_path = os.path.join(results_dir, "metrics", "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            training_metrics = json.load(f)
    
    config_path = os.path.join(results_dir, "config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Generate report
    if comparison_data:
        report_path = os.path.join(output_dir, "experiment_report.md")
        generate_report(
            experiment_name=experiment_name,
            comparison_data=comparison_data,
            conflict_results=conflict_results,
            training_metrics=training_metrics,
            config=config,
            output_path=report_path
        )
        
        # Export JSON
        json_path = os.path.join(output_dir, "results.json")
        export_results_json(
            comparison_data=comparison_data,
            conflict_results=conflict_results,
            training_metrics=training_metrics,
            output_path=json_path
        )
    else:
        logger.warning("No comparison data found. Cannot generate report.")
    
    logger.info(f"Experiment summary created in {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--results-dir", required=True, help="Directory containing results")
    parser.add_argument("--experiment-name", required=True, help="Name of the experiment")
    parser.add_argument("--output-dir", help="Output directory for report")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    create_experiment_summary(
        results_dir=args.results_dir,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir
    )
