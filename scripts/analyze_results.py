#!/usr/bin/env python3
"""
Analyze and visualize experiment results.

Usage:
    python scripts/analyze_results.py --exp exp001_dual_adapter_fl
"""

import json
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_results(exp_name: str):
    """Load evaluation results from experiment."""
    results_dir = project_root / "results" / exp_name / "metrics"
    results_file = results_dir / "comprehensive_evaluation_results.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_summary(exp_name: str, results: dict):
    """Print formatted summary of results."""
    print("\n" + "="*80)
    print(f"EXPERIMENT RESULTS: {exp_name}")
    print("="*80)
    
    # Test-G: Universal Law Knowledge
    print("\n📚 Test-G: Universal Law Knowledge Retention")
    print("-" * 80)
    test_g = results['test_g']
    print(f"  Global Only:     {test_g['global_only']['accuracy']:.1%} ({test_g['global_only']['correct']}/{test_g['global_only']['total']})")
    print(f"  Strict Adapter:  {test_g['strict']['accuracy']:.1%} ({test_g['strict']['correct']}/{test_g['strict']['total']})")
    print(f"  Service Adapter: {test_g['service']['accuracy']:.1%} ({test_g['service']['correct']}/{test_g['service']['total']})")
    
    avg_retention = (test_g['strict']['accuracy'] + test_g['service']['accuracy']) / 2
    print(f"\n  Average Retention (with local adapters): {avg_retention:.1%}")
    print(f"  Degradation from Global Only: {test_g['global_only']['accuracy'] - avg_retention:+.1%}")
    
    # Test-A: Strict City Policy
    print("\n🔒 Test-A: Strict City Policy Memory")
    print("-" * 80)
    test_a = results['test_a']
    print(f"  Strict Adapter:  {test_a['strict']['accuracy']:.1%} ({test_a['strict']['correct']}/{test_a['strict']['total']}) ✅ Expected High")
    print(f"  Service Adapter: {test_a['service']['accuracy']:.1%} ({test_a['service']['correct']}/{test_a['service']['total']}) ⚠️  Expected Low")
    
    privacy_gap_a = test_a['strict']['accuracy'] - test_a['service']['accuracy']
    print(f"\n  Privacy Gap: {privacy_gap_a:+.1%}")
    if privacy_gap_a > 0.3:
        print(f"  Status: ✅ Good isolation (gap > 30%)")
    elif privacy_gap_a > 0.15:
        print(f"  Status: ⚠️  Moderate isolation (15% < gap < 30%)")
    else:
        print(f"  Status: ❌ Poor isolation (gap < 15%)")
    
    # Test-B: Service City Policy
    print("\n🤝 Test-B: Service City Policy Memory")
    print("-" * 80)
    test_b = results['test_b']
    print(f"  Service Adapter: {test_b['service']['accuracy']:.1%} ({test_b['service']['correct']}/{test_b['service']['total']}) ✅ Expected High")
    print(f"  Strict Adapter:  {test_b['strict']['accuracy']:.1%} ({test_b['strict']['correct']}/{test_b['strict']['total']}) ⚠️  Expected Low")
    
    privacy_gap_b = test_b['service']['accuracy'] - test_b['strict']['accuracy']
    print(f"\n  Privacy Gap: {privacy_gap_b:+.1%}")
    if privacy_gap_b > 0.3:
        print(f"  Status: ✅ Good isolation (gap > 30%)")
    elif privacy_gap_b > 0.15:
        print(f"  Status: ⚠️  Moderate isolation (15% < gap < 30%)")
    else:
        print(f"  Status: ❌ Poor isolation (gap < 15%)")
    
    # Conflict Test
    print("\n⚔️  Conflict Test: Jurisdiction-Specific Behavior")
    print("-" * 80)
    conflict = results['conflict']
    print(f"  Total Cases: {conflict['total_cases']}")
    print(f"  Passed: {conflict['passed']}")
    print(f"  Failed: {conflict['failed']}")
    print(f"  Ambiguous: {conflict['ambiguous']}")
    print(f"  No Match: {conflict['no_match']}")
    print(f"\n  Pass Rate: {conflict['pass_rate']:.1%}")
    
    if conflict['pass_rate'] > 0.7:
        print(f"  Status: ✅ Excellent conflict resolution")
    elif conflict['pass_rate'] > 0.5:
        print(f"  Status: ⚠️  Good conflict resolution")
    elif conflict['pass_rate'] > 0.3:
        print(f"  Status: ⚠️  Moderate conflict resolution")
    else:
        print(f"  Status: ❌ Poor conflict resolution")
    
    # Overall Assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    
    scores = {
        'Knowledge Retention': avg_retention,
        'Privacy (Strict)': privacy_gap_a,
        'Privacy (Service)': privacy_gap_b,
        'Conflict Resolution': conflict['pass_rate']
    }
    
    for metric, score in scores.items():
        bar_length = int(score * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {metric:.<25} {bar} {score:.1%}")
    
    avg_score = sum(scores.values()) / len(scores)
    print(f"\n  Average Score: {avg_score:.1%}")
    
    print("\n" + "="*80)


def analyze_failure_patterns(exp_name: str, results: dict):
    """Analyze common failure patterns."""
    print("\n" + "="*80)
    print("FAILURE PATTERN ANALYSIS")
    print("="*80)
    
    # Analyze Test-G failures
    print("\n📚 Test-G Failures (Knowledge Degradation)")
    print("-" * 80)
    
    for adapter_type in ['strict', 'service']:
        test_g = results['test_g'][adapter_type]
        failures = [d for d in test_g['details'] if not d['correct']]
        
        if failures:
            print(f"\n{adapter_type.upper()} Adapter - {len(failures)} failures:")
            for i, failure in enumerate(failures[:3], 1):  # Show first 3
                print(f"\n  {i}. Q: {failure['question'][:80]}...")
                print(f"     Expected: {failure['expected'][:80]}...")
                print(f"     Got: {failure['response'][:80]}...")
    
    # Analyze Privacy Leakage
    print("\n\n🔒 Privacy Leakage Analysis")
    print("-" * 80)
    
    # Test-A: Service adapter should NOT know strict policies
    test_a_service = results['test_a']['service']
    leakage_a = [d for d in test_a_service['details'] if d['correct']]
    if leakage_a:
        print(f"\n⚠️  Test-A Leakage: Service adapter correctly answered {len(leakage_a)} strict policy questions")
        print(f"   This suggests knowledge leakage from strict to service adapter")
    
    # Test-B: Strict adapter should NOT know service policies
    test_b_strict = results['test_b']['strict']
    leakage_b = [d for d in test_b_strict['details'] if d['correct']]
    if leakage_b:
        print(f"\n⚠️  Test-B Leakage: Strict adapter correctly answered {len(leakage_b)} service policy questions")
        print(f"   This suggests knowledge leakage from service to strict adapter")
    
    # Analyze Conflict Test failures
    print("\n\n⚔️  Conflict Test Failure Analysis")
    print("-" * 80)
    
    conflict = results['conflict']
    failures = [c for c in conflict['cases'] if not c['success']]
    
    if failures:
        print(f"\nTotal failures: {len(failures)}")
        
        # Categorize failures
        strict_failures = sum(1 for f in failures if f['behaviors']['strict'] != 'STRICT_BEHAVIOR')
        service_failures = sum(1 for f in failures if f['behaviors']['service'] != 'SERVICE_BEHAVIOR')
        
        print(f"  Strict adapter failures: {strict_failures}")
        print(f"  Service adapter failures: {service_failures}")
        
        # Show examples
        print(f"\nExample failures:")
        for i, failure in enumerate(failures[:2], 1):
            print(f"\n  {i}. Q: {failure['question'][:80]}...")
            print(f"     Strict behavior: {failure['behaviors']['strict']}")
            print(f"     Service behavior: {failure['behaviors']['service']}")
    
    print("\n" + "="*80)


def export_comparison_table(exp_name: str, results: dict):
    """Export results as comparison table."""
    output_file = project_root / "results" / exp_name / "metrics" / "summary_table.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"EXPERIMENT RESULTS: {exp_name}\n")
        f.write("="*80 + "\n\n")
        
        # Create comparison table
        f.write("Metric                          | Value      | Status\n")
        f.write("-" * 80 + "\n")
        
        test_g = results['test_g']
        f.write(f"Test-G (Global Only)            | {test_g['global_only']['accuracy']:>6.1%}    | Baseline\n")
        f.write(f"Test-G (Strict Adapter)         | {test_g['strict']['accuracy']:>6.1%}    | {'✅' if test_g['strict']['accuracy'] > 0.6 else '❌'}\n")
        f.write(f"Test-G (Service Adapter)        | {test_g['service']['accuracy']:>6.1%}    | {'✅' if test_g['service']['accuracy'] > 0.6 else '❌'}\n")
        f.write("-" * 80 + "\n")
        
        test_a = results['test_a']
        privacy_a = test_a['strict']['accuracy'] - test_a['service']['accuracy']
        f.write(f"Test-A (Strict Adapter)         | {test_a['strict']['accuracy']:>6.1%}    | {'✅' if test_a['strict']['accuracy'] > 0.7 else '❌'}\n")
        f.write(f"Test-A (Service Adapter)        | {test_a['service']['accuracy']:>6.1%}    | {'✅' if test_a['service']['accuracy'] < 0.3 else '❌'}\n")
        f.write(f"Test-A Privacy Gap              | {privacy_a:>+6.1%}    | {'✅' if privacy_a > 0.3 else '❌'}\n")
        f.write("-" * 80 + "\n")
        
        test_b = results['test_b']
        privacy_b = test_b['service']['accuracy'] - test_b['strict']['accuracy']
        f.write(f"Test-B (Service Adapter)        | {test_b['service']['accuracy']:>6.1%}    | {'✅' if test_b['service']['accuracy'] > 0.7 else '❌'}\n")
        f.write(f"Test-B (Strict Adapter)         | {test_b['strict']['accuracy']:>6.1%}    | {'✅' if test_b['strict']['accuracy'] < 0.3 else '❌'}\n")
        f.write(f"Test-B Privacy Gap              | {privacy_b:>+6.1%}    | {'✅' if privacy_b > 0.3 else '❌'}\n")
        f.write("-" * 80 + "\n")
        
        conflict = results['conflict']
        f.write(f"Conflict Test Pass Rate         | {conflict['pass_rate']:>6.1%}    | {'✅' if conflict['pass_rate'] > 0.5 else '❌'}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Summary table exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--exp', type=str, default='exp001_dual_adapter_fl',
                       help='Experiment name (default: exp001_dual_adapter_fl)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed failure analysis')
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.exp)
    if results is None:
        return 1
    
    # Print summary
    print_summary(args.exp, results)
    
    # Detailed analysis
    if args.detailed:
        analyze_failure_patterns(args.exp, results)
    
    # Export table
    export_comparison_table(args.exp, results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
