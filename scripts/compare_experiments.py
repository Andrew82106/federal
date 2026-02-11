#!/usr/bin/env python3
"""
对比 exp001 和 exp002 的完整评估结果
生成详细的性能对比报告
"""

import json
from pathlib import Path
from typing import Dict, Any

def load_eval_results(exp_dir: Path) -> Dict[str, Any]:
    """加载实验的评估结果"""
    results = {
        'test_g': {},
        'test_a': {},
        'test_b': {},
        'conflict': None
    }
    
    eval_dir = exp_dir / 'eval_checkpoints'
    if not eval_dir.exists():
        return results
    
    # Load Test-G results
    for f in eval_dir.glob('test_g_*.json'):
        data = json.load(open(f))
        adapter_type = data.get('adapter_type', f.stem.replace('test_g_', ''))
        results['test_g'][adapter_type] = {
            'accuracy': data.get('accuracy', 0),
            'correct': data.get('correct', 0),
            'total': data.get('total', 0)
        }
    
    # Load Test-A results
    for f in eval_dir.glob('test_a_*.json'):
        data = json.load(open(f))
        adapter_type = data.get('adapter_type', f.stem.replace('test_a_', ''))
        results['test_a'][adapter_type] = {
            'accuracy': data.get('accuracy', 0),
            'correct': data.get('correct', 0),
            'total': data.get('total', 0)
        }
    
    # Load Test-B results
    for f in eval_dir.glob('test_b_*.json'):
        data = json.load(open(f))
        adapter_type = data.get('adapter_type', f.stem.replace('test_b_', ''))
        results['test_b'][adapter_type] = {
            'accuracy': data.get('accuracy', 0),
            'correct': data.get('correct', 0),
            'total': data.get('total', 0)
        }
    
    # Load Conflict test results
    metrics_dir = exp_dir / 'metrics'
    conflict_file = metrics_dir / 'conflict_test_results.json'
    if conflict_file.exists():
        results['conflict'] = json.load(open(conflict_file))
    
    return results

def print_section_header(title: str):
    """打印章节标题"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")

def print_test_comparison(test_name: str, exp001_data: Dict, exp002_data: Dict):
    """打印单个测试集的对比"""
    print(f"\n{test_name}")
    print("-" * 80)
    
    # 获取所有 adapter 类型
    all_adapters = sorted(set(list(exp001_data.keys()) + list(exp002_data.keys())))
    
    if not all_adapters:
        print("   No data available")
        return
    
    # 表头
    print(f"{'Adapter Type':<20} {'EXP001':>15} {'EXP002':>15} {'Δ':>10}")
    print("-" * 80)
    
    # 逐行对比
    for adapter in all_adapters:
        exp001_acc = exp001_data.get(adapter, {}).get('accuracy', 0)
        exp002_acc = exp002_data.get(adapter, {}).get('accuracy', 0)
        delta = exp002_acc - exp001_acc if exp001_acc > 0 else 0
        
        exp001_str = f"{exp001_acc:.1%}" if exp001_acc > 0 else "N/A"
        exp002_str = f"{exp002_acc:.1%}" if exp002_acc > 0 else "N/A"
        delta_str = f"{delta:+.1%}" if delta != 0 else "-"
        
        print(f"{adapter:<20} {exp001_str:>15} {exp002_str:>15} {delta_str:>10}")

def calculate_privacy_gap(test_data: Dict, strict_key: str, service_key: str) -> float:
    """计算隐私保护差距"""
    if strict_key in test_data and service_key in test_data:
        return test_data[strict_key]['accuracy'] - test_data[service_key]['accuracy']
    return 0.0

def main():
    results_dir = Path('results')
    
    # 加载两个实验的结果
    exp001_results = load_eval_results(results_dir / 'exp001_dual_adapter_fl')
    exp002_results = load_eval_results(results_dir / 'exp002_improved_dual_adapter')
    
    print_section_header("实验对比报告：EXP001 vs EXP002")
    
    # 实验配置对比
    print("\n📋 实验配置")
    print("-" * 80)
    print(f"{'Parameter':<30} {'EXP001':<25} {'EXP002':<25}")
    print("-" * 80)
    print(f"{'LoRA Rank (r)':<30} {'16':<25} {'32':<25}")
    print(f"{'LoRA Alpha':<30} {'32':<25} {'64':<25}")
    print(f"{'Epochs per Round':<30} {'2':<25} {'3':<25}")
    print(f"{'Federated Rounds':<30} {'5':<25} {'5':<25}")
    print(f"{'Batch Size':<30} {'4':<25} {'4':<25}")
    
    # Test-G: 通用法律知识保持
    print_test_comparison(
        "📚 Test-G: Universal Law Knowledge Retention",
        exp001_results['test_g'],
        exp002_results['test_g']
    )
    
    # Test-A: 严管城市政策记忆
    print_test_comparison(
        "🔒 Test-A: Strict City Policy Memory",
        exp001_results['test_a'],
        exp002_results['test_a']
    )
    
    # 计算 Test-A 的隐私保护差距
    exp001_privacy_a = calculate_privacy_gap(exp001_results['test_a'], 'strict', 'service')
    exp002_privacy_a = calculate_privacy_gap(exp002_results['test_a'], 'strict', 'service')
    
    if exp001_privacy_a > 0 or exp002_privacy_a > 0:
        print(f"\n   Privacy Gap (Strict - Service):")
        print(f"   EXP001: {exp001_privacy_a:+.1%}  |  EXP002: {exp002_privacy_a:+.1%}")
    
    # Test-B: 服务型城市政策记忆
    print_test_comparison(
        "🤝 Test-B: Service City Policy Memory",
        exp001_results['test_b'],
        exp002_results['test_b']
    )
    
    # 计算 Test-B 的隐私保护差距
    exp001_privacy_b = calculate_privacy_gap(exp001_results['test_b'], 'service', 'strict')
    exp002_privacy_b = calculate_privacy_gap(exp002_results['test_b'], 'service', 'strict')
    
    if exp001_privacy_b > 0 or exp002_privacy_b > 0:
        print(f"\n   Privacy Gap (Service - Strict):")
        print(f"   EXP001: {exp001_privacy_b:+.1%}  |  EXP002: {exp002_privacy_b:+.1%}")
    
    # Conflict Test 对比
    print_section_header("⚔️  Conflict Test: Jurisdiction-Specific Response")
    
    exp001_conflict = exp001_results['conflict']
    exp002_conflict = exp002_results['conflict']
    
    if exp001_conflict or exp002_conflict:
        print(f"\n{'Metric':<30} {'EXP001':>15} {'EXP002':>15} {'Δ':>10}")
        print("-" * 80)
        
        if exp001_conflict and exp002_conflict:
            exp001_pass = exp001_conflict.get('pass_rate', 0)
            exp002_pass = exp002_conflict.get('pass_rate', 0)
            delta_pass = exp002_pass - exp001_pass
            
            print(f"{'Pass Rate':<30} {exp001_pass:>14.1%} {exp002_pass:>14.1%} {delta_pass:>9.1%}")
            print(f"{'Passed Cases':<30} {exp001_conflict.get('passed', 0):>15} {exp002_conflict.get('passed', 0):>15}")
            print(f"{'Failed Cases':<30} {exp001_conflict.get('failed', 0):>15} {exp002_conflict.get('failed', 0):>15}")
            print(f"{'Total Cases':<30} {exp001_conflict.get('total_cases', 0):>15} {exp002_conflict.get('total_cases', 0):>15}")
        elif exp001_conflict:
            print(f"{'Pass Rate':<30} {exp001_conflict.get('pass_rate', 0):>14.1%} {'Pending':>15}")
        elif exp002_conflict:
            print(f"{'Pass Rate':<30} {'Pending':>15} {exp002_conflict.get('pass_rate', 0):>14.1%}")
    else:
        print("\n   Status: Both experiments pending conflict test results")
    
    # 综合评估
    print_section_header("📊 综合评估")
    
    print("\n✅ 关键发现:")
    print("-" * 80)
    
    # 计算平均准确率
    def calc_avg_accuracy(results: Dict) -> float:
        all_acc = []
        for test in ['test_g', 'test_a', 'test_b']:
            for adapter_data in results[test].values():
                if adapter_data.get('accuracy', 0) > 0:
                    all_acc.append(adapter_data['accuracy'])
        return sum(all_acc) / len(all_acc) if all_acc else 0
    
    exp001_avg = calc_avg_accuracy(exp001_results)
    exp002_avg = calc_avg_accuracy(exp002_results)
    
    print(f"\n1. 平均准确率:")
    print(f"   EXP001: {exp001_avg:.1%}")
    print(f"   EXP002: {exp002_avg:.1%}")
    print(f"   提升: {exp002_avg - exp001_avg:+.1%}")
    
    print(f"\n2. 模型容量:")
    print(f"   EXP001 (r=16): 更轻量，训练更快")
    print(f"   EXP002 (r=32): 更大容量，表达能力更强")
    
    print(f"\n3. 隐私保护:")
    if exp001_privacy_a > 0 and exp002_privacy_a > 0:
        print(f"   Test-A Privacy Gap: EXP001={exp001_privacy_a:.1%}, EXP002={exp002_privacy_a:.1%}")
    if exp001_privacy_b > 0 and exp002_privacy_b > 0:
        print(f"   Test-B Privacy Gap: EXP001={exp001_privacy_b:.1%}, EXP002={exp002_privacy_b:.1%}")
    
    print(f"\n4. 训练成本:")
    print(f"   EXP001: 2 epochs/round × 5 rounds = 10 epochs")
    print(f"   EXP002: 3 epochs/round × 5 rounds = 15 epochs (+50%)")
    
    # 结论
    print_section_header("🎯 结论与建议")
    
    print("\n如果追求:")
    print("  • 更高准确率 → 选择 EXP002 (r=32, 3 epochs)")
    print("  • 训练效率   → 选择 EXP001 (r=16, 2 epochs)")
    print("  • 平衡方案   → EXP001 配置已足够，性价比高")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
