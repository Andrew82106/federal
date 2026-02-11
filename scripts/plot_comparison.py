#!/usr/bin/env python3
"""
生成实验对比图表
展示 FedAvg vs Dual-Adapter 在冲突场景下的表现
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(exp_name: str):
    """加载实验结果"""
    results_dir = Path(f'results/{exp_name}')
    
    # Load eval checkpoints
    eval_dir = results_dir / 'eval_checkpoints'
    test_a_strict = None
    test_a_service = None
    test_b_strict = None
    test_b_service = None
    
    if eval_dir.exists():
        # Test-A
        if (eval_dir / 'test_a_strict.json').exists():
            with open(eval_dir / 'test_a_strict.json') as f:
                test_a_strict = json.load(f).get('accuracy', 0)
        if (eval_dir / 'test_a_service.json').exists():
            with open(eval_dir / 'test_a_service.json') as f:
                test_a_service = json.load(f).get('accuracy', 0)
        
        # Test-B
        if (eval_dir / 'test_b_strict.json').exists():
            with open(eval_dir / 'test_b_strict.json') as f:
                test_b_strict = json.load(f).get('accuracy', 0)
        if (eval_dir / 'test_b_service.json').exists():
            with open(eval_dir / 'test_b_service.json') as f:
                test_b_service = json.load(f).get('accuracy', 0)
    
    # Load conflict test
    conflict_file = results_dir / 'metrics' / 'conflict_test_results.json'
    conflict_pass_rate = None
    if conflict_file.exists():
        with open(conflict_file) as f:
            conflict_pass_rate = json.load(f).get('pass_rate', 0)
    
    return {
        'test_a_strict': test_a_strict,
        'test_a_service': test_a_service,
        'test_b_strict': test_b_strict,
        'test_b_service': test_b_service,
        'conflict_pass_rate': conflict_pass_rate
    }

def plot_test_comparison():
    """绘制 Test-A 和 Test-B 的对比图"""
    
    # Load data
    exp000 = load_results('exp000_fedavg_baseline')
    exp001 = load_results('exp001_dual_adapter_fl')
    exp002 = load_results('exp002_improved_dual_adapter')
    
    # 准备数据
    models = ['FedAvg\n(Baseline)', 'Dual-Adapter\n(EXP001)', 'Dual-Adapter\n(EXP002)']
    
    # Test-A: Strict City Policy
    test_a_strict_scores = [
        exp000['test_a_strict'] or 0,
        exp001['test_a_strict'] or 0,
        exp002['test_a_strict'] or 0
    ]
    test_a_service_scores = [
        exp000['test_a_service'] or 0,
        exp001['test_a_service'] or 0,
        exp002['test_a_service'] or 0
    ]
    
    # Test-B: Service City Policy
    test_b_service_scores = [
        exp000['test_b_service'] or 0,
        exp001['test_b_service'] or 0,
        exp002['test_b_service'] or 0
    ]
    test_b_strict_scores = [
        exp000['test_b_strict'] or 0,
        exp001['test_b_strict'] or 0,
        exp002['test_b_strict'] or 0
    ]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Test-A 子图
    bars1 = ax1.bar(x - width/2, [s*100 for s in test_a_strict_scores], width, 
                    label='Strict Adapter', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, [s*100 for s in test_a_service_scores], width,
                    label='Service Adapter', color='#3498db', alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Test-A: Strict City Policy Memory', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
    
    # Test-B 子图
    bars3 = ax2.bar(x - width/2, [s*100 for s in test_b_service_scores], width,
                    label='Service Adapter', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, [s*100 for s in test_b_strict_scores], width,
                    label='Strict Adapter', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Test-B: Service City Policy Memory', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 添加数值标签
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path('results/comparison_test_ab.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.show()

def plot_conflict_comparison():
    """绘制 Conflict Test 对比图"""
    
    # Load data
    exp000 = load_results('exp000_fedavg_baseline')
    exp001 = load_results('exp001_dual_adapter_fl')
    exp002 = load_results('exp002_improved_dual_adapter')
    
    models = ['FedAvg\n(Baseline)', 'Dual-Adapter\n(EXP001)', 'Dual-Adapter\n(EXP002)']
    pass_rates = [
        (exp000['conflict_pass_rate'] or 0) * 100,
        (exp001['conflict_pass_rate'] or 0) * 100,
        (exp002['conflict_pass_rate'] or 0) * 100
    ]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#95a5a6', '#2ecc71', '#27ae60']
    bars = ax.bar(models, pass_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Conflict Test: Jurisdiction-Specific Response Ability', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加注释
    ax.text(0.5, 0.95, 'Higher is better: Model can distinguish city-specific policies',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path('results/comparison_conflict.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.show()

def plot_comprehensive_comparison():
    """绘制综合对比图（4个子图）"""
    
    # Load data
    exp000 = load_results('exp000_fedavg_baseline')
    exp001 = load_results('exp001_dual_adapter_fl')
    exp002 = load_results('exp002_improved_dual_adapter')
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    models = ['FedAvg', 'Ours\n(r=16)', 'Ours\n(r=32)']
    x = np.arange(len(models))
    width = 0.35
    
    # 1. Test-A
    ax1 = fig.add_subplot(gs[0, 0])
    test_a_strict = [
        (exp000['test_a_strict'] or 0) * 100,
        (exp001['test_a_strict'] or 0) * 100,
        (exp002['test_a_strict'] or 0) * 100
    ]
    test_a_service = [
        (exp000['test_a_service'] or 0) * 100,
        (exp001['test_a_service'] or 0) * 100,
        (exp002['test_a_service'] or 0) * 100
    ]
    
    bars1 = ax1.bar(x - width/2, test_a_strict, width, label='Strict Prompt', 
                    color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_a_service, width, label='Service Prompt',
                    color='#3498db', alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('(A) Test-A: Strict City Policy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 50)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Test-B
    ax2 = fig.add_subplot(gs[0, 1])
    test_b_service = [
        (exp000['test_b_service'] or 0) * 100,
        (exp001['test_b_service'] or 0) * 100,
        (exp002['test_b_service'] or 0) * 100
    ]
    test_b_strict = [
        (exp000['test_b_strict'] or 0) * 100,
        (exp001['test_b_strict'] or 0) * 100,
        (exp002['test_b_strict'] or 0) * 100
    ]
    
    bars3 = ax2.bar(x - width/2, test_b_service, width, label='Service Prompt',
                    color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, test_b_strict, width, label='Strict Prompt',
                    color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('(B) Test-B: Service City Policy', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 70)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Privacy Gap
    ax3 = fig.add_subplot(gs[1, 0])
    privacy_gap_a = [
        test_a_strict[i] - test_a_service[i] for i in range(3)
    ]
    privacy_gap_b = [
        test_b_service[i] - test_b_strict[i] for i in range(3)
    ]
    
    bars5 = ax3.bar(x - width/2, privacy_gap_a, width, label='Test-A Gap',
                    color='#9b59b6', alpha=0.8)
    bars6 = ax3.bar(x + width/2, privacy_gap_b, width, label='Test-B Gap',
                    color='#1abc9c', alpha=0.8)
    
    ax3.set_ylabel('Privacy Gap (%)', fontsize=11)
    ax3.set_title('(C) Privacy Protection Capability', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.1:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:+.1f}', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=8)
    
    # 4. Conflict Test
    ax4 = fig.add_subplot(gs[1, 1])
    conflict_rates = [
        (exp000['conflict_pass_rate'] or 0) * 100,
        (exp001['conflict_pass_rate'] or 0) * 100,
        (exp002['conflict_pass_rate'] or 0) * 100
    ]
    
    colors = ['#95a5a6', '#2ecc71', '#27ae60']
    bars7 = ax4.bar(models, conflict_rates, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('Pass Rate (%)', fontsize=11)
    ax4.set_title('(D) Conflict Resolution Ability', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 40)
    
    for bar in bars7:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    # 保存图表
    output_path = Path('results/comprehensive_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Generating comparison plots...")
    print("\n1. Test-A and Test-B Comparison")
    plot_test_comparison()
    
    print("\n2. Conflict Test Comparison")
    plot_conflict_comparison()
    
    print("\n3. Comprehensive Comparison (4 subplots)")
    plot_comprehensive_comparison()
    
    print("\n✅ All plots generated successfully!")
