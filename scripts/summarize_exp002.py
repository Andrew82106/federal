#!/usr/bin/env python3
"""汇总 exp002 评估结果"""

import json
from pathlib import Path

results_dir = Path('results/exp002_improved_dual_adapter/eval_checkpoints')

print('='*80)
print('EXP002 评估结果汇总 (LLM Judge)')
print('='*80)

print('\n📚 Test-G: Universal Law Knowledge Retention')
for f in sorted(results_dir.glob('test_g_*.json')):
    data = json.load(open(f))
    print(f"   {data['adapter_type']:15s}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

print('\n🔒 Test-A: Strict City Policy Memory')
test_a_data = {}
for f in sorted(results_dir.glob('test_a_*.json')):
    data = json.load(open(f))
    test_a_data[data['adapter_type']] = data
    marker = ' ✅' if data['adapter_type'] == 'strict' else ''
    print(f"   {data['adapter_type']:15s}: {data['accuracy']:.1%} ({data['correct']}/{data['total']}){marker}")

if 'strict' in test_a_data and 'service' in test_a_data:
    privacy_a = test_a_data['strict']['accuracy'] - test_a_data['service']['accuracy']
    print(f"   Privacy Gap: {privacy_a:+.1%}")

print('\n🤝 Test-B: Service City Policy Memory')
test_b_data = {}
for f in sorted(results_dir.glob('test_b_*.json')):
    data = json.load(open(f))
    test_b_data[data['adapter_type']] = data
    marker = ' ✅' if data['adapter_type'] == 'service' else ''
    print(f"   {data['adapter_type']:15s}: {data['accuracy']:.1%} ({data['correct']}/{data['total']}){marker}")

if 'service' in test_b_data and 'strict' in test_b_data:
    privacy_b = test_b_data['service']['accuracy'] - test_b_data['strict']['accuracy']
    print(f"   Privacy Gap: {privacy_b:+.1%}")

print('\n⚔️  Conflict Test')
print('   Status: Pending (PEFT loading issue)')

print('\n' + '='*80)
print('注：Conflict Test 因 PEFT 库的 adapter 加载问题暂未完成')
print('    前6个测试集已成功完成，足以评估模型性能')
print('='*80)
