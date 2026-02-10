---
inclusion: always
---

# Git 工作流程

## 项目部署架构

本项目采用 **本地开发 + 远端训练** 的架构：

- **本地（Mac）**: 代码开发、编辑、版本控制
- **远端（AutoDL GPU 服务器）**: 模型训练、推理测试
- **连接方式**: Git 仓库同步

## 代码同步规则

### 每次修改代码后必须执行

```bash
# 1. 查看修改
git status

# 2. 添加所有修改
git add .

# 3. 提交修改（使用描述性的 commit message）
git commit -m "描述你的修改"

# 4. 推送到远端仓库
git push origin master
```

### 在远端服务器拉取更新

```bash
# SSH 登录到远端服务器
ssh 5090

# 进入项目目录
cd ~/federal

# 拉取最新代码
git pull origin master
```

## Kiro 工作流程

作为 AI 助手，在每次修改代码文件后，我需要：

1. **修改文件** - 使用 fsWrite, strReplace 等工具
2. **提交更改** - 使用 executeBash 执行 git 命令
3. **告知用户** - 提示用户在远端拉取更新

### 标准操作流程

```bash
# 添加所有修改
git add .

# 提交（使用有意义的消息）
git commit -m "feat: 添加模型测试脚本"

# 推送到远端
git push origin master
```

## Commit Message 规范

使用语义化的 commit message：

- `feat:` - 新功能
- `fix:` - 修复 bug
- `docs:` - 文档更新
- `refactor:` - 代码重构
- `test:` - 测试相关
- `chore:` - 构建/工具相关

示例：
```
feat: 添加快速模型测试脚本
fix: 修复 tokenizer 加载问题
docs: 更新 DEPLOYMENT.md
refactor: 优化数据加载逻辑
```

## 注意事项

1. **不要提交大文件** - 模型文件、结果文件应该在 .gitignore 中
2. **不要提交敏感信息** - API keys, tokens 等
3. **保持提交原子性** - 每次提交只做一件事
4. **写清楚 commit message** - 方便追踪和回滚

## 远端服务器工作流程

### 首次设置

```bash
# 克隆仓库
git clone <repository_url> ~/federal
cd ~/federal

# 安装依赖
pip install -r requirements.txt
```

### 日常使用

```bash
# 拉取最新代码
git pull origin master

# 运行测试/训练
python scripts/quick_test.py
python experiments/exp001_dual_adapter_fl/train.py
```

### 如果有本地修改冲突

```bash
# 保存本地修改
git stash

# 拉取远端更新
git pull origin master

# 恢复本地修改
git stash pop
```

## 文件同步策略

### 需要同步的文件（通过 Git）
- 源代码: `src/`, `tools/`, `experiments/`
- 脚本: `scripts/`
- 配置: `*.yaml`, `requirements.txt`
- 文档: `README.md`, `DEPLOYMENT.md`, `.kiro/`
- 数据: `data/rule_data/`, `data/test/` (小文件)

### 不需要同步的文件（在 .gitignore 中）
- 结果: `results/`
- 模型: `*.safetensors`, `*.bin`
- 缓存: `__pycache__/`, `.cache/`
- 日志: `*.log`
- 环境: `venv/`, `.env`

## 故障排查

### 推送失败

```bash
# 检查远端状态
git remote -v

# 拉取远端更新后再推送
git pull origin master
git push origin master
```

### 拉取失败

```bash
# 查看本地修改
git status

# 如果有冲突，手动解决或重置
git reset --hard origin/master
```
