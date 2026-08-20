# GitHub 仓库与项目结构说明

## 一、仓库架构设计

### 1.1 双仓库策略

本项目采用**双仓库并行**的协作模式：

| 仓库类型 | 用途 | 位置 | 说明 |
|---------|------|------|------|
| **代码贡献仓库** | 向 YOLO-Master 提交 PR | `github.com/164321595/YOLO-Master` | Fork 自原仓库，用于代码贡献 |
| **实验成果仓库** | 展示实验结果和文档 | `github.com/164321595/Rhino-bird` | 独立仓库，用于展示完整项目过程 |

### 1.2 为什么采用双仓库

- **代码贡献仓库**：遵循开源项目规范，通过 PR 向原项目贡献代码（边界测试、YAML 配置等）
- **实验成果仓库**：保存训练日志、路由分析、技术文章等实验过程材料，不受原项目目录结构限制

## 二、代码贡献仓库（YOLO-Master Fork）

> **本项目的 Fork 仓库**：
> - Fork 仓库：`https://github.com/164321595/YOLO-Master`
> - 功能分支：`https://github.com/164321595/YOLO-Master/tree/issue-54-mot-experiments`
> - 原仓库（上游）：`https://github.com/Tencent/YOLO-Master`

### 2.1 Fork 与克隆

```powershell
# 1. 在 GitHub 上 Fork 原仓库
# 访问: https://github.com/Tencent/YOLO-Master
# 点击右上角 "Fork" 按钮（已 Fork 到 Zviolin/YOLO-Master）

# 2. 克隆 Fork 后的仓库
cd g:\Codes\OpenSource\Rhino-bird\Codes
git clone https://github.com/164321595/YOLO-Master.git
cd YOLO-Master

# 3. 添加上游远程仓库（用于同步原项目更新）
git remote add upstream https://github.com/Tencent/YOLO-Master.git

# 4. 验证远程仓库配置
git remote -v
# 应显示:
# origin    https://github.com/164321595/YOLO-Master.git (fetch)
# origin    https://github.com/164321595/YOLO-Master.git (push)
# upstream  https://github.com/Tencent/YOLO-Master.git (fetch)
# upstream  https://github.com/Tencent/YOLO-Master.git (push)
```

### 2.2 分支管理

#### 分支命名规范

```bash
# 主分支（保持与上游同步）
main

# 功能分支（用于 Issue #54，本项目使用）
issue-54-mot-experiments
# → https://github.com/164321595/YOLO-Master/tree/issue-54-mot-experiments

# 其他可能的分支
issue-54-boundary-tests       # 边界测试专用
issue-54-yaml-configs         # YAML 配置专用
```

#### 创建与切换分支

```powershell
# 1. 确保 main 分支是最新的
git checkout main
git pull upstream main

# 2. 创建新分支
git checkout -b issue-54-mot-experiments

# 3. 推送到远程仓库（首次创建）
git push -u origin issue-54-mot-experiments

# 4. 后续切换分支
git checkout issue-54-mot-experiments
```

### 2.3 文件添加规范

#### 应添加的文件

| 文件类型 | 目录位置 | 说明 | 示例 |
|---------|---------|------|------|
| 实验脚本 | `scripts/` | 训练、分析、可视化脚本 | `scripts/compare_mot_ablation.py` |
| 测试文件 | `tests/` | 边界测试、单元测试 | `tests/test_mot.py` |
| 配置文件 | `ultralytics/cfg/models/master/v0_8/det/` | YAML 模型配置 | `yolo-master-mot-n.yaml`、`yolo-master-moe-mot-shared-n.yaml` |
| 模块代码 | `ultralytics/nn/modules/` | 自定义神经网络模块 | `mot/mo_t_block.py`、`moe/shared_expert_moe.py` |

#### 不应添加的文件

| 文件类型 | 说明 | 处理方式 |
|---------|------|---------|
| 训练权重 | `.pt`、`.pth` 文件 | 使用 Git LFS 或外部存储 |
| 训练日志 | `runs/` 目录下的日志 | 添加到 `.gitignore` |
| 临时文件 | `__pycache__/`、`.DS_Store` | 添加到 `.gitignore` |
| 数据集 | 图像、标注文件 | 使用外部链接或数据集仓库 |

### 2.4 提交 PR 流程

```powershell
# 1. 添加修改的文件（避免使用 git add .）
git add scripts/compare_mot_ablation.py
git add tests/test_mot.py
git add ultralytics/cfg/models/master/v0_8/det/yolo-master-moe-mot-shared-n.yaml
git add ultralytics/nn/modules/moe/shared_expert_moe.py

# 2. 提交更改（使用中文提交信息）
git commit -m "feat: 添加 MoT 架构消融实验脚本和边界测试"

# 3. 推送到远程分支
git push origin issue-54-mot-experiments

# 4. 在 GitHub 上创建 PR
# 访问: https://github.com/164321595/YOLO-Master/pulls
# 点击 "New pull request"
# 选择: base repository: Tencent/YOLO-Master, base: main
#        head repository: Zviolin/YOLO-Master, compare: issue-54-mot-experiments
# PR 描述中使用: Closes #54
```

### 2.5 同步上游更新

```powershell
# 1. 获取上游更新
git fetch upstream

# 2. 合并到本地 main 分支
git checkout main
git merge upstream/main

# 3. 推送到远程 Fork
git push origin main

# 4. 将更新合并到功能分支
git checkout issue-54-mot-experiments
git merge main
```

## 三、实验成果仓库（Rhino-bird）

### 3.1 仓库结构

```
Rhino-bird/
├── Docs/                          # 文档目录
│   ├── Disscusion/                # 讨论与分析文档
│   │   ├── 00-犀牛鸟计划总览.md
│   │   ├── 01-Issue汇总与简介.md
│   │   ├── 02-Issue54排除法选择.md
│   │   ├── 03-Issue54完整课题文档.md
│   │   ├── 04-项目申请书.md
│   │   ├── 05-任务1-三变体训练.md
│   │   ├── 06-任务2-性能对比测量.md
│   │   ├── 07-任务3-路由可解释性分析.md
│   │   ├── 08-任务4-混合架构探索.md          # 含方案 D 创新
│   │   ├── 09-任务5-边界测试与修复.md
│   │   └── 10-任务6-交付物清单.md
│   ├── README.md                  # Docs 总入口
│   └── configs/                   # 配置说明文档
│       ├── 00-项目配置总览.md
│       ├── 01-设备硬件配置说明.md
│       ├── 02-Conda环境配置说明.md
│       └── 03-GitHub仓库与项目结构说明.md
├── Codes/                         # 代码目录
│   └── YOLO-Master/               # Zviolin Fork: https://github.com/164321595/YOLO-Master
├── Experiments/                   # 实验结果
│   ├── logs/                      # 训练日志
│   ├── weights/                   # 模型权重
│   └── results/                   # 实验结果图表
└── README.md                      # 项目介绍
```

### 3.2 实验结果组织

#### 训练日志

```
Experiments/logs/
├── moe_baseline/                  # MoE 基线训练日志 (v08)
│   ├── results.csv
│   ├── results.png
│   └── args.yaml
├── mot_variant/                   # MoT 变体训练日志 (v08_mot)
│   ├── results.csv
│   ├── results.png
│   └── args.yaml
├── moa_variant/                   # MoA 变体训练日志 (v08_moa)
│   ├── results.csv
│   ├── results.png
│   └── args.yaml
└── mot_shared/                    # 方案 D 训练日志 (v08_moe_mot_shared)
    ├── results.csv
    ├── results.png
    └── args.yaml
```

#### 路由分析

```
Experiments/results/
├── routing_analysis/              # 路由分析结果（任务 3）
│   ├── expert_usage_heatmap.png
│   ├── routing_entropy.png
│   └── token_flow_sankey.png
├── latency_benchmark/             # 性能对比（任务 2）
│   ├── latency_cpu_640.csv
│   ├── latency_cuda_0_640.csv
│   └── build_summary.csv
└── boundary_tests/                # 边界测试结果（任务 5）
    ├── test_mot_window_size.py
    └── test_mot_odd_input.py
```

### 3.3 文档编写规范

#### 文件命名

- 使用中文描述性命名
- 序号前缀便于排序：`01-`、`02-`、`03-`...
- 避免特殊字符：`@`、`#`、`&` 等

#### 内容结构

```markdown
# 标题

## 一、背景介绍
说明任务背景和目标

## 二、实验设计
描述实验方案和方法

## 三、实验结果
展示实验数据和图表

## 四、分析与讨论
分析结果，讨论创新点

## 五、总结与展望
总结经验，提出改进方向
```

## 四、本地开发工作流

### 4.1 日常开发流程

```powershell
# 1. 激活 Conda 环境
conda activate yolo-master

# 2. 进入 YOLO-Master 目录
cd g:\Codes\OpenSource\Rhino-bird\Codes\YOLO-Master

# 3. 确保在功能分支上
git status
# 应显示:
# On branch issue-54-mot-experiments
# 如果不是，切换：
git checkout issue-54-mot-experiments

# 4. 编写代码、运行实验
# ... 实验代码 ...

# 5. 提交更改
git add <修改的文件>
git commit -m "feat: 添加 MoT 路由分析可视化功能"
git push origin issue-54-mot-experiments

# 6. 在 GitHub 上创建 PR（如果是首次推送）
# 或更新现有 PR
# PR 链接：https://github.com/164321595/YOLO-Master/pull/new/issue-54-mot-experiments
```

### 4.2 实验记录流程

```powershell
# 1. 运行训练脚本
python scripts/compare_mot_ablation.py --train --models v08_mot --data ultralytics\cfg\datasets\VisDrone.yaml --epochs 100 --imgsz 640 --batch 8 --device 0 --project runs\mot_ablation

# 2. 保存训练日志到实验仓库
Copy-Item -Path runs\mot_ablation\v08_mot\results.csv -Destination g:\Codes\OpenSource\Rhino-bird\Experiments\logs\mot_variant\

# 3. 保存可视化结果
Copy-Item -Path runs\mot_ablation\v08_mot\results.png -Destination g:\Codes\OpenSource\Rhino-bird\Experiments\results\routing_analysis\

# 4. 在实验仓库提交更改
cd g:\Codes\OpenSource\Rhino-bird
git add Experiments/
git commit -m "feat: 添加 MoT 变体训练日志和路由分析结果"
git push origin main
```

## 五、协作注意事项

### 5.1 避免冲突

- **不要修改原项目核心代码**：除非是修复 bug 或添加功能，否则不要修改 `ultralytics/` 目录下的代码
- **保持 main 分支干净**：所有实验代码放在功能分支，不要直接推送到 main
- **及时同步上游更新**：定期执行 `git pull upstream main` 保持代码最新

### 5.2 PR 提交技巧

- **小步快跑**：每个 PR 只解决一个具体问题，避免大杂烩
- **清晰描述**：PR 描述中说明解决了什么问题、如何测试、相关 issue 编号
- **关联 Issue**：在 PR 描述中使用 `Closes #54` 自动关联 issue
- **等待审查**：提交后耐心等待维护者审查，及时响应反馈

### 5.3 实验成果展示

- **可复现性**：提供完整的实验配置和脚本，确保他人可以复现结果
- **可视化**：使用图表、表格直观展示实验结果
- **对比分析**：与基线模型进行对比，突出 MoT 架构的优势
- **技术文章**：撰写通俗易懂的技术文章，便于评审理解

## 六、常见问题

### 6.1 如何处理大文件（模型权重）

**方案一：使用 Git LFS**
```powershell
# 安装 Git LFS
git lfs install

# 跟踪 .pt 文件
git lfs track "*.pt"

# 提交 .gitattributes
git add .gitattributes
git commit -m "chore: 添加 Git LFS 跟踪 .pt 文件"
```

**方案二：使用外部存储**
- 将权重文件上传到 Google Drive、百度网盘等
- 在仓库中提供下载链接

### 6.2 如何处理敏感信息

```powershell
# 1. 检查是否提交了敏感文件
git log --all --full-history -- .env

# 2. 从历史中移除（危险操作，谨慎使用）
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch .env" --prune-empty --tag-name-filter cat -- --all

# 3. 推送到远程（覆盖历史）
git push origin --force --all
```

**预防措施**：
- 在 `.gitignore` 中添加 `.env`、`*.key` 等敏感文件
- 使用环境变量管理 API 密钥

### 6.3 如何处理冲突

```powershell
# 1. 获取上游更新
git fetch upstream

# 2. 合并到功能分支
git checkout issue-54-mot-experiments
git merge upstream/main

# 3. 如果有冲突，手动解决
# 编辑冲突文件，删除冲突标记

# 4. 标记为已解决
git add <冲突文件>
git commit -m "merge: 解决与上游的合并冲突"
```

## 七、PR 提交示例（本项目实际使用）

### 7.1 PR #1: 边界测试与修复（任务 5）

**标题**：`feat: 补全 MoT/MoA 边界测试与稳定性修复`

**描述模板**：
```markdown
## 概述
解决 Issue #54 任务 5：补全 MoT/MoA 边界测试并修复发现的缺陷

## 改动
1. tests/test_mot.py: 新增 3 个边界测试
   - window_size > feature_map 降级处理
   - _WindowTransformerExpert shift 奇数尺寸边界
   - exploration_eps eval 模式禁用
2. tests/test_moa.py: 修复 NameError: name 'nn' is not defined（添加 import torch.nn as nn）

## 测试
- tests/test_mot.py: 11 passed
- tests/test_moa.py: 16 passed

## 关联
Closes #54
```

### 7.2 PR #2: 混合架构 YAML 配置（任务 4）

**标题**：`feat: 新增方案 D 跨尺度专家池共享 YAML 配置`

**描述模板**：
```markdown
## 概述
方案 D 创新：Cross-Scale Expert Sharing — P3 和 P4 共享同一 expert pool

## 改动
1. 新增 ultralytics/cfg/models/master/v0_8/det/yolo-master-moe-mot-shared-n.yaml
2. 新增 ultralytics/nn/modules/moe/shared_expert_moe.py 模块
3. 注册新变体 v08_moe_mot_shared 到 compare_mot_ablation.py

## 性能（vs MoT 基线）
- 参数量: 3.71M → 3.01M (-19.1%)
- CPU 延迟: 215.91ms → 182.15ms (-15.6%)
- mAP50-95: 16.93% → 16.80% (-0.13%)

## 关联
Closes #54
```

## 八、总结

✅ **双仓库策略**：代码贡献仓库用于 PR，实验成果仓库用于展示。

**本项目仓库链接**：
- Fork 仓库：`https://github.com/164321595/YOLO-Master`
- 功能分支：`https://github.com/164321595/YOLO-Master/tree/issue-54-mot-experiments`
- 实验仓库：`https://github.com/164321595/Rhino-bird`

**关键要点**：
- Fork 原仓库，创建 `issue-54-mot-experiments` 功能分支
- 只添加必要的文件（脚本、测试、配置），避免提交权重和日志
- 实验结果保存在独立仓库，便于展示完整项目过程
- 遵循提交规范，保持代码整洁和可复现性

**下一步**：
- 完成 Fork 和分支创建
- 开始编写实验脚本和测试
- 定期同步上游更新，保持代码最新