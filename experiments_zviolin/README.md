# ES-MoE 自适应推理优化实验

> **项目**：2026 腾讯犀牛鸟开源人才培养计划 — 实战项目一
> **关联 Issue**：https://github.com/Tencent/YOLO-Master/issues/54
> **关联项目页**：https://opensource.tencent.com/summer-of-code/project/108/practice
> **分支**：`esmoe-experiments`

## 项目概述

基于 YOLO-Master 的 ES-MoE（Efficient Sparse Mixture-of-Experts）架构，针对特定应用场景设计并优化自适应推理策略，实现计算资源的动态分配。

ES-MoE 通过 4 个不同核尺寸（3×3/5×5/7×7/9×9）的 Depthwise Separable Conv 专家，
结合 Soft-to-Hard Top-K 切换，实现按需激活的稀疏推理。

## 目录结构总览

```
experiments_zviolin/
├── README.md                        # 本文件（成果索引）
├── runs/                            # 实验产物（详见下方"目录详解"）
└── weights/                         # （可选）统一权重副本

# 另有训练权重存放于 ultralytics 默认位置：
runs/detect/experiments_zviolin/runs/   # K=1/K=3/Shared 训练权重
```

> ⚠️ **注意**：由于 ultralytics `save_dir` 前缀差异，训练权重分散在**两处**：
> - `experiments_zviolin/runs/esmoe_v0_k2-2`、`esmoe_v0_k2_bl01/bl20`（早期训练，project 直接为 `experiments_zviolin/runs`）
> - `runs/detect/experiments_zviolin/runs/esmoe_v0_k1-3`、`esmoe_v0_k3`、`esmoe_v0_shared`（修复后重训，多一层 `runs/detect` 前缀）

---

## 目录详解（每个文件夹/文件用途）

### 一、训练权重与训练日志

#### `runs/detect/experiments_zviolin/runs/`（修复后重训，最终结果）

| 目录/文件 | 内容 |
|-----------|------|
| `esmoe_v0_k1-3/` | **K=1 最终模型**（top_k=1，mAP50-95=16.13%，8/13 训练完成）|
| `esmoe_v0_k3/` | **K=3 最终模型**（top_k=3，mAP50-95=15.20%，8/14 修复后训练完成）|
| `esmoe_v0_shared/` | **SharedExpertESMoE 跨尺度共享模型**（mAP50-95=15.16%，8/14 完成）|
| └─ `weights/best.pt` | 最佳权重（验证集 mAP 最高）|
| └─ `weights/last.pt` | 最终权重（epoch 100）|
| └─ `weights/last_healthy.pt` | NaN 恢复健康快照 |
| └─ `results.csv` | 100 行训练日志（loss/mAP/lr）|
| └─ `args.yaml` | 完整训练配置 |
| └─ `results.png` | 训练曲线图 |

#### `experiments_zviolin/runs/`（早期训练 + 修复前）

| 目录/文件 | 内容 |
|-----------|------|
| `esmoe_v0_k2-2/` | **K=2 基线**（top_k=2，mAP50-95=17.27%，精度最优）⭐ |
| `esmoe_v0_k2_bl01/` | K=2 + balance_loss=0.01（弱均衡，mAP50-95=17.319%）|
| `esmoe_v0_k2_bl20/` | K=2 + balance_loss=2.0（强均衡，mAP50-95=16.62%）|
| `v084/` | Issue #54 MoE 基线（mAP50-95=16.84%）|
| `v08_mot6/` | Issue #54 MoT 变体（mAP50-95=16.93%）|
| `v08_moa2/` | Issue #54 MoA 变体（mAP50-95=16.80%）|
| `v08_moe_mot_shared/` | Issue #54 方案 D（跨尺度共享，mAP50-95=16.80%）|

> 以上训练目录均含：`results.csv`（训练日志）、`args.yaml`（配置）、`results.png`（曲线）、`weights/best.pt` + `last.pt`。

### 二、路由诊断（验收 #2 输入数据）

| 目录/文件 | 内容 |
|-----------|------|
| `esmoe_routing_k1/` | **K=1 路由诊断**（8/14 实测，路由熵 H=0.198）|
| `esmoe_routing_k2/` | **K=2 路由诊断**（8/14 实测，H=0.490）|
| `esmoe_routing_k3/` | **K=3 路由诊断**（8/14 实测，H=0.741）|
| `esmoe_routing_shared/` | **Shared 路由诊断**（8/14 实测，H=0.430）|
| `esmoe_routing_bl01/` | balance=0.01 路由诊断（早期）|
| `esmoe_routing_bl20/` | balance=2.0 路由诊断（早期）|
| └─ `routing_summary.csv` | 16 行路由数据（4 层 × 4 专家，含 mean_weight/is_top_k）|
| └─ `routing_heatmap.png` | 专家激活热力图 |
| └─ `recommendations.json` | 场景化推荐 |

### 三、性能 Benchmark（验收 #4）

| 目录/文件 | 内容 |
|-----------|------|
| `edge_benchmark_full/edge_benchmark_cpu_640.csv` | **6 变体 CPU 延迟**（8/14 第二轮实测：K=1 56.69ms / K=2 59.94ms / K=3 59.44ms / bl01 60.17ms / bl20 63.26ms / shared 59.05ms；第一轮 58.55/90.63/69.75/69.80/69.30/63.68ms，CPU 波动 ±30%）|
| `latency_cpu_640.csv` | 早期延迟数据（2026-08-09，历史）|
| `esmoe_inference_mode/inference_modes.csv` | 推理模式对比（Hard Top-K vs Dense Softmax，早期）|
| `inference_modes_k1/inference_modes.csv` | **K=1 同权重 Hard vs Dense**（GPU 27.34/35.85ms，-23.7%）|
| `inference_modes_k3/inference_modes.csv` | **K=3 同权重 Hard vs Dense**（GPU 31.12/34.78ms，-10.5%）|
| `inference_modes_shared/inference_modes.csv` | **Shared 同权重 Hard vs Dense**（GPU 29.20/33.93ms，-13.9%；CPU 65.56/70.51ms，-7.0%）|
| `inference_modes_k1_cpu/`、`k2_cpu/`、`k3_cpu/` | **K=1/K=2/K=3 CPU 推理模式**（-18.9% / -13.3% / -15.7%，2026-08-15）|

### 四、动态 Top-K（任务 4 创新）

| 目录/文件 | 内容 |
|-----------|------|
| `dynamic_topk_k1/`、`k2/`、`k3/`、`bl01/`、`bl20/` | 各变体动态 Top-K 验证（含 `dynamic_topk_per_image.csv` + 50 张合成图）|

> 实测：简单图像 34% 触发 K=1、66% 触发 K=2（avg_top_k=1.66）

### 五、Issue #54 历史分析（对比参考）

| 目录/文件 | 内容 |
|-----------|------|
| `mot_routing/`、`mot_routing_dryrun/`、`mot_routing_shared/` | MoT / 方案 D 路由诊断 |
| `routing_compare/`、`routing_compare_mot/` | 路由对比 |
| `scene_complexity_mot/` | 场景复杂度与路由关联 |
| `esmoe_ablation/` | 测试日志（test_all_results.txt / test_results.txt）|

---

## 关键结果（2026-08-14 修复后真实数据）

| 变体 | top_k | mAP50-95 | mAP50 | CPU 延迟（第二轮） | 路由熵 H_norm |
|------|-------|---------|-------|-------------------|--------------|
| **K=1** | 1 | 16.13% | 28.79% | 56.69ms | 0.198 |
| **K=2（基线）** ⭐ | 2 | **17.27%** | **30.55%** | 59.94ms | 0.490 |
| **K=3** | 3 | 15.20% | 27.44% | 59.44ms | 0.741 |
| **Shared** | 2 | 15.16% | 27.30% | 59.05ms | 0.430 |
| K=2_bl01 | 2 | 17.319% | 30.602% | 60.17ms | — |
| K=2_bl20 | 2 | 16.62% | 29.62% | 63.26ms | — |

**结论**：
- **精度最优**：K=2（17.27%，相对 MoE 基线 v084 +0.43%）
- **稀疏化收益**（2026-08-15 GPU+CPU 交叉验证）：同权重 Hard Top-K vs 强制 Dense——GPU：K=1 **-23.7%** / K=3 **-10.5%** / Shared **-13.9%**；CPU：K=1 **-18.9%** / K=2 **-13.3%** / K=3 **-15.7%** / Shared **-7.0%**——稀疏化有效（跨模型横向对比受 CPU 波动 ±30% 干扰，不可靠）
- **Shared 定位**：架构创新探索（跨尺度共享，专家对象 -25%，参数 -3.1%），精度 -2.11pp 非生产首选

> ⚠️ **重要修复记录**（2026-08-14）：
> 1. `dynamic_threshold=0.4` 二次过滤 bug 导致 K=3 推理 mAP 崩到 1% → 修复为 0.0 后恢复 15.20%
> 2. SharedExpertESMoE 模块 + YAML 修复 5 处缺陷（详见 [06-任务3](../../Docs/06-任务3-性能指标验证.md)）

## 使用方法

### 1. 路由诊断（任务 2）

```bash
python scripts/diagnose_esmoe_routing.py \
    --model runs/detect/experiments_zviolin/runs/esmoe_v0_k1-3/weights/best.pt \
    --dry-run --device cpu --plot \
    --output experiments_zviolin/runs/esmoe_routing_k1
```

### 2. 性能对比（任务 3/4）

```bash
python scripts/compare_esmoe_edge.py \
    --models esmoe_v0_k1 esmoe_v0_k2-2 esmoe_v0_k3 esmoe_v0_shared \
    --imgsz 640 --warmup 100 --reps 500 --device cpu
```

### 3. 动态 Top-K（任务 4）

```bash
python scripts/dynamic_topk_router.py --verify-distribution \
    --model runs/detect/experiments_zviolin/runs/esmoe_v0_k1-3/weights/best.pt \
    --num-samples 50
```

### 4. 边界测试（任务 5）

```bash
pytest tests/test_esmoe.py tests/test_shared_expert_esmoe.py -v
```

## 与 Issue #54 的关联

本项目作为 Issue #54（MoT 消融实验）的延伸，验证了 ES-MoE 架构在 4 专家稀疏计算场景下的有效性：

- **复用资产**：路由分析脚本（diagnose_mot_routing.py）、测试框架（30/30 测试经验）、Benchmark 体系
- **新增能力**：ES-MoE 4 专家路由诊断、动态 Top-K、Soft-to-Hard 切换验证
- **创新点**：多尺度专家协同（3×3/5×5/7×7/9×9）相比 MoT 注意力变体更直观；跨尺度专家共享（SharedExpertESMoE）

## 详细文档

完整任务分析见：[practices/Docs/](../../Docs/)（9 篇文档）

- 00-实战项目原文.md
- 01-项目分析说明.md
- 02-项目结合分析.md
- 03-实战项目任务分析与实验文档.md
- 04-任务1-ES-MoE训练.md
- 05-任务2-专家利用率分析与负载均衡调优.md
- 06-任务3-性能指标验证.md
- 07-任务4-推理效率优化.md
- 08-任务5-可视化分析与交付物.md
