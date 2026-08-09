# ES-MoE 自适应推理优化实验

> **项目**：2026 腾讯犀牛鸟开源人才培养计划 — 实战项目一
> **关联 Issue**：https://github.com/Tencent/YOLO-Master/issues/54
> **关联项目页**：https://opensource.tencent.com/summer-of-code/project/108/practice
> **分支**：`esmoe-experiments`

## 项目概述

基于 YOLO-Master 的 ES-MoE（Efficient Sparse Mixture-of-Experts）架构，针对特定应用场景设计并优化自适应推理策略，实现计算资源的动态分配。

ES-MoE 通过 4 个不同核尺寸（3×3/5×5/7×7/9×9）的 Depthwise Separable Conv 专家，
结合 Soft-to-Hard Top-K 切换，实现按需激活的稀疏推理。

## 目录结构

```
experiments_zviolin/
├── runs/
│   ├── esmoe_v0_k2-2/                  # ES-MoE 训练（K=2，100 epoch）✅
│   │   ├── results.csv                 # 训练日志（101 行）
│   │   ├── args.yaml                   # 训练配置
│   │   ├── results.png                 # 训练曲线图
│   │   └── weights/
│   │       ├── best.pt                 # 最佳权重
│   │       └── last.pt                 # 最终权重
│   ├── v084/                           # MoE 基线（Issue #54）✅
│   ├── v08_mot6/                       # MoT 变体（Issue #54）✅
│   ├── v08_moe_mot_shared/             # 方案 D（Issue #54）✅
│   ├── esmoe_routing/                  # 路由诊断输出（任务 2）
│   ├── mot_routing/                    # MoT 路由（Issue #54 对比）
│   ├── mot_routing_shared/             # 方案 D 路由（Issue #54 对比）
│   └── routing_compare/                # 合成 vs 真实（Issue #54 对比）
└── README.md
```

## 关键结果（2026-08-09 实测）

| 变体 | mAP50-95 | mAP50 | Precision | Recall | 训练时长 |
|------|---------|-------|-----------|--------|---------|
| **ES-MoE (esmoe_v0_k2)** | **17.27%** | **30.55%** | **43.48%** | **32.70%** | **13.0h** |
| MoE 基线 (v084) | 16.84% | 29.79% | 39.47% | 31.05% | 13.3h |
| MoT (v08_mot6) | 16.93% | 29.77% | 39.37% | 31.22% | 19.5h |
| MoA (v08_moa2) | 16.80% | 29.55% | 40.75% | 30.04% | 10.9h |

**ES-MoE 是 4 个变体中 mAP 最高、Precision 最高的"性价比之王"**。

相对 MoE 基线（v084）：
- mAP50-95：**+0.43%**（超过原文 §1.4 通过标准 +0.27%）
- mAP50：**+0.76%**
- Precision：**+4.01%**（提升最显著）

## 使用方法

### 1. 路由诊断（任务 2）

```bash
python scripts/diagnose_esmoe_routing.py \
    --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \
    --dry-run --device cpu --plot \
    --output experiments_zviolin/runs/esmoe_routing
```

输出：
- `routing_summary.csv`：4 层 × 4 专家 = 16 条路由数据
- `recommendations.json`：4 条场景化推荐
- `routing_heatmap.png`：4 行 × 4 列激活热力图

### 2. 性能对比（任务 3）

```bash
# Benchmark 多个变体
python scripts/compare_esmoe_ablation.py --benchmark \
    --models esmoe_v0_k2 v084 v08_mot v08_moa \
    --imgsz 640 --warmup 500 --reps 2000 --device cpu

# 训练新变体
python scripts/compare_esmoe_ablation.py --train --models esmoe_v0_k1 \
    --data ultralytics/cfg/datasets/VisDrone.yaml \
    --epochs 100 --imgsz 640 --batch 4 --device 0
```

### 3. 动态 Top-K（任务 4）

```bash
# 验证动态 Top-K 分布
python scripts/dynamic_topk_router.py --verify-distribution \
    --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \
    --num-samples 50
```

### 4. 边界测试（任务 5）

```bash
# 仅 ES-MoE 测试
pytest tests/test_esmoe.py -v

# 全部测试（含 Issue #54 继承测试）
pytest tests/test_esmoe.py tests/test_mot.py tests/test_moa.py -v
```

## 与 Issue #54 的关联

本项目作为 Issue #54（MoT 消融实验）的延伸，验证了 ES-MoE 架构在 4 专家稀疏计算场景下的有效性：

- **复用资产**：路由分析脚本（diagnose_mot_routing.py）、测试框架（30/30 测试经验）、Benchmark 体系
- **新增能力**：ES-MoE 4 专家路由诊断、动态 Top-K、Soft-to-Hard 切换验证
- **创新点**：多尺度专家协同（3×3/5×5/7×7）相比 MoT 注意力变体更直观

## 详细文档

完整任务分析见：[practices/Docs/](../../Docs/)（10 篇文档）

- 00-实战项目原文.md
- 01-项目分析说明.md
- 02-项目结合分析.md
- 03-实战项目任务分析与实验文档.md
- 04-任务1-ES-MoE训练.md
- 05-任务2-路由可视化与诊断.md
- 06-任务3-TopK对比与负载均衡调优.md
- 07-任务4-Soft-to-Hard切换与推理优化.md
- 08-任务5-边界测试与稳定性修复.md
- 09-任务6-交付物清单与场景化洞察.md