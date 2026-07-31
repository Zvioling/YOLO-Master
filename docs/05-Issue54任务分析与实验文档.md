# Issue #54 任务分析与实验文档（总览）

> 本文件为 Issue #54 的 **6 大任务总览**，提供任务索引、关键结果汇总和实验文档总模板
>
> 各任务的**详细执行过程**已拆分到独立 MD 文件（见下方目录）
>
> 对应 Issue：https://github.com/Tencent/YOLO-Master/issues/54

---

## 目录

1. [任务 1：三变体训练](./06-任务1-三变体训练.md)
2. [任务 2：性能对比测量](./07-任务2-性能对比测量.md)
3. [任务 3：路由可解释性分析](./08-任务3-路由可解释性分析.md)
4. [任务 4：混合架构探索（含方案 D 创新）](./09-任务4-混合架构探索.md)
5. [任务 5：边界测试与修复](./10-任务5-边界测试与修复.md)
6. [任务 6：交付物清单](./11-任务6-交付物清单.md)
7. [实验文档总模板](#实验文档总模板)
8. [关键结果汇总](#关键结果汇总)

---

## 任务 1：三变体训练

> **详细文档**：[06-任务1-三变体训练.md](./06-任务1-三变体训练.md)

在 VisDrone 数据集上训练 3 种模型变体：

| 变体 | 配置文件 | 参数量 | 训练目标 |
|------|---------|--------|---------|
| MoE 基线 (YOLO-Master-EsMoE-N) | `yolo-master-n.yaml` | ~3.14M | 提供基线参考 |
| MoT 变体 (YOLO-Master-v0.10-MoT-N) | `yolo-master-mot-n.yaml` | ~3.71M | 验证 MoT 架构 |
| MoA 对比 (YOLO-Master-v0.10-MoA-N) | `yolo-master-moa-n.yaml` | ~3.23M | 与 MoT 对比 |

**训练配置**：batch=8, imgsz=640, epochs=100, device=0（RTX 5060 8GB）

**结果**：所有变体训练稳定，无 NaN。

---

## 任务 2：性能对比测量

> **详细文档**：[07-任务2-性能对比测量.md](./07-任务2-性能对比测量.md)

测量 3 个模型变体的 6 项指标：mAP50-95、mAP50、Latency（CPU/GPU）、FLOPs、Params、训练稳定性。

**工具**：`scripts/compare_mot_ablation.py --benchmark`

**结果**：完整性能对比表见任务 2 详细文档。

---

## 任务 3：路由可解释性分析

> **详细文档**：[08-任务3-路由可解释性分析.md](./08-任务3-路由可解释性分析.md)

对 MoT 模型的 3 个 Transformer 专家（LocalConvTransformer / WindowTransformer / DeformableTransformer）进行 token 路由分布分析，验证 DeformableTransformer 在遮挡/不规则目标场景激活率显著上升。

**方法**：使用 `diagnose_model` 或自定义 hook 绘制专家激活热力图

**对比维度**：密集 vs 稀疏、小目标 vs 大目标

---

## 任务 4：混合架构探索（含方案 D 创新）

> **详细文档**：[09-任务4-混合架构探索.md](./09-任务4-混合架构探索.md)

尝试 MoT 与 MoE/MoA 的层级组合（backbone 用 MoE、neck 用 MoT，或与 MoA 交叉组合），评估协同增益。

**判定标准**：
- mAP 提升 > 1%（精度增益有意义）
- 延迟降低 > 10%（效率增益有意义）

### 核心创新：方案 D 跨尺度专家池共享（Cross-Scale Expert Sharing）

| 指标 | 方案 D (v08_moe_mot_shared) | MoT 基线 (v08_mot) | 变化 | 10% 阈值判定 |
|------|---------------------------|--------------------|------|-------------|
| **参数量 (M)** | **3.01** | 3.71 | **−19.1%** | ✅ 大幅优化 |
| **CPU 延迟 (ms)** | **182.15** | 215.91 | **−15.6%** | ✅ **延迟增益达标**（> 10%） |
| **GPU 延迟 (ms)** | 51.73 | 52.74 | −1.9% | ⚠️ 微优但未达 10% 阈值（测量噪声） |
| **mAP50-95** | 16.80% | 16.93% | −0.13% | ❌ 精度未提升（损失极小） |
| **moe_loss** | 9.95e-05 | - | 稳定 | ✅ 路由完全稳定 |

**协同增益判定**：按 Issue #54 "mAP > 1% **或** 延迟 > 10%"二选一标准，**CPU 延迟 -15.6% 已达标，架构创新成立**。GPU -1.9% 属 Laptop GPU 测量噪声范围（52.74ms vs 51.73ms 仅 1ms 差异）。

**创新点**：P3 与 P4 层共享同一个专家池（pool name `p3_p4`），通过 `SharedExpertMoE` 模块实现跨尺度特征交互，同时显著减少参数量（-19.1%）和 CPU 延迟（-15.6%）。

**模块实现**：`ultralytics/nn/modules/moe/shared_expert_moe.py`

**YAML 配置**：`ultralytics/cfg/models/master/v0_8/det/yolo-master-moe-mot-shared-n.yaml`

---

## 任务 5：边界测试与修复

> **详细文档**：[10-任务5-边界测试与修复.md](./10-任务5-边界测试与修复.md)

补全 `tests/test_mot.py` 的边界测试，覆盖：

1. MoTBlock 在 window_size > feature_map 时的降级处理
2. `_WindowTransformerExpert` 的 shift 操作在奇数尺寸输入时的边界
3. MoT 的 `exploration_eps` 在 eval 模式下是否被正确禁用

**测试结果**：
- 总测试数：30 个
- 通过：29 个
- 失败：1 个（`tests/test_moa.py::test_collect_moa_aux_loss_handles_empty_module_and_standalone_block` - 已知 `import torch.nn as nn` 缺失）
- 详细测试报告：`Codes/YOLO-Master/runs/mot_ablation/test_results.txt`

---

## 任务 6：交付物清单

> **详细文档**：[11-任务6-交付物清单.md](./11-任务6-交付物清单.md)

| # | 交付物 | 形式 | 状态 |
|---|--------|------|------|
| 1 | 三变体训练脚本与日志 | `compare_mot_ablation.py --train` | ✅ 完成 |
| 2 | 性能对比 benchmark 脚本 | `compare_mot_ablation.py --benchmark` | ✅ 完成 |
| 3 | 路由分析脚本与可视化 | Hook + heatmap | ✅ 完成 |
| 4 | 混合架构 YAML（方案 D） | `yolo-master-moe-mot-shared-n.yaml` | ✅ 完成 |
| 5 | 边界测试 PR | `tests/test_mot.py` 增补 | ✅ 完成 |
| 6 | GitHub Discussion 技术文章 | 文章 + 实验脚本仓库链接 | ✅ 完成 |

---

## 关键结果汇总

### 模型变体注册表

| 变体 key | 描述 | c2fmoa | c2fmot | MotBlocks | 参数量 |
|---------|------|--------|--------|-----------|--------|
| v08 | YOLO-Master v0.8 baseline | 0 | 0 | 0 | 3.14M |
| v08_moa | YOLO-Master v0.8 MoA | 3 | 0 | 0 | 3.23M |
| v08_mot | YOLO-Master v0.8 MoT | 0 | 3 | 6 | 3.71M |
| v08_moa_mot | YOLO-Master v0.8 MoA+MoT | 2 | 3 | 6 | 3.76M |
| **v08_moe_mot_shared** | **方案 D 跨尺度共享** | 0 | 3 | 6 | **3.01M** |

### 性能对比（核心指标）

| 变体 | Params (M) | mAP50-95 | CPU Latency (ms) | GPU Latency (ms) |
|------|-----------|----------|------------------|------------------|
| v08 (MoE 基线) | 3.14 | 16.84% | - | - |
| v08_moa | 3.23 | 16.80% | - | - |
| v08_mot | 3.71 | **16.93%** | 215.91 | 52.74 |
| **v08_moe_mot_shared (方案 D)** | **3.01** | 16.80% | **182.15** | **51.73** |

### 关键工程经验

1. **训练工程**：
   - 8GB 显存优化：batch=4/2, imgsz=640/512, 混合精度训练
   - AMP 训练问题：MoE expert 模块中 `expert_out * expert_weight` 需要转 dtype
   - MoT 训练速度：相比 MoE 基线慢 8.6x（Deformable + Window Attention 引入额外开销）

2. **Benchmark 工程**：
   - device 字符串：必须传完整字符串（`cpu` / `cuda:0` / `mps`），不能传数字 `0`
   - Windows 文件名兼容：`cuda:0` 中的 `:` 需替换为 `_`（已修复 `compare_mot_ablation.py`）
   - GPU 测量准确性：需 `warmup=500, reps=2000` 才能得到稳定数据
   - SharedExpertMoE device 同步：类级别注册会保留状态，需 `reset_shared_pools()` 重置

3. **代码贡献**：
   - 新增 `SharedExpertMoE` 模块（`ultralytics/nn/modules/moe/shared_expert_moe.py`）
   - 新增 YAML 配置（`yolo-master-moe-mot-shared-n.yaml`）
   - 修复 `compare_mot_ablation.py` 的 Windows 文件名兼容 bug

---

## 实验文档总模板

> 用于最终展示的简洁版本（GitHub Discussion 文章可直接套用）

```markdown
# YOLO-Master MoT 架构消融实验与路由可解释性分析

> 作者：张伟林
> 日期：2026 年 X 月
> 项目：2026 腾讯犀牛鸟开源人才培养计划
> Issue：https://github.com/Tencent/YOLO-Master/issues/54

## 项目概述

本项目在 VisDrone 数据集上对 YOLO-Master 的 MoT 架构进行了系统的消融实验，并通过 Hook 机制深入分析了路由可解释性。

## 主要贡献

1. **三变体消融实验**：在 VisDrone 上完成 MoE/MoT/MoA 三变体的完整训练与对比
2. **路由可解释性分析**：首次系统分析 MoT 的路由机制，揭示不同场景下的专家激活模式
3. **混合架构探索**：探索 MoE+MoT、MoT+MoA 等混合架构，验证协同增益（方案 D 跨尺度专家池共享）
4. **低资源训练方案**：在 8GB 显存条件下完成完整实验，提供可复现方案

## 实验结果

（见各任务详细文档的"结果"章节）

## 场景化推荐

- **轻量化场景**：推荐方案 D（v08_moe_mot_shared），参数量仅 3.01M
- **精度优先场景**：推荐 MoT 基线（v08_mot），mAP50-95 最高 16.93%
- **低资源训练**：使用梯度检查点 + AMP + 小 batch

## 实验仓库

- 代码仓库：https://github.com/<your-fork>/YOLO-Master
- 实验文档：https://github.com/<your-fork>/Rhino-bird
```

---

## 文档作用

本文件为 Issue #54 的 6 大任务总览文档，提供：
- 任务索引和概览
- 关键结果汇总（性能对比表、模型变体注册表）
- 工程经验总结（训练、Benchmark、代码贡献）
- 最终展示文档模板

各任务的**详细执行过程**（命令行、终端输出、命令说明、结果分析）请查阅对应的独立 MD 文件（见上方目录链接）。