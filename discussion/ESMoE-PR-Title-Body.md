# PR 描述 (Discussion 风格汇报) — ES-MoE 自适应推理优化（实战项目一）

> **分支**: `feat/esmoe-adaptive-inference`（代码改动）
> **数据分支**: `esmoe-experiments-data`（实验数据）
> **PR 链接**: https://github.com/Zviolin/YOLO-Master/pull/new/feat/esmoe-adaptive-inference

---

## 一、PR Title

```
feat: ES-MoE Adaptive Inference Optimization (Top-K Sparse + Dynamic Routing + Shared Experts)
```

---

## 二、PR Body (Discussion 风格)

```markdown
## 概述

本 PR 基于 YOLO-Master 官方 **ES-MoE（Efficient Sparse Mixture-of-Experts）** 架构，
针对特定应用场景设计并优化自适应推理策略：通过 **Top-K 稀疏路由（K=1/2/3）**、
**Soft-to-Hard 推理切换** 与 **动态 Top-K 路由器** 实现计算资源的动态分配；
并新增 **SharedExpertESMoE 跨尺度专家共享** 模块（参考 Issue #54 方案 D 思路迁移）。

训练与评测基于 VisDrone 2019（航拍密集小目标），硬件 RTX 5060 Laptop 8GB。

## TL;DR

- ✅ **Top-K 权衡验证**：K=2 精度最优 17.27%（相对 MoE 基线 v084 **+0.43%**），K=1 最稀疏、K=3 因强均衡反降精度
- ✅ **稀疏化收益确认**（同权重 Hard Top-K vs 强制 Dense，2026-08-15 GPU+CPU 交叉验证）：
  - GPU：K=1 **-23.7%** / K=3 **-10.5%** / Shared **-13.9%**
  - CPU：K=1 **-18.9%** / K=2 **-13.3%** / K=3 **-15.7%** / Shared **-7.0%**
- ✅ **动态 Top-K**：按图像复杂度自适应选择 top_k，实测 avg_top_k=1.66（34% 触发 K=1、66% 触发 K=2）
- ✅ **SharedExpertESMoE**：跨尺度专家共享，参数 **-3.1%**（2.814M→2.727M）、专家对象 **-25%**（16→12）
- ✅ **边界测试**：`test_esmoe.py` 19 passed 1 skipped、`test_shared_expert_esmoe.py` 9/9 通过

## 实验结果

### 1. 四模型消融对比（VisDrone 2019, RTX 5060 Laptop 8GB, 100 epoch）

| 变体 | top_k | Params (M) | mAP50-95 | mAP50 | CPU 延迟 (ms) | 路由熵 H_norm |
|------|-------|-----------|----------|-------|--------------|--------------|
| ES-MoE K=1 | 1 | 2.814 | 16.13% | 28.79% | 56.69 | 0.198 |
| **ES-MoE K=2（基线）** ⭐ | 2 | 2.814 | **17.27%** | **30.55%** | 59.94 | 0.490 |
| ES-MoE K=3 | 3 | 2.814 | 15.20% | 27.44% | 59.44 | 0.741 |
| **SharedExpertESMoE** | 2 | **2.727** | 15.16% | 27.30% | 59.05 | 0.430 |
| MoE 基线 v084（参考） | — | 3.14 | 16.84% | 29.79% | — | — |

**关键观察**：
- K=2 精度最高（17.27%），相对 MoE 基线 v084 **+0.43%**，达项目验收 #3 的"相对增益"标准
- K=3 路由熵最高（0.741，4 专家均匀）但精度反而最低（15.20%）——强均衡稀释了专家专门化
- K=1 路由熵最低（0.198，几乎只用 Expert 1），稀疏度最高，GPU 延迟收益最大（-23.7%）

### 2. 稀疏化收益（同权重 Hard Top-K vs 强制 Dense，2026-08-15）

**GPU（cuda:0）**：

| 变体 | Hard (ms) | Dense (ms) | 收益 |
|------|----------|-----------|------|
| K=1 | 27.341 | 35.853 | **-23.7%** |
| K=3 | 31.118 | 34.780 | **-10.5%** |
| Shared | 29.202 | 33.933 | **-13.9%** |

**CPU（reps=2000）**：

| 变体 | Hard (ms) | Dense (ms) | 收益 |
|------|----------|-----------|------|
| K=1 | 56.658 | 69.900 | **-18.9%** |
| K=2 | 60.687 | 69.970 | **-13.3%** |
| K=3 | 58.254 | 69.126 | **-15.7%** |
| Shared | 65.559 | 70.513 | **-7.0%** |

**洞察**：同一权重复跑（仅开关 sparse/dense），Hard Top-K 在 GPU 与 CPU 上均稳定快于 Dense，
证明稀疏计算（跳过非 Top-K 专家）是真实收益，而非模型差异。

### 3. 动态 Top-K（任务 4 创新）

基于图像边缘密度的复杂度评估器，简单场景触发 K=1、复杂场景触发 K=2/3：

```
[verify_distribution] Summary: 50 images
  avg_complexity=0.0619, avg_top_k=1.66
  distribution: Counter({2: 33, 1: 17})
```

**洞察**：合成简单图像 34% 触发 K=1、66% 触发 K=2，平均只激活 1.66 个专家（< K=2 基线的 2 个）。

### 4. 路由可解释性（K=1/K=2/K=3 熵对比）

| 变体 | Expert0(3×3) | Expert1(5×5) | Expert2(7×7) | Expert3(9×9) | H_norm |
|------|-------------|-------------|-------------|-------------|--------|
| K=1 | 0.250 | **0.750** | 0.000 | 0.000 | 0.198 |
| K=2 | 0.209 | 0.241 | 0.215 | **0.335** | 0.490 |
| K=3 | 0.269 | 0.218 | 0.190 | **0.322** | 0.741 |
| Shared | 0.257 | 0.106 | 0.214 | **0.423** | 0.430 |

**洞察**：top_k 越高路由越均匀（熵越高）；K=1 时 5×5 专家主导（0.750），K=2 时 9×9 大核专家主导（0.335）。

## 改动

### 新增模块与配置

- **`ultralytics/nn/modules/moe/shared_expert_esmoe.py`**（SharedExpertESMoE）
  - 跨尺度专家池共享（参考 Issue #54 方案 D 思路），通过类级 pool 注册表实现
  - P4/P5（128ch, stride 8/16）共享同一组 4 专家；P3/P6 独立（通道不同不可共享）
  - 修复 5 处缺陷：`out_channels` 透传、MIXTURE_MODULES 注册、元数据同步、YAML 语法、测试逻辑

- **`ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone-shared.yaml`**
  - 共享版 ES-MoE 配置（4 个 SharedExpertESMoE 块，pool_id 划分）

### 修复与注册

- `ultralytics/nn/modules/moe/modules.py`：`dynamic_threshold` 默认值 0.4→0.0（修复 K=3 推理二次过滤 bug，mAP 0.98%→15.20%）
- `ultralytics/nn/modules/moe/config.py`：`apply_mixture_config` 修复 `moe_top_k` 正确传递（K=1/K=3 变体此前误用 K=2）
- `ultralytics/nn/mixture_registry.py` + `ultralytics/mixture_metadata.py`：注册 SharedExpertESMoE
- `ultralytics/engine/trainer.py`、`ultralytics/nn/tasks.py`、`ultralytics/nn/modules/__init__.py`、`ultralytics/nn/modules/moe/__init__.py`：支持新模块

### 下游分析脚本（6 个）

| 脚本 | 作用 |
|------|------|
| `scripts/diagnose_esmoe_routing.py` | 路由诊断（4 专家热力图 + Shannon 熵 + 场景推荐）|
| `scripts/compare_esmoe_ablation.py` | Top-K 训练/benchmark/summary 三模式 |
| `scripts/compare_esmoe_edge.py` | 端侧 CPU 延迟 benchmark（6 变体）|
| `scripts/compare_esmoe_inference_modes.py` | 同权重 Hard vs Dense 推理模式对比 |
| `scripts/dynamic_topk_router.py` | 按图像复杂度自适应 top_k |
| `scripts/export_esmoe_int8.py` / `prune_esmoe_for_edge.py` | INT8 导出 / 剪枝（可选探索）|

### 边界测试（2 个）

- `tests/test_esmoe.py`：19 passed 1 skipped（Top-K 边界、稀疏前向、ONNX fallback 等）
- `tests/test_shared_expert_esmoe.py`：9/9 通过（共享参数同步、pool 隔离、YAML 加载）

## 验证

- ✅ 4 模型训练完成（K=1/K=2/K=3/Shared），无 NaN、训练稳定
- ✅ 边界测试 28 项全通过（19+1 skipped + 9）
- ✅ 推理模式 GPU/CPU 交叉验证一致（稀疏化有效）
- ✅ 路由熵与路由热力图可复现（`esmoe_routing_k1/k2/k3/shared/`）

## 复现命令

```bash
conda activate yolo-master
cd YOLO-Master

# 1. 路由诊断（任一模型）
python scripts/diagnose_esmoe_routing.py \
  --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \
  --dry-run --device cpu --plot \
  --output experiments_zviolin/runs/esmoe_routing_k2

# 2. 推理模式对比（GPU，同权重 Hard vs Dense）
python scripts/compare_esmoe_inference_modes.py \
  --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \
  --imgsz 640 --reps 2000 --device cuda:0 \
  --output experiments_zviolin/runs/inference_modes_k2

# 3. 端侧 CPU benchmark
python scripts/compare_esmoe_edge.py \
  --models esmoe_v0_k1 esmoe_v0_k2-2 esmoe_v0_k3 esmoe_v0_shared \
  --imgsz 640 --warmup 100 --reps 500 --device cpu

# 4. 动态 Top-K
python scripts/dynamic_topk_router.py --verify-distribution \
  --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \
  --num-samples 50

# 5. 边界测试
python -m pytest tests/test_esmoe.py tests/test_shared_expert_esmoe.py -v --color=no
```

## 关联资源

- Issue: https://github.com/Tencent/YOLO-Master/issues/54
- 实验数据分支: https://github.com/Zviolin/YOLO-Master/tree/esmoe-experiments-data
- 代码分支: https://github.com/Zviolin/YOLO-Master/tree/feat/esmoe-adaptive-inference
- Discussion 文章: https://github.com/Tencent/YOLO-Master/discussions/（⚠️ 待发布后回填真实编号，参考 https://github.com/Tencent/YOLO-Master/discussions/215）
- 详细文档: `practices/DATA/docs/04-08-任务*.md`

## 局限

- 仅在 VisDrone 上验证，未测 COCO/LVIS 等通用数据集
- SharedExpertESMoE 参数削减有限（-3.1%，ES-MoE 专家为轻量 depthwise conv，共享空间小）——如实记录为创新探索
- CPU 单 batch 延迟波动 ±30%，跨模型横向对比受噪声影响（同权重开关对比更可靠）

## 致谢

感谢 2026 腾讯犀牛鸟开源人才培养计划提供实战机会，感谢 YOLO-Master 项目导师的指导。
```

---

## 三、可选项

| 选项 | 内容 | 是否修改 |
|------|------|----------|
| PR Title 风格 | `feat: ES-MoE Adaptive Inference Optimization (...)` | 推荐 |
| PR Body 详细度 | 含 4 模型消融 + GPU/CPU 推理模式 + 路由熵 + 动态 Top-K | 推荐 |
| PR 范围 | 仅 ES-MoE 相关文件（新增模块/脚本/测试 + 修复）| 保持（避免混入无关改动）|
| 数据分支 | `esmoe-experiments-data`（experiments_zviolin 实验数据）| 推荐 |
| Discussion 链接 | 加入 PR body | 推荐 |

---

## 四、创建 PR

**方式 1（浏览器）**:
```
https://github.com/Zviolin/YOLO-Master/pull/new/feat/esmoe-adaptive-inference
```
粘贴上面的 Title + Body。

**方式 2（gh CLI，如果已登录）**:
```bash
cd "G:\Codes\OpenSource\Rhino-bird\practices\Codes copy 3\YOLO-Master"
git push zviolin feat/esmoe-adaptive-inference
git push zviolin esmoe-experiments-data
gh pr create --base main --head feat/esmoe-adaptive-inference --title "feat: ES-MoE Adaptive Inference Optimization (Top-K Sparse + Dynamic Routing + Shared Experts)" --body "见上方 PR Body"
```
