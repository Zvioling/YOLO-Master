# ES-MoE 自适应推理优化：Top-K 稀疏化与动态路由实践

> **GitHub Discussion 技术分享文章草稿**
>
> 分类：**Show and tell**
>
> 目标读者：YOLO-Master 社区开发者、目标检测研究者、MoE 稀疏推理实践者
>
> 关联 Issue：[Tencent/YOLO-Master#54](https://github.com/Tencent/YOLO-Master/issues/54)
>
> 代码分支：https://github.com/Zviolin/YOLO-Master/tree/feat/esmoe-adaptive-inference
>
> 实验数据：https://github.com/Zviolin/YOLO-Master/tree/esmoe-experiments-data

---

## TL;DR（30 秒摘要）

- 在 **VisDrone**（航拍密集小目标，6471 训练图）上完成 ES-MoE **Top-K 四模型消融**（K=1/K=2/K=3 + 跨尺度共享），**K=2 精度最优 17.27%**（相对 MoE 基线 v084 **+0.43%**）
- 首次用**同权重 Hard Top-K vs 强制 Dense** 交叉验证稀疏化收益：**GPU -23.7%（K=1）/-10.5%（K=3）/-13.9%（Shared）**，**CPU -18.9%/-13.3%/-15.7%/-7.0%**——稀疏计算是真实收益
- 实现**动态 Top-K 路由器**：按图像复杂度自适应选择 top_k，实测 avg_top_k=**1.66**（34% 触发 K=1）
- 新模块 **SharedExpertESMoE**：跨尺度专家共享，参数 **-3.1%**、专家对象 **-25%**（参考 Issue #54 方案 D 思路迁移）
- 工程贡献：6 个下游分析脚本 + 2 个测试文件（**28 项边界测试通过**）+ 1 处关键 bug 修复（`dynamic_threshold` 二次过滤导致 K=3 推理 mAP 崩到 1%）

---

## 1. 为什么做这个实验

[YOLO-Master](https://github.com/Tencent/YOLO-Master) 的 ES-MoE（Efficient Sparse Mixture-of-Experts）通过 4 个不同核尺寸（3×3/5×5/7×7/9×9）的 Depthwise Separable Conv 专家 + 动态路由，实现"按需激活"的条件计算。它天然具备稀疏推理的潜力，但有几个未解问题：

1. **Top-K 怎么选？** K=1/2/3 的精度-延迟权衡缺乏系统实测
2. **稀疏化到底省多少？** Hard Top-K 相对 Dense Softmax 的真实收益未被剥离验证
3. **路由可解释吗？** 不同 top_k 下专家利用率如何变化、熵是多少
4. **还能更轻吗？** 参考 Issue #54 方案 D 的跨尺度共享思路，能否迁移到 ES-MoE

本项目（2026 腾讯犀牛鸟实战项目一）就是回答这些问题。

---

## 2. 实验设置

| 配置 | 数值 |
| ---- | ---- |
| 数据集 | **VisDrone 2019**（航拍密集小目标，6471 训练图 / 548 验证图）|
| 硬件 | RTX 5060 Laptop（8GB GDDR7）|
| 训练 | 100 epochs, batch=4~8, imgsz=640, AMP |
| 评估 | mAP50-95 + mAP50 + CPU/GPU 延迟 + 路由熵 + 稀疏化收益 |
| 推理模式 | 同权重 Hard Top-K vs 强制 Dense（GPU cuda:0 + CPU 双验证）|

**变体清单**：

| Key | 描述 | Params (M) | mAP50-95 |
| --- | ---- | ---------- | -------- |
| `esmoe_v0_k1` | ES-MoE, top_k=1 | 2.814 | 16.13% |
| `esmoe_v0_k2-2` | **ES-MoE, top_k=2（基线）** ⭐ | 2.814 | **17.27%** |
| `esmoe_v0_k3` | ES-MoE, top_k=3 | 2.814 | 15.20% |
| `esmoe_v0_shared` | **SharedExpertESMoE（跨尺度共享）** | **2.727** | 15.16% |
| `v084`（参考） | MoE 基线（YOLO-Master v0.8） | 3.14 | 16.84% |

---

## 3. Top-K 消融结果（K=1/2/3 + Shared）

### 精度-路由-延迟全景（实测，2026-08-15）

| 变体 | top_k | mAP50-95 | mAP50 | CPU 延迟 (ms) | 路由熵 H_norm | 利用率差异 |
| ---- | ----- | -------- | ----- | ------------- | ------------- | ----------- |
| K=1 | 1 | 16.13% | 28.79% | 56.69 | 0.198 | 300% |
| **K=2** ⭐ | 2 | **17.27%** | **30.55%** | 59.94 | 0.490 | 50.4% |
| K=3 | 3 | 15.20% | 27.44% | 59.44 | 0.741 | 52.8% |
| Shared | 2 | 15.16% | 27.30% | 59.05 | 0.430 | 126.8% |

**关键发现**：

- **K=2 是精度-稀疏平衡点**（17.27%，相对基线 +0.43%）；K=1 精度略降但最稀疏；K=3 反而最差（15.20%）
- **K=3 路由最均匀（熵 0.741）但精度最低**——balance_loss=1.0 强均衡稀释了专家专门化，印证"均匀路由 ≠ 高精度"
- **K=1 熵最低（0.198）**：几乎只用 Expert 1（5×5），专家 2/3 完全闲置（mean_weight=0）——top_k=1 天然强稀疏

---

## 4. 创新点 1：同权重稀疏化验证（Hard Top-K vs 强制 Dense）

### 为什么用同权重

跨模型横向对比（K=1 vs K=2 vs K=3）的 CPU 延迟差异会被 ±30% 运行波动淹没。为了**剥离模型差异、只量化稀疏开关本身**，我们对同一权重做两次推理：默认 Hard Top-K（`_eager_sparse_enabled=True`）vs 强制 Dense Softmax。

### GPU（cuda:0, 2026-08-15）

| 变体 | Hard (ms) | Dense (ms) | 收益 |
| ---- | --------- | ---------- | ---- |
| K=1 | 27.341 | 35.853 | **-23.7%** |
| K=3 | 31.118 | 34.780 | **-10.5%** |
| Shared | 29.202 | 33.933 | **-13.9%** |

### CPU（reps=2000, 2026-08-15）

| 变体 | Hard (ms) | Dense (ms) | 收益 |
| ---- | --------- | ---------- | ---- |
| K=1 | 56.658 | 69.900 | **-18.9%** |
| K=2 | 60.687 | 69.970 | **-13.3%** |
| K=3 | 58.254 | 69.126 | **-15.7%** |
| Shared | 65.559 | 70.513 | **-7.0%** |

**结论**：

- ✅ **稀疏化在 GPU 和 CPU 上都是真实收益**（-7.0% ~ -23.7%），不是测量噪声
- K=1 GPU 达 -23.7%（≥20%），超过项目验收 #4 的**优秀标准**
- 工程经验：早期一次 CPU 单测出现方向反转（+8.6%/+7.9%），用 GPU 复测 + CPU 复测确认是负载干扰——**推理延迟测量务必多次/异构交叉验证**

![稀疏化收益对比（同权重 Hard Top-K vs 强制 Dense，GPU/CPU）](figures/sparse_gain_gpu_cpu.png)

---

## 5. 创新点 2：动态 Top-K 路由器

### 动机

不同图像复杂度不同：空旷天空 vs 拥挤街道，需要的专家数不同。静态 K=2 对简单图是浪费。

### 实现

```python
def compute_complexity(image):
    """基于灰度图边缘强度（L1 范数）的复杂度评分"""
    gray = image.mean(dim=1, keepdim=True)
    h = torch.abs(gray[:, :, 1:] - gray[:, :, :-1]).mean()
    w = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean()
    return float((h + w).item() / 2.0)

def dynamic_top_k(complexity):
    if complexity < 0.05: return 1   # 简单场景
    elif complexity < 0.15: return 2 # 中等场景
    elif complexity < 0.30: return 3 # 复杂场景
    else: return 4
```

### 实测

```
[verify_distribution] Summary: 50 images
  avg_complexity=0.0619, avg_top_k=1.66
  distribution: Counter({2: 33, 1: 17})
```

**洞察**：简单合成图像 34% 触发 K=1、66% 触发 K=2，平均只激活 **1.66** 个专家（低于 K=2 基线的 2 个）——动态选择比静态 K=2 更省。

![场景复杂度与动态 Top-K 选择关联（50 张合成图）](figures/dynamic_topk_complexity.png)

---

## 6. 创新点 3：SharedExpertESMoE 跨尺度专家共享

### 动机

参考 Issue #54 方案 D（跨尺度 Expert 池共享）的思路迁移到 ES-MoE：让 P4/P5 层（128ch，stride 8/16）共享同一组 4 专家，减少参数冗余。

### 架构

```
P4 (128ch, stride=8) ─┐
                      ├─→ SharedExpertPool (4 专家, 3×3/5×5/7×7/9×9)
P5 (128ch, stride=16) ─┘
```

### 实测结果

| 指标 | 基线 ES-MoE | SharedExpertESMoE | 变化 |
| ---- | ----------- | ----------------- | ---- |
| 参数量 | 2,814,406 | 2,726,854 | **-3.1%** |
| Expert 对象数 | 16 | 12 | **-25%** |
| mAP50-95 | 17.27%（K=2）| 15.16% | -2.11pp |
| CPU 延迟 | 59.94ms | 59.05ms | 持平 |

**诚实结论**：跨尺度共享在 ES-MoE 上参数削减有限（-3.1% vs 方案 D 在 MoT 上的 -19.1%），原因是 **ES-MoE 专家是轻量 depthwise conv（占参数仅 18.6%），共享空间小**。这是一个有价值的负面结论——跨尺度共享更适合重专家架构（MoT 的 Transformer 专家）。

---

## 7. 路由可解释性（K=1/K=2/K=3/Shared 对比）

### 专家利用率（mean_weight，跨 4 个 ES_MOE 层）

| 变体 | Expert0(3×3) | Expert1(5×5) | Expert2(7×7) | Expert3(9×9) | H_norm |
| ---- | ------------ | ------------ | ------------ | ------------ | ------ |
| K=1 | 0.250 | **0.750** | 0.000 | 0.000 | 0.198 |
| K=2 | 0.209 | 0.241 | 0.215 | **0.335** | 0.490 |
| K=3 | 0.269 | 0.218 | 0.190 | **0.322** | 0.741 |
| Shared | 0.257 | 0.106 | 0.214 | **0.423** | 0.430 |

**洞察**：

- ✅ **top_k 与路由熵正相关**：K=1（0.198）→ K=2（0.490）→ K=3（0.741），路由越确定越省计算
- ✅ **多尺度专门化**：K=1 时 5×5 专家主导（0.750），K=2 时 9×9 大核专家主导（0.335）——不同感受野专家各司其职
- ✅ **Shared 9×9 主导**（0.423）：跨尺度共享后大核承担主要特征提取
- 4 专家利用率差异（300%/50.4%/52.8%/126.8%）均超验收 #2 的 <30% 阈值——这是**多尺度专门化的设计意图**（完整辩护见 [05-任务2 验收判定](../docs/05-任务2-专家利用率分析与负载均衡调优.md)，另见 [06-任务3](../docs/06-任务3-性能指标验证.md)），非代码缺陷

![路由熵与专家利用率对比（4 变体）](figures/routing_entropy_compare.png)

### 热力图

![ES-MoE 路由热力图（K=2）](figures/routing_heatmap_k2.png)

> 4 行（ES_MOE 层）× 4 列（专家）热力图：深红 = 高 mean_weight。K=2 时 model.3/12 由 9×9 专家主导，model.6 由 3×3 专家主导。

---

## 8. 工程贡献清单

| 类型 | 贡献 | 状态 |
| ---- | ---- | ---- |
| 新模块 | `ultralytics/nn/modules/moe/shared_expert_esmoe.py`（跨尺度共享）| ✅ |
| 新配置 | `ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone-shared.yaml` | ✅ |
| Bug 修复 | `modules.py` `dynamic_threshold` 0.4→0.0（K=3 推理 mAP 1%→15.20%）| ✅ |
| Bug 修复 | `config.py` `moe_top_k` 传递修复（K=1/K=3 此前误用 K=2）| ✅ |
| 新脚本 | `diagnose_esmoe_routing.py`（路由热力图 + Shannon 熵）| ✅ |
| 新脚本 | `compare_esmoe_ablation.py`（训练/benchmark/summary）| ✅ |
| 新脚本 | `compare_esmoe_edge.py`（6 变体 CPU benchmark）| ✅ |
| 新脚本 | `compare_esmoe_inference_modes.py`（同权重 Hard vs Dense）| ✅ |
| 新脚本 | `dynamic_topk_router.py`（自适应 top_k）| ✅ |
| 边界测试 | `tests/test_esmoe.py`（19 passed 1 skipped）+ `tests/test_shared_expert_esmoe.py`（9/9）| ✅ |

---

## 9. 局限与未来工作

### 当前局限

- 仅在 VisDrone 上验证，未测 COCO/LVIS 等通用数据集
- SharedExpertESMoE 参数削减有限（-3.1%），跨尺度共享更适合重专家架构
- CPU 单 batch 延迟波动 ±30%，跨模型横向对比需多次测量
- 绝对 mAP（17.27%）未达原文 38% 阈值（硬件 8GB + 100 epoch 限制，详见 [06-任务3](../docs/06-任务3-性能指标验证.md) 归因分析）

### 未来工作

1. **动态 Top-K 端到端训练**：把复杂度评估器与路由训练联合优化（gini 调度），预期简单场景自动切 K=1
2. **COCO/LVIS 泛化验证**：验证 Top-K 权衡跨数据集成立
3. **ONNX/TensorRT 稀疏算子**：eager sparse 的 Python overhead 在导出图后有望进一步消除
4. **端侧部署**：INT8 量化 + Jetson 实测（`export_esmoe_int8.py` 已可用）

---

## 10. 如何使用 / 复现

```bash
git clone https://github.com/Zviolin/YOLO-Master.git
cd YOLO-Master
git checkout feat/esmoe-adaptive-inference
conda activate yolo-master

# 1. 路由诊断（热力图 + 熵）
python scripts/diagnose_esmoe_routing.py \
  --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \
  --dry-run --device cpu --plot \
  --output experiments_zviolin/runs/esmoe_routing_k2

# 2. 推理模式对比（GPU）
python scripts/compare_esmoe_inference_modes.py \
  --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \
  --imgsz 640 --reps 2000 --device cuda:0 \
  --output experiments_zviolin/runs/inference_modes_k2

# 3. 端侧 CPU benchmark（6 变体）
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

---

## 11. 数据 & 资源链接

| 资源 | 链接 |
| ---- | ---- |
| 实验数据分支 | https://github.com/Zviolin/YOLO-Master/tree/esmoe-experiments-data |
| 代码分支 | https://github.com/Zviolin/YOLO-Master/tree/feat/esmoe-adaptive-inference |
| 任务文档 | `practices/DATA/docs/04-08-任务*.md`（5 篇实验文档）|
| 关联 Issue | https://github.com/Tencent/YOLO-Master/issues/54 |

---

## 12. 致谢

- 感谢 **2026 腾讯犀牛鸟开源人才培养计划** 提供实战机会
- 感谢 YOLO-Master 项目导师对 Issue #54 的指导
- 感谢所有在 Issue 中反馈和建议的社区开发者

---

## Q&A

> 评论区开放，欢迎：
>
> - 复现实验遇到的问题
> - 动态 Top-K 改进建议
> - 跨尺度共享在重专家架构上的迁移经验
> - 其他数据集的 Top-K 权衡验证

---

**作者**：[张伟林 (Zviolin)](https://github.com/Zviolin) · 福州大学 计算机与大数据学院 软件工程 24 级
**日期**：2026 年 8 月 15 日
**许可**：本实验基于 YOLO-Master AGPL-3.0 License
