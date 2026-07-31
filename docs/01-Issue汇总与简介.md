# ncnn与YOLO-Master 犀牛鸟专属Issue原文汇总

> 直接拷贝自GitHub Issue的完整原文，未做任何简述与改写。原文为Markdown格式，含正文、相关链接、难度标签与认领规则。
>
> 数据来源：
> - YOLO-Master：https://github.com/Tencent/YOLO-Master/issues?q=is%3Aissue+%E8%85%BE%E8%AE%AF%E7%8A%80%E7%89%9B%E9%B8%9F%E5%BC%80%E6%BA%90%E4%B8%93%E5%B1%9E+is%3Aopen
> - ncnn：https://github.com/Tencent/ncnn/issues?q=is%3Aissue+%E8%85%BE%E8%AE%AF%E7%8A%80%E7%89%9B%E9%B8%9F%E5%BC%80%E6%BA%90%E4%B8%93%E5%B1%9E+is%3Aopen
> - 抓取时间：2026-07-05

---

## 通用认领规则（适用于所有 issue）

> 以下为每个犀牛鸟专属 issue 末尾附带的认领规则（原文直接拷贝）：

```
**本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q
```

---

## 一、YOLO-Master 犀牛鸟专属 Issue（6 个，标签：`腾讯犀牛鸟开源专属`）

> YOLO-Master 仓库中标记为"腾讯犀牛鸟开源专属"的 issue 共有 6 个，按 issue 编号排列如下。

### Issue #49：【低难度】模型训练专项 - 垂类数据集基线训练

- **GitHub链接**：https://github.com/Tencent/YOLO-Master/issues/49
- **状态**：已关闭（completed）
- **难度标签**：`犀牛鸟-低难度`

**Issue 原文（直接拷贝）**：

> 在 VisDrone（航拍密集小目标）和 SKU-110K（零售密集商品）等垂类数据集上(可选，换用其他的垂类公开数据集)，分别使用 YOLO-Master-v0.1-N 和 YOLO-Master-EsMoE-N 进行训练复现
> - 使用项目内置数据集配置 ultralytics/cfg/datasets/VisDrone.yaml 和 ultralytics/cfg/datasets/SKU-110K.yaml
> - 训练配置参考论文推荐：imgsz=640，epochs 100~300（根据 GPU 资源调整）
> - 记录完整训练日志，包含每 epoch 的 mAP50、mAP50-95、box_loss、cls_loss、moe_loss(使用wandb进行配置，可以将url设置为公开贴出来)
> - 在scripts/reproduce/ 下提交可复现脚本 reproduce_visdrone.py 和 reproduce_sku110k.py
> 提供 README.md 说明数据集下载命令、训练命令、预期结果、已知问题与解决方案
> 提交 Pull Request，包含训练脚本、运行日志、结果对比表
>
> https://github.com/Tencent/YOLO-Master
> https://github.com/Tencent/YOLO-Master/releases/tag/YOLO-Master-v26.02
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/datasets/VisDrone.yaml
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml
> https://github.com/Tencent/YOLO-Master/blob/main/scripts/compare_moe_coco128.py
> 难度：低
>
> ---
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

### Issue #50：【低难度】训练优化专项：垂类场景 LoRA 高效微调适配

- **GitHub链接**：https://github.com/Tencent/YOLO-Master/issues/50
- **状态**：已关闭（completed）
- **难度标签**：`犀牛鸟-低难度`

**Issue 原文（直接拷贝）**：

> - 选择两个差异显著的垂类场景（建议 VisDrone 密集航拍 + brain-tumor 稀疏医疗）
> -为 YOLO-Master-EsMoE-N 在 examples/lora_examples/ 下新增 yolo_master_visdrone_lora.yaml 和 yolo_master_brain_tumor_lora.yaml
> - 配置需覆盖：lora_r、lora_alpha、lora_use_rslora、lora_target_modules、lora_include_attention、lora_gradient_checkpointing
> - 针对 MoE 模型明确路由层（routing）是否纳入 LoRA 目标模块，并在配置文件中注释说明理由
> - 在每个场景上对比至少 3 组 rank（r=4, 8, 16）的微调效果，记录 mAP50-95、可训练参数量、训练时间、峰值显存
> - 训练限制在 20~50 epoch 内完成，模拟少样本快速迭代场景
> 提供 README.md 说明各场景最佳 rank 推荐、目标模块选择建议、常见陷阱（如医疗灰度通道处理、航拍尺度变化）
> 提交 Pull Request，包含 LoRA 配置文件、训练脚本、对比表格、适配指南
>
> https://github.com/Tencent/YOLO-Master
> https://github.com/Tencent/YOLO-Master/tree/main/examples/lora_examples
> https://github.com/Tencent/YOLO-Master/blob/main/examples/lora_examples/README.md
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/datasets/VisDrone.yaml
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml
> 难度：低
>
> ---
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

### Issue #51：【中高难度】推理加速专项：垂类模型边缘端推理加速与一致性验证

- **GitHub链接**：https://github.com/Tencent/YOLO-Master/issues/51
- **状态**：开放（open）
- **难度标签**：`犀牛鸟-中高难度`

**Issue 原文（直接拷贝）**：

> - 在 VisDrone 或 SKU-110K 数据集上训练或微调 YOLO-Master-EsMoE-N 模型（可直接复用 Issue #1 或 #2 的 checkpoint）
> - 将训练好的模型导出到至少两种格式：ONNX + NCNN（或 ONNX + MNN）
> - 对 ONNX 使用 onnxsim 简化并验证 opset 兼容性；对 NCNN 使用 pnnx 转换并验证参数文件；可选尝试 INT8 量化，校准数据集不少于 300 张
> - 实现边缘端推理代码（Python 或 C++），支持垂类图像预处理（如 SKU-110K 高分辨率 Resize + Letterbox、VisDrone 特定长宽比保留）
> - 后处理需针对垂类调优 NMS 参数（如 VisDrone 小目标可能需要更低 conf 阈值）
> - 使用 CMake 构建，确保至少在两个平台（如 Linux x86_64 + Linux ARM64 / NVIDIA Jetson / Windows）上编译并运行成功
> - 在相同输入下（至少 500 张验证集图像），对比 PyTorch 原版与导出模型的 mAP50-95，目标误差 < 0.5%（非量化）或 < 1.0%（INT8 量化）
> - 若误差超出，提供逐层中间输出对比或单张可视化差异分析
> - 报告边缘端推理延迟（ms/frame）和吞吐量（FPS），对比不同导出格式（ONNX vs NCNN vs MNN）的 benchmark 数据
> 完成后，在 GitHub Discussion 发表技术总结文章并提供部署仓库链接
> 可选：向 examples/ 目录发 Pull Request，补充垂类边缘端推理示例
>
> https://github.com/Tencent/YOLO-Master
> https://github.com/Tencent/ncnn
> https://github.com/pnnx/pnnx
> https://github.com/Tencent/YOLO-Master/tree/main/examples/YOLOv8-MNN-CPP
> https://github.com/Tencent/YOLO-Master/tree/main/examples/YOLOv8-ONNXRuntime
> https://github.com/Tencent/YOLO-Master/tree/main/examples/YOLOv8-SAHI-Inference-Video
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/engine/exporter.py
> 难度：中
>
> ---
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

### Issue #52：【中高难度】MoE 优化专项：MoE 专家剪枝与动态超参数调度优化

- **GitHub链接**：https://github.com/Tencent/YOLO-Master/issues/52
- **状态**：已关闭（completed）
- **难度标签**：`犀牛鸟-中高难度`

**Issue 原文（直接拷贝）**：

> 在 COCO 或 VisDrone 数据集上训练 YOLO-Master-EsMoE-N 模型
> 使用 MoEPruner 对训练好的模型进行专家剪枝，对比至少 5 组阈值（threshold ∈ {0.05, 0.10, 0.15, 0.20, 0.30}）
> 对比两种恢复策略：剪枝后直接推理 vs 剪枝后 + LoRA 微调 10-epoch 恢复
> 测量并记录每个实验点的：mAP50-95、mAP50、FLOPs（通过 get_gflops()）、Latency（ms）、Params（M）、每层保留专家数、专家利用率 Gini 系数
> 绘制 "阈值 → mAP / FLOPs / Latency" 三维曲线和 Pareto 前沿图（精度 vs 延迟），标注推荐的 "Sweet Spot"
> 设计并实现至少一种动态超参数调度策略（如根据 expert_usage 的 Gini 系数自动调整 balance_loss_coeff，或根据验证 mAP 饱和情况动态退火 top_k），需给出调度公式并论证合理性
> 与固定参数基线进行对照实验（至少 3 组：基线 + 动态调度 + 消融组），计算收敛加速比（实验组达到基线最终精度 95% 所需 epoch 比例）
> 分析动态调度的副作用（如训练不稳定、最终 mAP 下降），提出改进建议
> 提出场景化推荐：如服务器端推荐阈值 0.10、边缘端推荐阈值 0.20，并附数据支撑
> 完成后，在 GitHub Discussion 发表技术总结文章并提供实验脚本仓库链接
> 动态调度策略代码修改需保持向后兼容，可提交 Pull Request
>
> https://github.com/Tencent/YOLO-Master
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moe/pruning.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moe/diagnostics.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moe/analysis.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moe/loss.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moe/modules.py
> https://github.com/Tencent/YOLO-Master/blob/main/scripts/compare_moe_coco128.py
> https://github.com/Tencent/YOLO-Master/blob/main/scripts/bench_moe_micro.py
> 难度：中
>
> ---
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

### Issue #53：【低难度】MoA(mixture of Attention)：边界测试补全与垂类训练验证

- **GitHub链接**：https://github.com/Tencent/YOLO-Master/issues/53
- **状态**：已关闭（completed）
- **难度标签**：`犀牛鸟-低难度`

**Issue 原文（直接拷贝）**：

> 在现有 tests/test_moa.py 基础上，系统补全 MoA 模块的边界测试与训练验证，提升测试覆盖率
> 测试补全（至少覆盖以下场景）：
> NeckMoAFusion 在跨尺度输入尺寸不匹配（如 hi 为 15×15、lo 为 7×7，非严格 2× 下采样）时的前向稳定性与形状保持
> MoABlock 的 temperature 退火到极小值（如 temperature < 1e-4）时，softmax 路由概率的数值稳定性（是否出现 NaN 或均匀分布）
> _LocalAttnHead 与 _GlobalAttnHead 在 num_heads 不能被 dim 整除时的降级处理（_safe_groups 的边界）
> C2fMoA 的 aux_loss 在多 MoABlock 嵌套时是否存在重复计数（类似 MoE 的 MOE_LOSS_REGISTRY 双计数问题）
> 缺陷修复：在补充测试过程中，若发现任何边界缺陷（如 IndexError、NaN 传播、形状不匹配），需一并定位并修复
> 覆盖率报告：提供 pytest --cov 前后的覆盖率对比，至少覆盖 ultralytics/nn/modules/moa/ 目录
> 垂类训练验证：在 VisDrone 或 SKU-110K 上，使用 YOLO-Master-v0.10-MoA-N 训练 50~100 epoch，验证 MoA 模块在真实数据集上的收敛性，记录 mAP50-95 与 loss 曲线，与同配置的 MoE 基线对比
> 提交 Pull Request，包含测试代码、修复代码、覆盖率报告、训练日志
>
> https://github.com/Tencent/YOLO-Master
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moa/moa.py
> https://github.com/Tencent/YOLO-Master/blob/main/tests/test_moa.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml
> https://github.com/Tencent/YOLO-Master/blob/main/scripts/compare_moa_ablation.py
> 难度：低
>
> ---
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

### Issue #54：【中高难度】MoT(mixture of Transformer)：消融对比、路由可解释性与混合架构探索

- **GitHub链接**：https://github.com/Tencent/YOLO-Master/issues/54
- **状态**：开放（open）
- **难度标签**：`犀牛鸟-中高难度`

**Issue 原文（直接拷贝）**：

> 在 COCO 或 VisDrone 数据集上，使用 scripts/compare_mot_ablation.py 作为参考脚本，训练至少 3 种模型变体：YOLO-Master-EsMoE-N（MoE 基线）、YOLO-Master-v0.10-MoT-N（MoT 实验模块）、YOLO-Master-v0.10-MoA-N（MoA 对比组）
> 对比测量每种变体的：mAP50-95、mAP50、Latency（ms，P50/P95/P99）、FLOPs（实际）、Params（M）、训练稳定性（loss 曲线是否发散、是否出现 NaN）
> 路由行为可解释性分析：
> 对 MoT：使用 diagnose_model 或自定义 hook 分析 MoTBlock 中各 Transformer expert（LocalConvTransformer / WindowTransformer / DeformableTransformer）的 token 路由分布，绘制专家激活热力图
> 对比不同场景（密集 vs 稀疏、小目标 vs 大目标）下的专家激活模式，验证 DeformableTransformer 是否在遮挡/不规则目标场景激活率显著上升
> 混合架构探索：尝试将 MoT 与 MoE 进行层级组合（如 backbone 用 MoE、neck 用 MoT），或与 MoA 进行交叉组合，评估是否产生协同增益（mAP 提升 > 1% 或延迟降低 > 10% 视为有意义）
> 边界测试与稳定性修复：
> 补全 tests/test_mot.py 的边界测试（至少覆盖：MoTBlock 在 window_size 大于 feature map 时的降级处理、_WindowTransformerExpert 的 shift 操作在奇数尺寸输入时的边界、MoT 的 exploration_eps 在 eval 模式下是否被正确禁用）
> 若发现边界缺陷（如 IndexError、NaN、shape mismatch），需定位并修复
> 场景化洞察产出：基于对比数据，提出至少 3 条场景化推荐（如「密集小目标场景 MoT 的 WindowTransformer 激活率最高」；「复杂遮挡场景 DeformableTransformer 专家被优先路由」；「MoE + MoT 组合在 backbone 层带来 +X% mAP 但延迟增加 Y%」），每条需附数据支撑
> 完成后，在 GitHub Discussion 发表技术总结文章并提供实验脚本仓库链接
> 边界测试修复代码可提交 Pull Request；混合架构实验若产生稳定增益，可提交 Pull Request 补充新 YAML 配置
>
> https://github.com/Tencent/YOLO-Master
> https://github.com/Tencent/YOLO-Master/blob/main/scripts/compare_mot_ablation.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/mot/mot.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moa/moa.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moe/diagnostics.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moe/pruning.py
> https://github.com/Tencent/YOLO-Master/blob/main/tests/test_mot.py
> https://github.com/Tencent/YOLO-Master/blob/main/tests/test_moa.py
> https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml
> 难度：中
>
> ---
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

## 二、ncnn 犀牛鸟专属 Issue（5 个，标签：`腾讯犀牛鸟开源专属`）

> ncnn 仓库中标记为"腾讯犀牛鸟开源专属"的 issue 共有 5 个，按 issue 编号排列如下。

### Issue #6787：Tencent-Hunyuan/HunyuanOCR ncnn 移植与多平台部署

- **GitHub链接**：https://github.com/Tencent/ncnn/issues/6787
- **状态**：开放（open）
- **难度标签**：`犀牛鸟-中高难度`（issue 内标注：难度：中）

**Issue 原文（直接拷贝）**：

> - 使用 pnnx 将 tencent/HunyuanOCR 模型转换到ncnn
> - 参考或 fork ncnn_llm 项目，C++ 实现 LLM 解码等，尽量减少第三方依赖
> - 在相同输入下，ncnn 输出的最终文本须与 PyTorch 原版一致
> - 使用 CMake 构建，确保至少在两个平台（如 Linux Windows）完美编译与运行
> - 完成后，GitHub Discussion 发表技术总结文章并提供 HunyuanOCR-ncnn repo url
> - 可选：向 ncnn_llm 项目发 Pull Request
>
> 难度：中
>
> https://github.com/Tencent-Hunyuan/HunyuanOCR
> https://github.com/futz12/ncnn_llm
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

### Issue #6788：tencent-ailab/Penguin-VL ncnn 移植与多平台部署

- **GitHub链接**：https://github.com/Tencent/ncnn/issues/6788
- **状态**：开放（open）
- **难度标签**：`犀牛鸟-中高难度`（issue 内标注：难度：中）

**Issue 原文（直接拷贝）**：

> - 使用 pnnx 将 tencent/Penguin-VL 模型转换到ncnn
> - 参考或 fork ncnn_llm 项目，C++ 实现 LLM 解码等，尽量减少第三方依赖
> - 在相同输入下，ncnn 输出的最终文本须与 PyTorch 原版一致
> - 使用 CMake 构建，确保至少在两个平台（如 Linux Windows）完美编译与运行
> - 完成后，GitHub Discussion 发表技术总结文章并提供 Penguin-VL-ncnn repo url
> - 可选：向 ncnn_llm 项目发 Pull Request
>
> 难度：中
>
> https://github.com/tencent-ailab/Penguin-VL
> https://github.com/futz12/ncnn_llm
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

### Issue #6789：TencentCloudADP/youtu-vl ncnn 移植与多平台部署

- **GitHub链接**：https://github.com/Tencent/ncnn/issues/6789
- **状态**：开放（open）
- **难度标签**：`犀牛鸟-中高难度`（issue 内标注：难度：中）

**Issue 原文（直接拷贝）**：

> - 使用 pnnx 将 tencent/Youtu-VL 模型转换到ncnn
> - 参考或 fork ncnn_llm 项目，C++ 实现 LLM 解码等，尽量减少第三方依赖
> - 在相同输入下，ncnn 输出的最终文本须与 PyTorch 原版一致
> - 使用 CMake 构建，确保至少在两个平台（如 Linux Windows）完美编译与运行
> - 完成后，GitHub Discussion 发表技术总结文章并提供 Youtu-VL-ncnn repo url
> - 可选：向 ncnn_llm 项目发 Pull Request
>
> 难度：中
>
> https://github.com/TencentCloudADP/youtu-vl
> https://github.com/futz12/ncnn_llm
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

### Issue #6790：QwenLM/Qwen3-ASR ncnn 移植与多平台部署

- **GitHub链接**：https://github.com/Tencent/ncnn/issues/6790
- **状态**：开放（open）
- **难度标签**：`犀牛鸟-中高难度`（issue 内标注：难度：高）

**Issue 原文（直接拷贝）**：

> - 使用 pnnx 将 Qwen/Qwen3-ASR 模型转换到ncnn
> - 参考或 fork ncnn_llm 项目，C++ 实现 LLM 解码等，尽量减少第三方依赖
> - 在相同输入下，ncnn 输出的最终文本须与 PyTorch 原版一致
> - 使用 CMake 构建，确保至少在两个平台（如 Linux Windows）完美编译与运行
> - 完成后，GitHub Discussion 发表技术总结文章并提供 Qwen3-ASR-ncnn repo url
> - 可选：向 ncnn_llm 项目发 Pull Request
>
> 难度：高
>
> https://github.com/QwenLM/Qwen3-ASR
> https://github.com/futz12/ncnn_llm
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

### Issue #6791：QwenLM/Qwen3-TTS ncnn 移植与多平台部署

- **GitHub链接**：https://github.com/Tencent/ncnn/issues/6791
- **状态**：开放（open）
- **难度标签**：`犀牛鸟-中高难度`（issue 内标注：难度：高）

**Issue 原文（直接拷贝）**：

> - 使用 pnnx 将 Qwen/Qwen3-TTS 模型转换到ncnn
> - 参考或 fork ncnn_llm 项目，C++ 实现 LLM 解码等，尽量减少第三方依赖
> - 在相同输入下，ncnn 输出的最终音频须与 PyTorch 原版一致
> - 使用 CMake 构建，确保至少在两个平台（如 Linux Windows）完美编译与运行
> - 完成后，GitHub Discussion 发表技术总结文章并提供 Qwen/Qwen3-TTS-ncnn repo url
> - 可选：向 ncnn_llm 项目发 Pull Request
>
> 难度：高
>
> https://github.com/QwenLM/Qwen3-TTS
> https://github.com/futz12/ncnn_llm
>
> **本issue为2026犀牛鸟开源人才培养活动专属issue，仅供已报名参与犀牛鸟活动的同学认领**
> 【认领时间】7月1日～7月31日（7月1日前认领视为无效❗️）
> 【认领方式】在本issue评论区回复"已认领本任务"，即视为认领成功
> 【活动报名】需提前完成犀牛鸟报名问卷，问卷将用于活动登记和奖励发放：https://wj.qq.com/s2/26888567/gh2q

---

## 三、汇总对照表

| 仓库 | Issue # | 标题 | 难度标签 | 内部标注 | 状态 |
|------|---------|------|---------|---------|------|
| YOLO-Master | [#49](https://github.com/Tencent/YOLO-Master/issues/49) | 模型训练专项-垂类数据集基线训练 | 低 | 低 | 已关闭 |
| YOLO-Master | [#50](https://github.com/Tencent/YOLO-Master/issues/50) | 训练优化专项：垂类场景 LoRA 高效微调适配 | 低 | 低 | 已关闭 |
| YOLO-Master | [#51](https://github.com/Tencent/YOLO-Master/issues/51) | 推理加速专项：垂类模型边缘端推理加速与一致性验证 | 中高 | 中 | 开放 |
| YOLO-Master | [#52](https://github.com/Tencent/YOLO-Master/issues/52) | MoE 优化专项：MoE 专家剪枝与动态超参数调度优化 | 中高 | 中 | 已关闭 |
| YOLO-Master | [#53](https://github.com/Tencent/YOLO-Master/issues/53) | MoA(mixture of Attention)：边界测试补全与垂类训练验证 | 低 | 低 | 已关闭 |
| YOLO-Master | [#54](https://github.com/Tencent/YOLO-Master/issues/54) | MoT(mixture of Transformer)：消融对比、路由可解释性与混合架构探索 | 中高 | 中 | 开放 |
| ncnn | [#6787](https://github.com/Tencent/ncnn/issues/6787) | Tencent-Hunyuan/HunyuanOCR ncnn 移植与多平台部署 | 中高 | 中 | 开放 |
| ncnn | [#6788](https://github.com/Tencent/ncnn/issues/6788) | tencent-ailab/Penguin-VL ncnn 移植与多平台部署 | 中高 | 中 | 开放 |
| ncnn | [#6789](https://github.com/Tencent/ncnn/issues/6789) | TencentCloudADP/youtu-vl ncnn 移植与多平台部署 | 中高 | 中 | 开放 |
| ncnn | [#6790](https://github.com/Tencent/ncnn/issues/6790) | QwenLM/Qwen3-ASR ncnn 移植与多平台部署 | 中高 | 高 | 开放 |
| ncnn | [#6791](https://github.com/Tencent/ncnn/issues/6791) | QwenLM/Qwen3-TTS ncnn 移植与多平台部署 | 中高 | 高 | 开放 |

**注**：本表仅汇总，issue 正文的完整原文已逐条按编号顺序列于上文。下一份文档（`02-Issue54排除法选择.md`）将基于本文件的原文做选型决策。
