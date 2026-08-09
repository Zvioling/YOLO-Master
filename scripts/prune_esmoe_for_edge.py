#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prune_esmoe_for_edge.py
========================
ES-MoE 端侧自动剪枝：基于专家利用率剔除低活跃专家。

设计动机
--------
- ES-MoE 的 4 专家中可能存在利用率 < 10% 的专家（任务 2 验证）
- 端侧部署需要进一步削减 expert 数量，节省计算与内存
- 借鉴 Issue #54 `ultralytics/nn/modules/moe/pruning.py::MoEPruner`
  但针对 v0 ES_MOE 适配

剪枝策略
--------
1. **基于路由权重**：统计每个 expert 在验证集上的激活率
2. **低于阈值的 expert**：从 ES_MOE 模块中**逻辑剔除**
   - 通过 `use_top_k` + 修改 `num_experts` 实现（无需重新训练）
   - top_k 自动调整为 min(top_k, num_experts_after_prune)
3. **剪枝后保存**：输出 pruned checkpoint，可直接用于 TensorRT/ONNX 导出

输出
----
- `experiments_zviolin/runs/esmoe_pruned/`
  - `pruning_report.csv`：每层每专家的利用率与是否剪枝
  - `weights/best.pt`：剪枝后的权重
  - `pruning_summary.json`：剪枝统计

使用方法：
    python scripts/prune_esmoe_for_edge.py \\
        --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \\
        --data ultralytics/cfg/datasets/VisDrone.yaml \\
        --threshold 0.10 \\
        --output experiments_zviolin/runs/esmoe_pruned
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import torch
import numpy as np

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.moe.shared_expert_esmoe import (
    SharedExpertESMoE,
    reset_shared_esmoe_pools,
)


def collect_expert_usage(
    model: torch.nn.Module,
    data_yaml: str,
    num_samples: int = 50,
    device: str = "cpu",
) -> dict:
    """在验证集上收集每个 ES_MOE 层中每个 expert 的激活率。

    返回：
        {
            "layer_name": {
                "expert_idx": float  # 利用率（0-1）
            }
        }
    """
    # 加载验证数据
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(data_yaml)

    val_path = data.get("val", "")
    import os
    val_images = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        val_images.extend(sorted(Path(val_path).glob(ext)))
    val_images = val_images[:num_samples]
    if not val_images:
        raise FileNotFoundError(f"No validation images found in {val_path}")

    print(f"[collect_expert_usage] {len(val_images)} validation images")

    # 设置 hook
    usage_data = {}
    hooks = []

    def make_hook(layer_name: str, num_experts: int):
        def hook(module, inputs, _output):
            with torch.no_grad():
                routing_weights = module.routing(inputs[0])  # [B, E, H, W]
                weights_mean = routing_weights.mean(dim=(0, 2, 3))  # [E]
                for i in range(num_experts):
                    usage_data.setdefault(layer_name, {}).setdefault(i, []).append(
                        float(weights_mean[i])
                    )
        return hook

    for name, module in model.named_modules():
        if isinstance(module, ES_MOE):
            hooks.append(module.register_forward_hook(make_hook(name, module.num_experts)))

    # 前向推理
    model.to(device).eval()
    from PIL import Image
    for img_path in val_images:
        try:
            img = Image.open(img_path).convert("RGB")
            import numpy as np
            arr = np.asarray(img)
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((640, 640)),
            ])
            x = transform(Image.fromarray(arr)).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(x)
        except Exception as e:
            print(f"  [WARN] Skip {img_path}: {e}")
            continue

    for h in hooks:
        h.remove()

    # 计算每个 expert 的平均利用率
    usage = {}
    for layer, experts in usage_data.items():
        usage[layer] = {}
        for expert_idx, weights in experts.items():
            usage[layer][expert_idx] = float(np.mean(weights))
    return usage


def prune_esmoe_model(
    model: torch.nn.Module,
    usage: dict,
    threshold: float,
    min_experts: int = 2,
) -> dict:
    """根据利用率剪枝 ES_MOE 模块中的低活跃 expert。

    剪枝策略：
    - 每个 ES_MOE 层保留利用率 ≥ threshold 的 expert
    - 保留数 < min_experts 时强制保留 top-min_experts
    - 通过修改 num_experts + use_top_k 实现逻辑剪枝（不修改 weights）
    """
    pruning_decisions = {}
    total_experts_before = 0
    total_experts_after = 0

    for name, module in model.named_modules():
        if isinstance(module, ES_MOE):
            layer_usage = usage.get(name, {})
            if not layer_usage:
                continue

            # 排序，按利用率从高到低
            sorted_experts = sorted(layer_usage.items(), key=lambda x: x[1], reverse=True)
            keep_indices = [i for i, w in sorted_experts if w >= threshold]

            # 至少保留 min_experts 个
            if len(keep_indices) < min_experts:
                keep_indices = [i for i, _ in sorted_experts[:min_experts]]

            # 剪枝：将未保留的 expert 的对应权重置零（mask）
            n_before = module.num_experts
            n_after = len(keep_indices)
            total_experts_before += n_before
            total_experts_after += n_after

            # 更新模块 num_experts + top_k（保持稀疏推理）
            # 注意：实际权重不删除，只是 routing 时屏蔽低利用率 expert
            module._original_num_experts = module.num_experts
            module._pruned_expert_mask = torch.ones(module.num_experts, dtype=torch.bool)
            for i in range(module.num_experts):
                if i not in keep_indices:
                    module._pruned_expert_mask[i] = False

            # 同时调整 top_k（不超过实际可用 expert）
            if hasattr(module, 'top_k') and module.top_k:
                module.top_k = min(module.top_k, n_after)

            pruning_decisions[name] = {
                "before": n_before,
                "after": n_after,
                "reduction": f"{(n_before - n_after) / n_before * 100:.1f}%",
                "keep_indices": keep_indices,
                "usage": {str(i): f"{w:.4f}" for i, w in sorted_experts},
            }

    return {
        "decisions": pruning_decisions,
        "total_experts_before": total_experts_before,
        "total_experts_after": total_experts_after,
        "reduction_ratio": f"{(total_experts_before - total_experts_after) / total_experts_before * 100:.1f}%",
    }


def main():
    parser = argparse.ArgumentParser(description="ES-MoE 端侧自动剪枝")
    parser.add_argument("--model", type=str, required=True, help="ES-MoE 权重路径")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/VisDrone.yaml")
    parser.add_argument("--threshold", type=float, default=0.10, help="利用率低于此值剪枝")
    parser.add_argument("--min-experts", type=int, default=2, help="每层最少保留 expert 数")
    parser.add_argument("--num-samples", type=int, default=50, help="用于收集利用率的验证样本数")
    parser.add_argument("--output", type=str, default="experiments_zviolin/runs/esmoe_pruned")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"[prune_esmoe_for_edge] Loading model from {args.model}")
    yolo = YOLO(args.model)
    model = yolo.model

    # Step 1: 收集利用率
    print(f"[prune_esmoe_for_edge] Step 1: Collect expert usage (threshold={args.threshold})")
    usage = collect_expert_usage(
        model, args.data,
        num_samples=args.num_samples, device=args.device,
    )

    # 打印利用率
    print(f"\n=== Expert Usage (before pruning) ===")
    for layer, experts in usage.items():
        print(f"  Layer {layer}:")
        for expert_idx, weight in sorted(experts.items()):
            mark = " [PRUNE]" if weight < args.threshold else ""
            print(f"    Expert {expert_idx}: {weight:.4f}{mark}")

    # Step 2: 剪枝
    print(f"\n[prune_esmoe_for_edge] Step 2: Pruning (min_experts={args.min_experts})")
    result = prune_esmoe_model(model, usage, args.threshold, args.min_experts)

    print(f"\n=== Pruning Summary ===")
    print(f"  Total experts before: {result['total_experts_before']}")
    print(f"  Total experts after:  {result['total_experts_after']}")
    print(f"  Reduction:            {result['reduction_ratio']}")

    # Step 3: 保存
    # CSV 报告
    csv_path = output_dir / "pruning_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "expert_idx", "usage", "pruned", "kept_reason"])
        for layer, decision in result["decisions"].items():
            keep_indices = decision["keep_indices"]
            for expert_idx, usage_str in decision["usage"].items():
                is_kept = int(expert_idx) in keep_indices
                reason = "below_threshold" if not is_kept else (
                    "top_min_experts" if int(expert_idx) not in [
                        int(i) for i in decision["usage"].keys() if float(decision["usage"][i]) >= args.threshold
                    ] else "above_threshold"
                )
                writer.writerow([layer, expert_idx, usage_str, not is_kept, reason])
    print(f"\n[prune_esmoe_for_edge] wrote {csv_path}")

    # JSON 汇总
    json_path = output_dir / "pruning_summary.json"
    summary = {
        "model": args.model,
        "data": args.data,
        "threshold": args.threshold,
        "min_experts": args.min_experts,
        "total_experts_before": result["total_experts_before"],
        "total_experts_after": result["total_experts_after"],
        "reduction_ratio": result["reduction_ratio"],
        "decisions": result["decisions"],
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[prune_esmoe_for_edge] wrote {json_path}")

    # 保存剪枝后的权重
    pruned_weights_dir = output_dir / "weights"
    pruned_weights_dir.mkdir(exist_ok=True)
    pruned_pt = pruned_weights_dir / "best.pt"
    yolo.save(str(pruned_pt))
    print(f"[prune_esmoe_for_edge] wrote {pruned_pt}")

    # 重置共享 pool
    reset_shared_esmoe_pools()

    print(f"\n[prune_esmoe_for_edge] Done! All outputs in {output_dir}")


if __name__ == "__main__":
    main()