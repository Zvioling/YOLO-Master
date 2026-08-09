#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_esmoe_routing.py
=========================
ES-MoE (Efficient Sparse Mixture-of-Experts) 路由诊断脚本。

针对 YOLO-Master 的 ES_MOE 模块，分析每个 ES_MOE 层中各专家的激活率与权重分布，
帮助理解多尺度专家（3x3 / 5x5 / 7x7 / 9x9）的实际分工。

与 diagnose_mot_routing.py 的差异：
- MoT 用 3 个 Transformer 变体（LocalConvTransformer / WindowTransformer / DeformableTransformer）
- ES-MoE 用 4 个不同核尺寸的 Depthwise Separable Conv（3x3 / 5x5 / 7x7 / 9x9）

输出文件：
- routing_summary.csv ：4 层 × 4 专家 = 16 条路由数据
- recommendations.json：4 条场景化推荐 + summary
- routing_heatmap.png ：4 行 × 4 列激活热力图

使用方法：
    python scripts/diagnose_esmoe_routing.py \\
        --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \\
        --dry-run --device cpu --plot \\
        --output experiments_zviolin/runs/esmoe_routing
"""
from __future__ import annotations

# 必须在所有 ultralytics 导入前确保 sys.path 正确（PowerShell 下 cwd 不会自动加入 sys.path）
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json

import numpy as np
import torch
from ultralytics import YOLO

from ultralytics.nn.modules.moe.modules import ES_MOE


def collect_routing(model: torch.nn.Module) -> tuple[list[dict], list]:
    """为每个 ES_MOE 模块注册 forward hook，收集路由权重。"""
    rows: list[dict] = []
    handles: list = []

    def make_hook(name: str, num_experts: int):
        def hook(module: ES_MOE, inputs, _output):
            with torch.no_grad():
                routing_weights = module.routing(inputs[0])  # [B, E, H, W]
                weights_mean = routing_weights.mean(dim=(0, 2, 3))  # [E]
                # top-k indices (默认 2)
                topk_vals, topk_indices = torch.topk(weights_mean, module.top_k)
                topk_set = set(topk_indices.cpu().tolist())
                for i in range(num_experts):
                    kernel_size = module.experts[i].conv.depthwise.kernel_size[0]
                    rows.append({
                        "layer": name,
                        "expert": i,
                        "kernel_size": int(kernel_size),
                        "mean_weight": float(weights_mean[i]),
                        "is_top_k": int(i in topk_set),
                    })
        return hook

    for name, module in model.named_modules():
        if isinstance(module, ES_MOE):
            handles.append(module.register_forward_hook(make_hook(name, module.num_experts)))
    return rows, handles


def compute_normalized_entropy(weights: torch.Tensor) -> float:
    """计算归一化 Shannon 熵，值越低说明路由越确定。"""
    p = weights / weights.sum().clamp_min(1e-8)
    p = p[p > 0]
    if p.numel() <= 1:
        return 0.0
    H = -(p * torch.log(p)).sum()
    H_norm = float(H / np.log(len(weights)))
    return H_norm


def plot_heatmap(rows: list[dict], output_path: Path) -> None:
    """绘制 4 行 × 4 列激活热力图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diagnose_esmoe_routing] matplotlib not installed, skipping heatmap")
        return

    # 提取唯一层和专家
    def layer_sort_key(name: str) -> int:
        try:
            return int(name.split(".")[-1])
        except (ValueError, IndexError):
            return 0

    layers = sorted({r["layer"] for r in rows}, key=layer_sort_key)
    expert_ids = sorted({r["expert"] for r in rows})

    matrix = np.zeros((len(layers), len(expert_ids)))
    kernel_sizes = np.zeros((len(layers), len(expert_ids)), dtype=int)

    layer_idx = {layer: i for i, layer in enumerate(layers)}
    expert_idx = {expert: j for j, expert in enumerate(expert_ids)}

    for r in rows:
        i = layer_idx[r["layer"]]
        j = expert_idx[r["expert"]]
        matrix[i, j] = r["mean_weight"]
        kernel_sizes[i, j] = r["kernel_size"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(expert_ids)))
    ax.set_xticklabels(
        [f"Expert {e}\n(ks={kernel_sizes[0, expert_idx[e]]})" for e in expert_ids]
    )
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_title("ES-MoE Routing Heatmap (mean_weight across spatial dims)")
    plt.colorbar(im, ax=ax, label="mean_weight")

    for i in range(len(layers)):
        for j in range(len(expert_ids)):
            color = "white" if matrix[i, j] > 0.3 else "black"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color=color, fontsize=10)

    plt.xlabel("Expert (kernel size)")
    plt.ylabel("ES_MOE Layer")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_recommendations(rows: list[dict]) -> dict:
    """基于路由数据生成场景化推荐 + summary。"""
    # layer 名格式：model.3 / model.6 / model.9 / model.12
    def layer_sort_key(name: str) -> int:
        try:
            return int(name.split(".")[-1])
        except (ValueError, IndexError):
            return 0

    layers = sorted({r["layer"] for r in rows}, key=layer_sort_key)

    recommendations = []
    expert_weight_sums = {}

    for layer in layers:
        entries = [r for r in rows if r["layer"] == layer]
        if not entries:
            continue
        top_entry = max(entries, key=lambda r: r["mean_weight"])
        layer_num = layer.split(".")[-1]  # "model.3" → "3"
        # 通道映射：3/6 → 256/512, 9 → 512, 12 → 1024
        try:
            ln = int(layer_num)
            if ln <= 3:
                channel = 256
            elif ln <= 9:
                channel = 512
            else:
                channel = 1024
        except ValueError:
            channel = "?"
        recommendations.append(
            f"Layer {layer} (P{layer_num}, channel={channel}): "
            f"Top expert is Expert {top_entry['expert']} (kernel_size={top_entry['kernel_size']}, "
            f"mean_weight={top_entry['mean_weight']:.3f})"
        )

        for e in entries:
            expert_weight_sums.setdefault(e["expert"], []).append(e["mean_weight"])

    # summary 统计
    summary = {}
    for expert_id, weights in expert_weight_sums.items():
        w_tensor = torch.tensor(weights)
        summary[f"expert_{expert_id}"] = {
            "kernel_size": int([r["kernel_size"] for r in rows if r["expert"] == expert_id][0]),
            "mean": float(w_tensor.mean()),
            "std": float(w_tensor.std()),
            "is_top1_active_rate": float(np.mean([1 if w > 0.25 else 0 for w in weights])),
            "routing_entropy": compute_normalized_entropy(w_tensor),
        }

    avg_entropy = float(np.mean([v["routing_entropy"] for v in summary.values()]))
    summary["__avg_routing_entropy__"] = avg_entropy

    return {"recommendations": recommendations, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="ES-MoE 路由诊断")
    parser.add_argument("--model", type=str, required=True, help="权重 (.pt) 或 YAML 路径")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda / cuda:0 / mps")
    parser.add_argument("--dry-run", action="store_true", help="合成输入模式（无需数据集）")
    parser.add_argument("--image-dir", type=str, default=None, help="真实图像目录（与 --dry-run 二选一）")
    parser.add_argument("--num-images", type=int, default=20, help="真实图像采样数")
    parser.add_argument("--plot", action="store_true", help="生成热力图 PNG")
    parser.add_argument("--output", type=str, default="experiments_zviolin/runs/esmoe_routing",
                        help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diagnose_esmoe_routing] Loading model from {args.model}")
    model = YOLO(args.model).model
    model.eval()

    # 决定输入来源
    inputs = []
    if args.dry_run:
        print(f"[diagnose_esmoe_routing] Using synthetic input (1 random tensor)")
        inputs = [torch.randn(1, 3, 640, 640, device=args.device)]
    elif args.image_dir:
        from PIL import Image
        import numpy as np
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {image_dir}")
        # 加载 jpg/png
        candidates = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            candidates.extend(sorted(image_dir.glob(ext)))
        candidates = candidates[:args.num_images]
        if not candidates:
            raise FileNotFoundError(f"No images found in {image_dir}")
        print(f"[diagnose_esmoe_routing] Using {len(candidates)} real images from {image_dir}")
        for img_path in candidates:
            img = Image.open(img_path).convert("RGB").resize((640, 640))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            tensor = torch.from_numpy(arr).unsqueeze(0).to(args.device)
            inputs.append(tensor)
    else:
        print(f"[diagnose_esmoe_routing] No input specified, using synthetic")
        inputs = [torch.randn(1, 3, 640, 640, device=args.device)]

    print(f"[diagnose_esmoe_routing] Running forward pass on {args.device} ({len(inputs)} inputs) ...")

    # 收集多张图像的路由数据
    all_rows_per_image = []
    for idx, x in enumerate(inputs):
        rows, handles = collect_routing(model)
        with torch.no_grad():
            _ = model(x)
        for h in handles:
            h.remove()
        all_rows_per_image.append((idx, rows))
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(inputs)} images")

    # 聚合所有图像的路由数据（按 layer+expert 求平均）
    aggregated = {}
    for idx, rows in all_rows_per_image:
        for r in rows:
            key = (r["layer"], r["expert"])
            aggregated.setdefault(key, []).append(r["mean_weight"])

    final_rows = []
    for (layer, expert), weights in aggregated.items():
        # 取最后一个 row 的 kernel_size/is_top_k 字段（同一层同一专家这些值不变）
        sample_row = next(r for r in all_rows_per_image[-1][1] if r["layer"] == layer and r["expert"] == expert)
        final_rows.append({
            "layer": layer,
            "expert": expert,
            "kernel_size": sample_row["kernel_size"],
            "mean_weight": float(np.mean(weights)),
            "is_top_k": sample_row["is_top_k"],
        })

    n_esmoe_layers = len({r["layer"] for r in final_rows})
    print(f"[diagnose_esmoe_routing] ES_MOE hooked: {n_esmoe_layers}")
    print(f"[diagnose_esmoe_routing] Aggregated routing records: {len(final_rows)}")

    # 保存 CSV
    csv_path = output_dir / "routing_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "expert", "kernel_size", "mean_weight", "is_top_k"])
        writer.writeheader()
        for r in final_rows:
            writer.writerow(r)
    print(f"[diagnose_esmoe_routing] wrote {csv_path} ({len(final_rows)} rows)")

    # 场景化推荐 + summary
    recs = generate_recommendations(final_rows)
    rec_path = output_dir / "recommendations.json"
    rec_path.write_text(json.dumps(recs, indent=2, ensure_ascii=False))
    print(f"[diagnose_esmoe_routing] wrote {rec_path}")

    # 控制台输出
    print("\n=== ES-MoE 路由摘要（跨 4 个 ES_MOE 层）===")
    summary = recs["summary"]
    for expert_id in sorted(k for k in summary if k.startswith("expert_")):
        info = summary[expert_id]
        print(f"  Expert {expert_id[-1]} (ks={info['kernel_size']}): "
              f"mean={info['mean']:.3f}  std={info['std']:.3f}  "
              f"top1_rate={info['is_top1_active_rate']:.2f}")
    print(f"  __avg_routing_entropy__: H_norm={summary['__avg_routing_entropy__']:.3f}")
    print(f"  total_records={len(rows)}")

    print("\n=== 场景化推荐 ===")
    for rec in recs["recommendations"]:
        print(f"  • {rec}")

    # 热力图
    if args.plot:
        plot_path = output_dir / "routing_heatmap.png"
        plot_heatmap(rows, plot_path)
        print(f"[diagnose_esmoe_routing] wrote {plot_path}")


if __name__ == "__main__":
    main()