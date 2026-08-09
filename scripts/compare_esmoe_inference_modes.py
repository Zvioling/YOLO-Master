#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_esmoe_inference_modes.py
================================
ES-MoE 推理模式对比：Dense / Soft / Hard / 自适应

输入：ES-MoE 权重
输出：experiments_zviolin/runs/esmoe_inference_mode/inference_modes.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

# 确保 sys.path
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE


def measure(model, x, warmup, reps, device):
    """测量模型在指定设备上的延迟。"""
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / reps * 1000


def set_esmoe_mode(model, mode: str):
    """设置所有 ES_MOE 模块的路由模式。"""
    count = 0
    for module in model.modules():
        if isinstance(module, ES_MOE):
            if mode == "dense":
                # 强制 dense softmax（所有 4 专家激活）
                module.use_sparse_inference = False
                module.use_top_k = False
            elif mode == "soft":
                # Soft Top-K（训练时使用的 mask 模式）
                module.use_sparse_inference = True
                module.use_top_k = True
                module.top_k = 2
            elif mode == "hard":
                # Hard Top-K（推理时使用的 sparse scatter）
                module.use_sparse_inference = True
                module.use_top_k = True
                module.top_k = 2
                # 关键：让 forward 走 sparse 分支（不是 training）
                module.training = False
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="ES-MoE 推理模式对比")
    parser.add_argument("--model", type=str, required=True, help="ES-MoE 权重路径")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--output", type=str, default="experiments_zviolin/runs/esmoe_inference_mode")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare_esmoe_inference_modes] Loading {args.model}")
    yolo = YOLO(args.model)
    model = yolo.model.to(args.device).eval()

    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)

    # 默认模式（默认 sparse_forward）+ 强制 dense
    results = []
    # 模式 1：默认（sparse_forward，2/4 experts）
    print(f"\n[compare_esmoe_inference_modes] === Mode: hard (default, 2/4 experts) ===")
    n = set_esmoe_mode(model, "hard")
    print(f"  Set {n} ES_MOE modules to default (sparse)")
    # 验证 _eager_sparse_enabled
    eager_states = [m._eager_sparse_enabled() for m in model.modules() if isinstance(m, ES_MOE)]
    print(f"  _eager_sparse_enabled states: {set(eager_states)}")
    latency_hard = measure(model, x, args.warmup, args.reps, args.device)
    print(f"  Latency: {latency_hard:.3f} ms/frame")
    results.append({
        "mode": "hard_default_sparse",
        "latency_ms": round(latency_hard, 3),
        "compute_experts": "2/4 (sparse)",
        "note": "推理 Hard Top-K，默认 _eager_sparse_enabled=True",
    })

    # 模式 2：强制 dense
    print(f"\n[compare_esmoe_inference_modes] === Mode: dense (forced, 4/4 experts) ===")
    n = set_esmoe_mode(model, "dense")
    print(f"  Set {n} ES_MOE modules to dense (forced)")
    eager_states = [m._eager_sparse_enabled() for m in model.modules() if isinstance(m, ES_MOE)]
    print(f"  _eager_sparse_enabled states: {set(eager_states)}")
    latency_dense = measure(model, x, args.warmup, args.reps, args.device)
    print(f"  Latency: {latency_dense:.3f} ms/frame")
    results.append({
        "mode": "dense_forced",
        "latency_ms": round(latency_dense, 3),
        "compute_experts": "4/4 (dense)",
        "note": "强制 dense softmax，所有 4 个专家都计算",
    })

    # 保存 CSV
    csv_path = output_dir / "inference_modes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n[compare_esmoe_inference_modes] wrote {csv_path}")

    # 对比
    if len(results) >= 2:
        dense_lat = next(r["latency_ms"] for r in results if r["mode"] == "dense_forced")
        hard_lat = next(r["latency_ms"] for r in results if r["mode"] == "hard_default_sparse")
        reduction = (dense_lat - hard_lat) / dense_lat * 100
        print(f"\n[compare_esmoe_inference_modes] === 对比 ===")
        print(f"  Dense (forced, 4/4): {dense_lat:.3f} ms")
        print(f"  Hard (default sparse, 2/4): {hard_lat:.3f} ms")
        if reduction > 0:
            print(f"  Hard vs Dense: -{reduction:.1f}% 延迟（稀疏化有效）")
        else:
            print(f"  Hard vs Dense: +{-reduction:.1f}% 延迟（稀疏化 overhead > 收益）")


if __name__ == "__main__":
    main()