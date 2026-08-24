#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_esmoe_edge.py
=======================
ES-MoE 端侧优化 Benchmark 脚本。

对比 4 种端侧部署配置：
1. **基线**：yolo-master-esmoe-n-visdrone.yaml（无端侧优化）
2. **跨尺度共享**：yolo-master-esmoe-n-visdrone-shared.yaml（SharedExpertESMoE）
3. **剪枝版**：基于基线 + prune_esmoe_for_edge.py 输出的低利用率专家剪枝
4. **跨尺度共享 + 剪枝**：方案 2 + 方案 3 的组合

测量指标：
- 参数量（M）
- 模型文件大小（MB）
- CPU 延迟（ms）
- GPU 延迟（ms）
- 内存占用峰值（MB）
- FLOPs（G）

使用方法：
    # 训练基线 + 共享变体
    python scripts/compare_esmoe_ablation.py --train \\
        --models esmoe_v0_k2 esmoe_v0_k2_shared --epochs 100 --imgsz 640

    # 剪枝
    python scripts/prune_esmoe_for_edge.py \\
        --model experiments_zviolin/runs/esmoe_v0_k2_shared/weights/best.pt \\
        --threshold 0.10 --output experiments_zviolin/runs/esmoe_pruned_shared

    # 端侧 benchmark
    python scripts/compare_esmoe_edge.py --benchmark \\
        --models esmoe_v0_k2 esmoe_v0_k2_shared esmoe_v0_k2_pruned esmoe_v0_k2_shared_pruned \\
        --imgsz 640 --warmup 500 --reps 1000 --device cpu
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from ultralytics.nn.modules.moe import ES_MOE
from ultralytics.nn.modules.moe.shared_expert_esmoe import SharedExpertESMoE


def measure_model_metrics(
    weights_path: str,
    device: str,
    imgsz: int,
    warmup: int = 500,
    reps: int = 1000,
) -> dict:
    """测量模型的参数、延迟、内存、FLOPs 等指标。"""
    print(f"  Loading {weights_path} ...")
    yolo = YOLO(weights_path)
    model = yolo.model.to(device).eval()

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 模型文件大小
    file_size_mb = os.path.getsize(weights_path) / 1024 / 1024

    # FLOPs（用 torch profiler 估算）
    try:
        from ultralytics.utils.torch_utils import get_flops
        flops_g = get_flops(model, imgsz=imgsz) / 1e9  # 转换为 G
    except Exception:
        flops_g = 0.0

    # 测量延迟
    x = torch.randn(1, 3, imgsz, imgsz, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 测量
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    latency_ms = (t1 - t0) / reps * 1000

    # 内存峰值
    peak_memory_mb = 0
    if device.startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "weights_path": weights_path,
        "total_params": total_params,
        "total_params_m": total_params / 1e6,
        "trainable_params": trainable_params,
        "file_size_mb": file_size_mb,
        "flops_g": flops_g,
        "latency_ms_mean": latency_ms,
        "peak_memory_mb": peak_memory_mb,
        "device": device,
        "imgsz": imgsz,
        "reps": reps,
        "warmup": warmup,
    }


def main():
    parser = argparse.ArgumentParser(description="ES-MoE 端侧优化 Benchmark")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="模型 key 列表（与 compare_esmoe_ablation.py 一致）")
    parser.add_argument("--project", type=str, default="experiments_zviolin/runs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="experiments_zviolin/runs/esmoe_edge")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[compare_esmoe_edge] Benchmark {len(args.models)} models on {args.device} ...")

    results = []
    for key in args.models:
        # 查找权重：优先显式路径，其次自动匹配（支持 runs/detect/ 前缀与数字后缀目录）
        candidates = [
            Path(args.project) / key / "weights" / "best.pt",
            Path(args.project) / f"{key}2" / "weights" / "best.pt",
            Path("runs/detect") / args.project / key / "weights" / "best.pt",
            Path("runs/detect") / args.project / f"{key}2" / "weights" / "best.pt",
        ]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            # 兜底：自动匹配带数字后缀的目录（如 esmoe_v0_k1 -> esmoe_v0_k1-3）
            for base in (Path(args.project), Path("runs/detect") / args.project):
                if not base.exists():
                    continue
                matched = sorted(base.glob(f"{key}*/weights/best.pt"))
                if matched:
                    ckpt = matched[0]
                    break
        if ckpt is None:
            print(f"  [WARN] No checkpoint for {key}, skipping")
            continue

        result = measure_model_metrics(
            str(ckpt), args.device, args.imgsz,
            warmup=args.warmup, reps=args.reps,
        )
        result["key"] = key
        results.append(result)
        print(f"  {key}: params={result['total_params_m']:.3f}M, "
              f"size={result['file_size_mb']:.2f}MB, "
              f"latency={result['latency_ms_mean']:.3f}ms, "
              f"flops={result['flops_g']:.2f}G")

    if not results:
        print("[compare_esmoe_edge] No results to save")
        return

    # 保存 CSV
    device_safe = args.device.replace(":", "_")
    csv_path = output_dir / f"edge_benchmark_{device_safe}_{args.imgsz}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n[compare_esmoe_edge] wrote {csv_path}")

    # 对比基线 + 共享版本
    if "esmoe_v0_k2" in [r["key"] for r in results]:
        base = next(r for r in results if r["key"] == "esmoe_v0_k2")
        print(f"\n=== Edge Optimization Comparison (vs baseline {base['key']}) ===")
        print(f"  Baseline: {base['total_params_m']:.3f}M, {base['latency_ms_mean']:.3f}ms, "
              f"{base['file_size_mb']:.2f}MB")

        for r in results:
            if r["key"] == "esmoe_v0_k2":
                continue
            param_reduce = (base['total_params_m'] - r['total_params_m']) / base['total_params_m'] * 100
            latency_reduce = (base['latency_ms_mean'] - r['latency_ms_mean']) / base['latency_ms_mean'] * 100
            size_reduce = (base['file_size_mb'] - r['file_size_mb']) / base['file_size_mb'] * 100
            print(f"  {r['key']:30s}: params -{param_reduce:5.1f}%, latency -{latency_reduce:5.1f}%, size -{size_reduce:5.1f}%")


if __name__ == "__main__":
    main()