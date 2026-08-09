#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_esmoe_ablation.py
==========================
ES-MoE 消融实验性能对比脚本。

支持的子命令：
  --train     : 在 VisDrone 数据集上训练 ES-MoE 变体（不同 top_k / balance_loss / z_loss）
  --benchmark : 测量延迟（CPU/GPU），与 v084 / v08_mot6 等基线对比
  --summary   : 汇总所有变体的 mAP / 延迟 / 参数量

与 compare_mot_ablation.py 的差异：
- 对比目标不同：ES-MoE vs MoE 基线（v084）vs MoT 变体（v08_mot6）
- YAML 配置路径：esmoe 用 v0/det/yolo-master-esmoe-n-visdrone.yaml

使用方法：
    # Benchmark 单个变体（CPU）
    python scripts/compare_esmoe_ablation.py --benchmark \\
        --models esmoe_v0_k2 --imgsz 640 --warmup 500 --reps 2000 --device cpu

    # Benchmark 多个变体对比
    python scripts/compare_esmoe_ablation.py --benchmark \\
        --models esmoe_v0_k2 esmoe_v0_k1 esmoe_v0_k3 v084 v08_mot6 \\
        --imgsz 640 --warmup 500 --reps 2000 --device cpu

    # 训练 ES-MoE（K=1 对照组）
    python scripts/compare_esmoe_ablation.py --train --models esmoe_v0_k1 \\
        --data ultralytics/cfg/datasets/VisDrone.yaml \\
        --epochs 100 --imgsz 640 --batch 4 --device 0

    # 汇总结果
    python scripts/compare_esmoe_ablation.py --summary \\
        --project experiments_zviolin/runs
"""
from __future__ import annotations

# 必须在 ultralytics 导入前确保 sys.path 正确（PowerShell 下 cwd 不会自动加入 sys.path）
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# 变体注册表（与 issue #54 的 compare_mot_ablation.py 风格一致）
# ============================================================
ESMOE_VARIANTS = [
    {
        "key": "esmoe_v0_k1",
        "label": "YOLO-Master ES-MoE (K=1, most sparse)",
        "cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml",
        "overrides": {"moe_top_k": 1, "moe_balance_loss": 1.0, "moe_router_z_loss": 0.1},
    },
    {
        "key": "esmoe_v0_k2",
        "label": "YOLO-Master ES-MoE (K=2, baseline)",
        "cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml",
        "overrides": {"moe_top_k": 2, "moe_balance_loss": 1.0, "moe_router_z_loss": 0.1},
    },
    {
        "key": "esmoe_v0_k3",
        "label": "YOLO-Master ES-MoE (K=3, most accurate)",
        "cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml",
        "overrides": {"moe_top_k": 3, "moe_balance_loss": 1.0, "moe_router_z_loss": 0.1},
    },
    {
        "key": "esmoe_v0_k2_bl01",
        "label": "YOLO-Master ES-MoE (K=2, balance_loss=0.01)",
        "cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml",
        "overrides": {"moe_top_k": 2, "moe_balance_loss": 0.01, "moe_router_z_loss": 0.1},
    },
    {
        "key": "esmoe_v0_k2_bl20",
        "label": "YOLO-Master ES-MoE (K=2, balance_loss=2.0)",
        "cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml",
        "overrides": {"moe_top_k": 2, "moe_balance_loss": 2.0, "moe_router_z_loss": 0.1},
    },
]

# 基线变体（来自 Issue #54）
BASELINE_VARIANTS = [
    {
        "key": "v084",
        "label": "YOLO-Master v0.8 MoE baseline",
        "cfg": "ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml",
        "overrides": {},
    },
    {
        "key": "v08_mot",
        "label": "YOLO-Master v0.8 MoT (3 experts)",
        "cfg": "ultralytics/cfg/models/master/v0_8/det/yolo-master-mot-n.yaml",
        "overrides": {},
    },
    {
        "key": "v08_moa",
        "label": "YOLO-Master v0.8 MoA (attention)",
        "cfg": "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-n.yaml",
        "overrides": {},
    },
]


# ============================================================
# 命令：train（训练 ES-MoE 变体）
# ============================================================
def cmd_train(args):
    """训练指定 ES-MoE 变体。"""
    requested = set(args.models or [v["key"] for v in ESMOE_VARIANTS])
    for variant in ESMOE_VARIANTS:
        if variant["key"] not in requested:
            continue

        print(f"\n[train] === Training {variant['key']} ===")
        print(f"  cfg={variant['cfg']}")
        print(f"  overrides={variant['overrides']}")

        model = YOLO(variant["cfg"])
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=variant["key"],
            exist_ok=False,
            amp=True,
            cos_lr=True,
            **variant["overrides"],
        )


# ============================================================
# 命令：benchmark（测量延迟）
# ============================================================
def cmd_benchmark(args):
    """测量指定变体的 CPU/GPU 延迟。"""
    # 合并 ES-MoE + 基线变体
    all_variants = {v["key"]: v for v in ESMOE_VARIANTS}
    for v in BASELINE_VARIANTS:
        all_variants.setdefault(v["key"], v)

    requested = args.models or [v["key"] for v in ESMOE_VARIANTS]
    device_safe = args.device.replace(":", "_")  # Windows 冒号兼容

    results = []
    for key in requested:
        if key not in all_variants:
            print(f"[benchmark] WARNING: unknown variant '{key}', skipping")
            continue

        variant = all_variants[key]
        # 优先从训练输出目录找 best.pt，否则用 YAML（YAML 模式路由不生效）
        # 支持多种命名：esmoe_v0_k2 / esmoe_v0_k2-2 / esmoe_v0_k2_2
        candidate_names = [key, f"{key}-2", f"{key}_2", f"{key}2"]
        ckpt = None
        for name in candidate_names:
            candidate = Path(args.project) / name / "weights" / "best.pt"
            if candidate.exists():
                ckpt = candidate
                break
        if ckpt is None:
            print(f"[benchmark] No checkpoint for {key}, using YAML cfg")
            ckpt = variant["cfg"]

        print(f"\n[benchmark] === {key} ===")
        print(f"  weights/cfg={ckpt}")

        try:
            yolo = YOLO(str(ckpt))
            model = yolo.model
            model.eval()
            model.to(args.device)
        except Exception as e:
            print(f"[benchmark] Failed to load {key}: {e}")
            continue

        # 模型信息
        try:
            params = sum(p.numel() for p in model.parameters())
            params_m = params / 1e6
        except Exception:
            params_m = 0.0

        # 预热
        x = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(x)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()

            # 测量
            t0 = time.perf_counter()
            for _ in range(args.reps):
                _ = model(x)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        latency_ms = (t1 - t0) / args.reps * 1000.0
        result = {
            "key": key,
            "label": variant["label"],
            "cfg": variant["cfg"],
            "params": str(params),
            "params_m": f"{params_m:.6f}",
            "device": args.device,
            "imgsz": str(args.imgsz),
            "latency_ms_mean": f"{latency_ms:.3f}",
            "latency_ms_min": "",  # 简化版不单独测 min/max
            "latency_ms_max": "",
            "reps": str(args.reps),
            "warmup": str(args.warmup),
        }
        results.append(result)
        print(f"  params={params_m:.3f}M  latency={latency_ms:.3f} ms/frame")

    # 保存 CSV
    if results:
        csv_path = Path(args.project) / f"latency_{device_safe}_{args.imgsz}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\n[benchmark] wrote {csv_path}")


# ============================================================
# 命令：summary（汇总所有变体结果）
# ============================================================
def cmd_summary(args):
    """汇总所有变体的 mAP / 延迟 / 参数量。"""
    project = Path(args.project)
    rows = []

    # 遍历所有可能的变体目录
    for variant in ESMOE_VARIANTS + BASELINE_VARIANTS:
        key = variant["key"]
        run_dir = project / key
        if not run_dir.exists():
            run_dir = project / f"{key}2"
        if not run_dir.exists():
            continue

        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            continue

        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            rows_list = list(reader)
        if not rows_list:
            continue
        last = rows_list[-1]
        try:
            map50 = float(last.get("metrics/mAP50(B)", 0))
            map50_95 = float(last.get("metrics/mAP50-95(B)", 0))
            precision = float(last.get("metrics/precision(B)", 0))
            recall = float(last.get("metrics/recall(B)", 0))
        except (ValueError, KeyError):
            continue

        rows.append({
            "key": key,
            "label": variant["label"],
            "epoch": last.get("epoch", ""),
            "mAP50": map50,
            "mAP50-95": map50_95,
            "precision": precision,
            "recall": recall,
            "run_dir": str(run_dir),
        })

    if not rows:
        print("[summary] No results found")
        return

    # 保存 summary.csv
    summary_path = project / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[summary] wrote {summary_path}")

    # 控制台输出
    print(f"\n{'key':<25} {'mAP50-95':>10} {'mAP50':>10} {'P':>8} {'R':>8}")
    print("-" * 65)
    for r in rows:
        print(f"{r['key']:<25} {r['mAP50-95']*100:>9.2f}% {r['mAP50']*100:>9.2f}% "
              f"{r['precision']*100:>7.2f}% {r['recall']*100:>7.2f}%")


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ES-MoE 消融实验对比")
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--benchmark", action="store_true", help="延迟测试模式")
    parser.add_argument("--summary", action="store_true", help="汇总模式")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="变体 key 列表")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/VisDrone.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--warmup", type=int, default=500, help="benchmark 预热次数")
    parser.add_argument("--reps", type=int, default=2000, help="benchmark 测量次数")
    parser.add_argument("--project", type=str, default="experiments_zviolin/runs")
    args = parser.parse_args()

    if args.train:
        cmd_train(args)
    elif args.benchmark:
        cmd_benchmark(args)
    elif args.summary:
        cmd_summary(args)
    else:
        print("Please specify --train / --benchmark / --summary")
        parser.print_help()


if __name__ == "__main__":
    main()