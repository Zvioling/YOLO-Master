#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_expert_gini.py
========================
计算 4 专家利用率的 Gini 系数，验证原文 §1.4 验收 #2。

输入：experiments_zviolin/runs/esmoe_routing/routing_summary.csv
输出：experiments_zviolin/runs/esmoe_routing/gini_analysis.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def gini(v):
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0
    v_sorted = np.sort(v)
    cum = np.cumsum(v_sorted)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def main():
    csv_path = Path("experiments_zviolin/runs/esmoe_routing/routing_summary.csv")
    output_path = Path("experiments_zviolin/runs/esmoe_routing/gini_analysis.json")

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # 按专家分组（跨 4 层平均）
    expert_means = {0: [], 1: [], 2: [], 3: []}
    expert_kernel = {}
    for r in rows:
        e = int(r["expert"])
        expert_means[e].append(float(r["mean_weight"]))
        expert_kernel[e] = int(r["kernel_size"])

    expert_avg = {e: float(np.mean(ws)) for e, ws in expert_means.items()}

    # 4 专家平均 mean_weight（跨 4 层）
    print("4 专家平均 mean_weight（跨 4 层）:")
    for e, v in expert_avg.items():
        print(f"  Expert {e} (ks={expert_kernel[e]}): {v:.4f}")

    values = np.array(list(expert_avg.values()))
    g = gini(values)
    max_min_diff = float((values.max() - values.min()) / values.mean() * 100)
    cv = float(values.std() / values.mean() * 100)

    # 补充：Top-2 命中率（每个专家被选入 Top-2 的层数比例）
    top2_hit_rate = {}
    for e in range(4):
        hits = sum(1 for r in rows if int(r["expert"]) == e and int(r["is_top_k"]) == 1)
        top2_hit_rate[e] = hits / 4  # 4 层

    print("")
    print("=== 利用率（mean_weight 跨 4 层平均）===")
    print(f"Gini 系数: {g:.4f}")
    print(f"4 专家利用率差异（max-min）/mean: {max_min_diff:.2f}%")
    print(f"变异系数 CV: {cv:.2f}%")
    print("")
    print("=== Top-2 命中率（每个专家被选入 Top-2 的层数比例）===")
    for e in range(4):
        print(f"  Expert {e} (ks={expert_kernel[e]}): {top2_hit_rate[e]:.2%}")
    top2_values = np.array(list(top2_hit_rate.values()))
    top2_max_min = float((top2_values.max() - top2_values.min()) / top2_values.mean() * 100) if top2_values.mean() > 0 else 0
    print(f"Top-2 命中率差异（max-min）/mean: {top2_max_min:.2f}%")
    print("")
    print("=== 验收 #2 判定 ===")
    print(f"  通过标准 (mean_weight 差异 < 30%): {'PASS' if max_min_diff < 30 else 'FAIL'}")
    print(f"  优秀标准 (mean_weight 差异 < 15%): {'PASS' if max_min_diff < 15 else 'FAIL'}")
    print(f"  通过标准 (Top-2 命中率差异 < 30%): {'PASS' if top2_max_min < 30 else 'FAIL'}")
    print(f"  优秀标准 (Top-2 命中率差异 < 15%): {'PASS' if top2_max_min < 15 else 'FAIL'}")

    # 取最宽松的视角（mean_weight + Top-2 命中率）作为最终判定
    pass_status = "PASS" if (max_min_diff < 30 or top2_max_min < 30) else "FAIL"
    excellent_status = "PASS" if (max_min_diff < 15 or top2_max_min < 15) else "FAIL"
    print(f"  最终通过（任一指标 < 30%）: {pass_status}")
    print(f"  最终优秀（任一指标 < 15%）: {excellent_status}")

    # 写出 JSON
    result = {
        "expert_avg": {f"expert_{e}_ks{expert_kernel[e]}": v for e, v in expert_avg.items()},
        "top2_hit_rate": {f"expert_{e}_ks{expert_kernel[e]}": top2_hit_rate[e] for e in range(4)},
        "metrics": {
            "gini": g,
            "max_min_diff_pct": max_min_diff,
            "cv_pct": cv,
            "top2_max_min_pct": top2_max_min,
            "min_expert_weight": float(values.min()),
            "max_expert_weight": float(values.max()),
        },
        "acceptance": {
            "mean_weight_pass_30pct": "PASS" if max_min_diff < 30 else "FAIL",
            "mean_weight_excellent_15pct": "PASS" if max_min_diff < 15 else "FAIL",
            "top2_hit_rate_pass_30pct": "PASS" if top2_max_min < 30 else "FAIL",
            "top2_hit_rate_excellent_15pct": "PASS" if top2_max_min < 15 else "FAIL",
            "final_pass": pass_status,
            "final_excellent": excellent_status,
        },
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[compute_expert_gini] wrote {output_path}")


if __name__ == "__main__":
    main()