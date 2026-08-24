#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_dynamic_topk_probe.py
===============================
动态 top_k 探路模型验证诊断脚本（2026-08-15）。

用途
----
排查"探路训练后验证 mAP 崩至 0.04%"的原因：
1. 打印 checkpoint 中所有 ES_MOE 的 top_k / routing.top_k 状态
   （探路训练会残留最后一个 batch 设置的 top_k，需确认是否污染验证）
2. 用固定 K=2 重新验证 → 若 mAP 仍低，说明是训练本身的问题而非验证污染

诊断结论（2026-08-15）
----------------------
- checkpoint 确实残留 top_k=1（最后训练 batch 的值被存进权重）
- 但固定 K=2 重新验证 mAP 仍只有 0.04%（K=2 基线同期 0.75%）
- → 确认是训练本身崩了（每 batch 切换 routing.top_k 导致梯度冲突），
   而非验证阶段 top_k 状态污染

用法
----
    python scripts/diagnose_dynamic_topk_probe.py \
        <checkpoint.pt> <VisDrone.yaml>
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.moe import ES_MOE


def inspect_topk_state(model, label: str) -> None:
    """打印所有 ES_MOE 模块的 top_k 状态。"""
    print(f"\n=== {label} ===")
    for name, module in model.named_modules():
        if isinstance(module, ES_MOE):
            routing = getattr(module, "routing", None)
            print(
                f"  {name}: top_k={module.top_k} use_top_k={module.use_top_k} "
                f"use_sparse={module.use_sparse_inference} "
                f"routing.top_k={getattr(routing, 'top_k', None)} "
                f"routing.use_top_k={getattr(routing, 'use_top_k', None)}"
            )


def set_fixed_k(model, k: int) -> int:
    """把所有 ES_MOE 固定为 top_k=k（与训练配置一致，仅验证用）。"""
    count = 0
    for module in model.modules():
        if isinstance(module, ES_MOE):
            module.top_k = k
            module.use_top_k = k < module.num_experts
            routing = getattr(module, "routing", None)
            if routing is not None:
                routing.top_k = k
                routing.use_top_k = k < module.num_experts
            count += 1
    return count


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else (
        r"runs/detect/experiments_zviolin/runs/esmoe_v0_dynamic_topk_probe/weights/best.pt")
    data = sys.argv[2] if len(sys.argv) > 2 else "ultralytics/cfg/datasets/VisDrone.yaml"

    print(f"checkpoint: {ckpt}")

    # 1. 加载后检查原始状态
    model = YOLO(str(ckpt))
    inspect_topk_state(model.model, "加载后原始状态")

    # 2. 用固定 K=2 验证
    n = set_fixed_k(model.model, 2)
    print(f"\n[diag] 已固定 {n} 个 ES_MOE 为 top_k=2，开始验证...")
    metrics = model.val(data=data, imgsz=640, batch=4, device=0, verbose=False)
    print(f"[diag] 固定 K=2 验证: mAP50-95={metrics.box.map:.4f} mAP50={metrics.box.map50:.4f}")

    # 3. 用固定 K=1 验证
    n = set_fixed_k(model.model, 1)
    print(f"\n[diag] 已固定 {n} 个 ES_MOE 为 top_k=1，开始验证...")
    metrics = model.val(data=data, imgsz=640, batch=4, device=0, verbose=False)
    print(f"[diag] 固定 K=1 验证: mAP50-95={metrics.box.map:.4f} mAP50={metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
