#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_esmoe_dynamic_topk.py
============================
ES-MoE 动态 Top-K 端到端训练（探路实验脚本，2026-08-15）。

实验结论（已判定激进方案不可行）
--------------------------------
训练中每 batch 按复杂度切换 routing.top_k（K=1/2/3，分位数自适应阈值），
2 epoch 后 mAP 崩至 0.04%（K=2 基线同期 0.75%，差 19 倍）。
根因：K=1 时仅 1 个专家有梯度、K=3 时 3 个全有，专家更新节奏剧烈波动，
MoE 部分无法收敛（box_loss 仅高 4% 但 mAP 崩 19 倍）。
与官方 AdaptiveCapacityMoE 设计记录一致：正确的后续方向是**可微复杂度调制**
（保持 top_k 固定、调制输出贡献），而非每 batch 硬切换。
本脚本保留作为探路记录，供复现与后续参考。

边界
----
仅验证了"每 batch 硬切换"这一激进方案；
固定 K=2 预训练后微调、按 epoch 切换等温和方案未测试。

用法
----
    python scripts/train_esmoe_dynamic_topk.py \
        --epochs 2 --batch 4 --imgsz 640 --device 0 \
        --name esmoe_v0_dynamic_topk_probe

输出
----
- 训练目录: {project}/{name}/
- 调度轨迹: {project}/{name}/dynamic_topk_schedule.csv
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.moe import ES_MOE


def compute_batch_complexity(img: torch.Tensor) -> float:
    """计算一个 batch 的平均图像复杂度（边缘强度 L1，与 dynamic_topk_router 一致）。

    注意：on_train_batch_start 时 batch["img"] 尚未 preprocess（uint8, 0~255），
    必须先转 float 并归一化到 [0, 1]，否则 mean() 对 Byte 报错。
    """
    img = img.float() / 255.0  # [B, 3, H, W] -> [0, 1]（非 in-place，不污染 batch）
    gray = img.mean(dim=1, keepdim=True)
    h_edges = torch.abs(gray[:, :, 1:] - gray[:, :, :-1]).mean()
    w_edges = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean()
    return float(((h_edges + w_edges) / 2.0).item())


def dynamic_top_k(complexity: float, num_experts: int = 4,
                  threshold_lo: float | None = None,
                  threshold_hi: float | None = None) -> int:
    """按 batch 复杂度映射 top_k（分位数自适应）。

    真实 VisDrone 图复杂度集中在 [0.008, 0.035]（合成图阈值 0.05/0.15/0.30 严重失配）。
    因此探路阶段用**分位数切分**：
        complexity < p33          -> top_k=1
        p33 <= complexity < p66   -> top_k=2
        complexity >= p66         -> top_k=3
    当未提供分位数时退化为固定阈值（便于直接调用）。
    """
    if threshold_lo is not None and threshold_hi is not None:
        if complexity < threshold_lo:
            return 1
        elif complexity < threshold_hi:
            return 2
        else:
            return min(3, num_experts)
    # 默认固定阈值（合成图适用）
    if complexity < 0.05:
        return 1
    elif complexity < 0.15:
        return 2
    elif complexity < 0.30:
        return min(3, num_experts)
    else:
        return num_experts


class ComplexityQuantileCalibrator:
    """预热期收集复杂度分布，之后用 p33/p66 分位数切 top_k。

    - 前 CALIB_BATCHES 个 batch 视为校准期：不设置 top_k（保持训练默认 K=2），
      只记录复杂度；校准期结束一次性计算分位数。
    - 校准期后每个 batch 按分位数映射 top_k。
    """

    CALIB_BATCHES = 128

    def __init__(self, num_experts: int = 4):
        self.num_experts = num_experts
        self._samples: list[float] = []
        self._thresholds: tuple[float, float] | None = None
        self.calibrated = False

    def step(self, complexity: float) -> int | None:
        """输入复杂度，返回该 batch 应使用的 top_k；校准期返回 None（不干预）。"""
        if not self.calibrated:
            self._samples.append(complexity)
            if len(self._samples) >= self.CALIB_BATCHES:
                s = sorted(self._samples)
                n = len(s)
                lo = s[max(int(n * 0.33) - 1, 0)]
                hi = s[max(int(n * 0.66) - 1, 0)]
                self._thresholds = (lo, hi)
                self.calibrated = True
                print(f"[QuantileCalibrator] calibrated over {n} batches: "
                      f"p33={lo:.4f} p66={hi:.4f} (complexity range "
                      f"{min(s):.4f}~{max(s):.4f})")
            return None
        return dynamic_top_k(complexity, self.num_experts, *self._thresholds)


def set_model_top_k(model, top_k: int) -> int:
    """设置模型中所有 ES_MOE 的 top_k 与 routing.top_k（训练侧同步）。"""
    count = 0
    for module in model.modules():
        if isinstance(module, ES_MOE):
            k = min(int(top_k), module.num_experts)
            module.top_k = k
            module.use_top_k = k < module.num_experts
            # 关键：同步 routing 层的 top_k，训练时 _soft_top_k 才会按新 k 掩码
            routing = getattr(module, "routing", None)
            if routing is not None:
                routing.top_k = k
                routing.use_top_k = k < module.num_experts
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="ES-MoE 动态 Top-K 端到端训练（探路）")
    parser.add_argument("--model", type=str,
                        default="ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml",
                        help="起始模型（YAML 或权重路径）")
    parser.add_argument("--data", type=str,
                        default="ultralytics/cfg/datasets/VisDrone.yaml",
                        help="数据集配置")
    parser.add_argument("--epochs", type=int, default=2, help="训练 epoch 数（探路用 2）")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="experiments_zviolin/runs")
    parser.add_argument("--name", type=str, default="esmoe_v0_dynamic_topk_probe")
    parser.add_argument("--balance-loss", type=float, default=1.0)
    parser.add_argument("--z-loss", type=float, default=0.1)
    args = parser.parse_args()

    # 调度轨迹 CSV
    csv_path = Path(args.project) / args.name / "dynamic_topk_schedule.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "top_k_distribution": {}, "modified": 0}
    calibrator = ComplexityQuantileCalibrator(num_experts=4)

    def on_train_batch_start(trainer):
        """每 batch 前：按复杂度设置动态 top_k（分位数自适应）。"""
        batch = getattr(trainer, "batch", None)
        if batch is None or "img" not in batch:
            return
        img = batch["img"]  # [B, 3, H, W]，GPU tensor
        complexity = compute_batch_complexity(img)

        top_k = calibrator.step(complexity)
        if top_k is None:
            return  # 校准期不干预（保持默认 K=2）

        n = set_model_top_k(trainer.model, top_k)
        stats["total"] += 1
        stats["top_k_distribution"][top_k] = stats["top_k_distribution"].get(top_k, 0) + 1
        stats["modified"] += n

        # 记录调度轨迹
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["epoch", "batch", "complexity", "top_k", "n_modules"])
            writer.writerow([
                getattr(trainer, "epoch", 0),
                stats["total"],
                f"{complexity:.4f}",
                top_k,
                n,
            ])

    model = YOLO(args.model)
    model.add_callback("on_train_batch_start", on_train_batch_start)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        amp=True,
        cos_lr=True,
        cls_remap=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_balance_loss=args.balance_loss,
        moe_router_z_loss=args.z_loss,
    )

    print(f"\n[DynamicTopK] 调度统计: total={stats['total']}, "
          f"distribution={stats['top_k_distribution']}")
    print(f"[DynamicTopK] 调度轨迹已写入: {csv_path}")


if __name__ == "__main__":
    main()
