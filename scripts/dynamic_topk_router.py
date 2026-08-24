#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dynamic_topk_router.py
=======================
动态 Top-K 路由器：根据输入图像复杂度自动选择 ES-MoE 的 top_k。

设计动机
--------
ES-MoE 训练时固定 top_k=2（激活 2/4 专家），但在真实推理场景中：
- 空旷走廊（30%）：仅需 top_k=1（节省 50% 计算）
- 中等密度（50%）：top_k=2 即可
- 拥挤商场（20%）：需 top_k=3（精度优先）

实现思路
--------
1. compute_complexity(image) - 简单的图像复杂度评分（基于边缘强度）
2. dynamic_top_k(complexity) - 根据复杂度返回 top_k
3. DynamicBatchPredictor     - 自动修改模型 top_k 并推理

使用方法：
    # 单张图像推理
    python scripts/dynamic_topk_router.py \\
        --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \\
        --image path/to/image.jpg

    # 验证动态 Top-K 分布
    python scripts/dynamic_topk_router.py --verify-distribution \\
        --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \\
        --num-samples 50
"""
from __future__ import annotations

# 确保 sys.path 正确（PowerShell 下 cwd 不会自动加入 sys.path）
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import torch


def compute_complexity(image: torch.Tensor) -> float:
    """
    简单的图像复杂度评分：基于灰度图的边缘强度（L1 范数）。

    Args:
        image: [B, 3, H, W] 或 [3, H, W] 范围的张量，像素值 [0, 1] 或任意范围

    Returns:
        复杂度评分（值越大表示越复杂），典型范围 [0, 0.5]
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # 转灰度
    gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # 水平边缘 + 垂直边缘
    h_edges = torch.abs(gray[:, :, 1:] - gray[:, :, :-1]).mean()
    w_edges = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean()
    edges = (h_edges + w_edges) / 2.0

    return float(edges.item())


def dynamic_top_k(complexity: float, num_experts: int = 4) -> int:
    """
    根据图像复杂度返回 top_k。

    Args:
        complexity: 图像复杂度评分（[0, 0.5]）
        num_experts: ES-MoE 专家总数

    Returns:
        top_k ∈ [1, num_experts]
    """
    if complexity < 0.05:
        return 1  # 极简场景（空旷走廊、纯背景）
    elif complexity < 0.15:
        return 2  # 简单场景
    elif complexity < 0.30:
        return min(3, num_experts)  # 中等场景
    else:
        return num_experts  # 复杂场景（拥挤、密集目标）


def set_esmoe_top_k(model: torch.nn.Module, top_k: int) -> int:
    """
    设置模型中所有 ES_MOE 模块的 top_k。

    Args:
        model: YOLO 模型（已加载权重）
        top_k: 新的 top_k 值

    Returns:
        成功修改的 ES_MOE 模块数
    """
    from ultralytics.nn.modules.moe import ES_MOE

    count = 0
    for module in model.modules():
        if isinstance(module, ES_MOE):
            module.top_k = min(top_k, module.num_experts)
            module.use_top_k = module.top_k < module.num_experts
            count += 1
    return count


class DynamicBatchPredictor:
    """
    根据输入图像复杂度动态调整 ES-MoE top_k 的预测器。

    使用方法：
        predictor = DynamicBatchPredictor('path/to/best.pt')
        results = predictor.predict(['img1.jpg', 'img2.jpg', ...])
    """

    def __init__(self, model_path: str, device: str = "cpu", verbose: bool = True):
        from ultralytics import YOLO
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model
        self.device = device
        self.verbose = verbose
        self._stats = {"total": 0, "top_k_distribution": {1: 0, 2: 0, 3: 0, 4: 0}}

    def _load_image_tensor(self, image_path: str) -> torch.Tensor:
        """加载图像为 [3, H, W] 张量（[0, 1] 范围，避免 numpy）。"""
        from PIL import Image
        img = Image.open(image_path).convert("RGB").resize((640, 640))
        # 用 torchvision.io.read_image 替代 numpy
        import torchvision.transforms.functional as TF
        tensor = TF.to_tensor(img)  # PIL -> tensor [0, 1]
        return tensor

    def predict_single(self, image_path: str):
        """对单张图像进行动态 Top-K 推理。"""
        x = self._load_image_tensor(image_path)
        complexity = compute_complexity(x)
        top_k = dynamic_top_k(complexity, num_experts=4)
        self._stats["total"] += 1
        self._stats["top_k_distribution"][top_k] = \
            self._stats["top_k_distribution"].get(top_k, 0) + 1

        if self.verbose:
            print(f"  {Path(image_path).name}: complexity={complexity:.3f}, top_k={top_k}")

        n = set_esmoe_top_k(self.model, top_k)
        if self.verbose and n:
            print(f"    -> updated {n} ES_MOE modules")

        # 用 YOLO 的标准推理接口
        return self.yolo(image_path, verbose=False)

    def predict_batch(self, image_paths: List[str]):
        """对一组图像进行动态 Top-K 推理。"""
        if self.verbose:
            print(f"\n[DynamicBatchPredictor] Processing {len(image_paths)} images ...")

        results = []
        for path in image_paths:
            result = self.predict_single(path)
            results.append(result)

        if self.verbose:
            self.print_stats()
        return results

    def print_stats(self):
        """打印动态 Top-K 分布统计。"""
        total = self._stats["total"]
        if total == 0:
            return
        print(f"\n[DynamicBatchPredictor] Stats: total={total}")
        print(f"  {'top_k':<10} {'count':>8} {'%':>8}")
        for k in sorted(self._stats["top_k_distribution"].keys()):
            cnt = self._stats["top_k_distribution"][k]
            pct = cnt / total * 100
            print(f"  {k:<10} {cnt:>8} {pct:>7.1f}%")


def verify_distribution(model_path: str, image_paths: List[str], output: str = None) -> dict:
    """
    验证动态 Top-K 在一组图像上的分布情况。

    Args:
        model_path: ES-MoE 权重路径
        image_paths: 待验证图像路径列表
        output: 输出 CSV 路径（可选）

    Returns:
        dict: {"complexity": [...], "top_k": [...], "image": [...]}
    """
    predictor = DynamicBatchPredictor(model_path, verbose=False)

    complexity_list = []
    top_k_list = []
    image_names = []

    for path in image_paths:
        x = predictor._load_image_tensor(path)
        complexity = compute_complexity(x)
        top_k = dynamic_top_k(complexity, num_experts=4)
        complexity_list.append(complexity)
        top_k_list.append(top_k)
        image_names.append(Path(path).name)

    # 统计
    from collections import Counter
    dist = Counter(top_k_list)

    result = {
        "image": image_names,
        "complexity": complexity_list,
        "top_k": top_k_list,
        "distribution": dict(dist),
        "avg_top_k": float(np.mean(top_k_list)),
        "avg_complexity": float(np.mean(complexity_list)),
    }

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "complexity", "top_k"])
            writer.writeheader()
            for i in range(len(image_names)):
                writer.writerow({
                    "image": image_names[i],
                    "complexity": f"{complexity_list[i]:.4f}",
                    "top_k": top_k_list[i],
                })
        print(f"[verify_distribution] wrote {output_path}")

    print(f"\n[verify_distribution] Summary: {len(image_paths)} images")
    print(f"  avg_complexity={result['avg_complexity']:.4f}, avg_top_k={result['avg_top_k']:.2f}")
    print(f"  distribution: {dist}")
    return result


def main():
    parser = argparse.ArgumentParser(description="ES-MoE 动态 Top-K 路由器")
    parser.add_argument("--model", type=str, required=True, help="ES-MoE 权重路径")
    parser.add_argument("--image", type=str, default=None, help="单张图像推理")
    parser.add_argument("--verify-distribution", action="store_true", help="验证 Top-K 分布")
    parser.add_argument("--image-dir", type=str, default=None, help="图像目录（与 --verify-distribution 配合）")
    parser.add_argument("--num-samples", type=int, default=50, help="采样数量")
    parser.add_argument("--output", type=str, default="experiments_zviolin/runs/esmoe_dynamic_topk",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.image:
        predictor = DynamicBatchPredictor(args.model, device=args.device)
        result = predictor.predict_single(args.image)
        predictor.print_stats()
        return

    if args.verify_distribution:
        image_dir = Path(args.image_dir) if args.image_dir else None
        if image_dir and image_dir.exists():
            image_paths = sorted([
                str(p) for p in image_dir.glob("*.jpg")
            ])[:args.num_samples]
            image_paths += sorted([
                str(p) for p in image_dir.glob("*.png")
            ])[:args.num_samples - len(image_paths)]
            image_paths = image_paths[:args.num_samples]
        else:
            print(f"[main] image_dir '{image_dir}' not found, falling back to synthetic data")
            image_paths = []

        if not image_paths:
            # 生成合成图像作为 demo
            print("[main] Generating synthetic images for demo ...")
            torch.manual_seed(42)
            image_paths = []
            for i in range(args.num_samples):
                # 不同复杂度的合成图像
                base_complexity = i / args.num_samples
                noise = torch.randn(3, 320, 320) * (0.05 + base_complexity * 0.3)
                img_tensor = (noise + 0.5).clamp(0, 1)
                path = Path(args.output) / f"synthetic_{i:04d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                # 用 torchvision.utils.save_image（不依赖 numpy）
                import torchvision.utils as tv_utils
                tv_utils.save_image(img_tensor, str(path))
                image_paths.append(str(path))

        output_csv = Path(args.output) / "dynamic_topk_per_image.csv"
        verify_distribution(args.model, image_paths, output=str(output_csv))
        return

    parser.print_help()


if __name__ == "__main__":
    main()