<<<<<<< HEAD
"""MoT 路由可解释性诊断脚本（论文投稿增强版）。

论文投稿级别增强（2026-08-01）：
  1. Top-k 路由熵（MoT top_k=2 不仅看 top1，加权熵衡量路由置信度）
  2. 真实数据路由对比（可选，VisDrone 图像验证合成输入结论）
  3. 路由热力图自动生成（matplotlib，6 层 × 3 专家矩阵）

输出：
  - routing_summary.csv   : 18 行（6 层 × 3 专家），新增 entropy 列
  - recommendations.json  : 3 条场景化推荐
  - routing_heatmap.png   : 路由分布热力图（论文用）

用法：
  # 1. dry-run 模式（合成随机输入，无需数据集）
  python scripts/diagnose_mot_routing.py ^
      --model experiments_zviolin\runs\v08_mot6\weights\last.pt ^
      --dry-run --device cpu --plot --output experiments_zviolin\runs\mot_routing

  # 2. 真实数据模式（用 VisDrone 图像推理）
  python scripts/diagnose_mot_routing.py ^
      --model experiments_zviolin\runs\v08_mot6\weights\last.pt ^
      --device cpu --plot --num-samples 50 ^
      --data ultralytics\cfg\datasets\VisDrone.yaml ^
      --output experiments_zviolin\runs\mot_routing_real
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yaml as pyyaml  # noqa: F401
except ImportError:
    pyyaml = None

from ultralytics.nn.modules import MoTBlock  # noqa: E402
from ultralytics.nn.tasks import yaml_model_load, parse_model  # noqa: E402

EXPERT_NAMES = ["LocalConvTransformer", "WindowTransformer", "DeformableTransformer"]


def build_model_from_yaml(cfg_path: str):
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = ROOT / cfg_file
    d = yaml_model_load(str(cfg_file))
    model, _save = parse_model(d, ch=3, verbose=False)
    return model


def build_model_from_pt(pt_path: str):
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        model = ckpt.get("ema") or ckpt.get("model") or ckpt
    else:
        model = ckpt
    if hasattr(model, "float"):
        model = model.float()
    if hasattr(model, "fuse"):
        try:
            model = model.fuse()
        except Exception:
            pass
    model.eval()
    return model


def compute_entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对每个 token 计算 3 个专家权重的 Shannon 熵。

    H = -sum(p_i * log(p_i))，归一化到 [0, 1]（除以 log(E=3)）
    高熵 ≈ 路由不确定 / 低熵 ≈ 路由确定（单一专家主导）

    Args:
        weights: [B, E, H, W]（softmax 输出）
    Returns:
        [B, H, W] 每个 token 的归一化熵（0~1）
    """
    p = weights.clamp_min(eps)
    H = -(p * p.log()).sum(dim=1)  # [B, H, W]
    H_norm = H / math.log(weights.shape[1])  # log(E)
    return H_norm


def summarize_router_weights(name: str, weights: torch.Tensor, top_k: int = 2):
    """统计单个 MoTBlock 的路由数据（含 top-k 熵）。

    Args:
        weights: [B, E, H, W]
        top_k: MoT 实际激活的专家数（用于 top-k 命中率）
    Returns:
        list of dict
    """
    rows = []
    top1 = weights.argmax(dim=1)  # [B, H, W]
    total_tokens = top1.numel()

    # Top-k 命中率：top-k 专家被实际使用的比例
    topk_vals, topk_idx = weights.topk(top_k, dim=1)  # [B, k, H, W]
    used_experts = torch.zeros(weights.shape[1], dtype=torch.long)
    for k in range(top_k):
        for e_idx in range(weights.shape[1]):
            used_experts[e_idx] += (topk_idx[:, k] == e_idx).sum().item()

    # 平均权重 + 平均熵
    mean_weight = weights.mean(dim=(0, 2, 3))  # [E]
    entropy_norm = compute_entropy(weights).mean().item()

    for e_idx, expert_name in enumerate(EXPERT_NAMES):
        active = (top1 == e_idx).sum().item()
        topk_active = used_experts[e_idx].item()
        rows.append({
            "layer": name,
            "expert": expert_name,
            "active_tokens": int(active),
            "activation_ratio": active / total_tokens,
            "mean_weight": float(mean_weight[e_idx]),
            "topk_active_tokens": int(topk_active),
            "topk_activation_ratio": topk_active / (total_tokens * top_k),
        })
    rows.append({
        "layer": name + ".__summary__",
        "expert": "ROUTING_ENTROPY",
        "active_tokens": 0,
        "activation_ratio": 0.0,
        "mean_weight": entropy_norm,
        "topk_active_tokens": 0,
        "topk_activation_ratio": 0.0,
    })
    return rows


def make_hook(name: str, rows: List[Dict]):
    def hook(module, inputs, _output):
        with torch.no_grad():
            if hasattr(module, "router") and inputs:
                router_out = module.router(inputs[0])
                weights = router_out[0] if isinstance(router_out, tuple) else router_out
                rows.extend(summarize_router_weights(name, weights))
    return hook


def register_hooks(model) -> Tuple[List, List]:
    rows = []
    hooks = []
    hooked_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MoTBlock":
            hooks.append(module.register_forward_hook(make_hook(name, rows)))
            hooked_count += 1
    print(f"[diagnose_mot_routing] MoTBlock hooked: {hooked_count}")
    return rows, hooks


def aggregate_by_expert(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """汇总所有层每个专家的平均激活率 + 平均 top-k 激活率 + 平均权重。"""
    agg = {e: {"top1": [], "topk": [], "weight": []} for e in EXPERT_NAMES}
    entropies = []
    for r in rows:
        if r["expert"] in agg:
            agg[r["expert"]]["top1"].append(r["activation_ratio"])
            agg[r["expert"]]["topk"].append(r["topk_activation_ratio"])
            agg[r["expert"]]["weight"].append(r["mean_weight"])
        elif r["expert"] == "ROUTING_ENTROPY":
            entropies.append(r["mean_weight"])
    summary = {}
    for e in EXPERT_NAMES:
        v = agg[e]
        summary[e] = {
            "top1_mean": sum(v["top1"]) / len(v["top1"]) if v["top1"] else 0.0,
            "topk_mean": sum(v["topk"]) / len(v["topk"]) if v["topk"] else 0.0,
            "weight_mean": sum(v["weight"]) / len(v["weight"]) if v["weight"] else 0.0,
        }
    summary["__avg_routing_entropy__"] = {
        "top1_mean": sum(entropies) / len(entropies) if entropies else 0.0,
        "topk_mean": 0.0,
        "weight_mean": 0.0,
    }
    return summary


def write_csv(rows: List[Dict], csv_path: Path) -> None:
    fieldnames = ["layer", "expert", "active_tokens", "activation_ratio",
                  "mean_weight", "topk_active_tokens", "topk_activation_ratio"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[diagnose_mot_routing] wrote {csv_path} ({len(rows)} rows)")


def write_recommendations(summary: Dict[str, Dict[str, float]], json_path: Path) -> List[str]:
    recs = [
        (f"Dense or small-object scenes should inspect WindowTransformer first "
         f"when its top-1 activation ratio is high "
         f"({summary['WindowTransformer']['top1_mean']:.3f})."),
        (f"Occluded or irregular-object scenes should inspect DeformableTransformer "
         f"when its top-1 activation ratio rises "
         f"({summary['DeformableTransformer']['top1_mean']:.3f})."),
        (f"Latency-sensitive simple scenes can prefer LocalConvTransformer-heavy "
         f"routing (top-1 {summary['LocalConvTransformer']['top1_mean']:.3f}) "
         f"before enabling deeper MoT/MoE hybrids."),
        (f"Routing confidence (avg normalized entropy): "
         f"{summary['__avg_routing_entropy__']['top1_mean']:.3f} "
         f"(lower = more confident expert selection)."),
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"recommendations": recs, "summary": summary}, f, indent=2, ensure_ascii=False)
    print(f"[diagnose_mot_routing] wrote {json_path}")
    return recs


def plot_heatmap(rows: List[Dict], png_path: Path, title: str = "MoT Routing Heatmap") -> None:
    """绘制 6 层 × 3 专家的 top-1 激活率热力图（论文用）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    layer_rows: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r["expert"] == "ROUTING_ENTROPY":
            continue
        layer_rows.setdefault(r["layer"], {})[r["expert"]] = r["activation_ratio"]

    layers = list(layer_rows.keys())
    matrix = np.array([[layer_rows[L][e] for e in EXPERT_NAMES] for L in layers])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(EXPERT_NAMES)))
    ax.set_xticklabels([e.replace("Transformer", "") for e in EXPERT_NAMES], fontsize=10)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=9)
    ax.set_title(title, fontsize=12, pad=12)

    for i in range(len(layers)):
        for j in range(len(EXPERT_NAMES)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Top-1 Activation Ratio", fontsize=10)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[diagnose_mot_routing] wrote {png_path}")


def load_real_samples(data_cfg: str, num_samples: int, imgsz: int) -> Optional[torch.Tensor]:
    """从 VisDrone 数据集加载 num_samples 张真实图像（论文投稿级证据）。"""
    try:
        from ultralytics.data import build_dataloader
        from ultralytics.data.utils import check_det_dataset
    except Exception as e:
        print(f"[diagnose_mot_routing] ⚠️ 无法加载数据集: {e}")
        return None

    data_dict = check_det_dataset(data_cfg)
    loader = build_dataloader(
        dataset=data_dict["train"],
        imgsz=imgsz,
        batch_size=1,
        stride=32,
        pad=0.5,
        single_cls=False,
        rect=False,
        workers=0,
    )[0]
    samples = []
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        # batch 可能是 dict 或 tuple，统一取 img
        img = batch[0] if isinstance(batch, (list, tuple)) else batch["img"]
        samples.append(img)
    if not samples:
        return None
    return torch.cat(samples, dim=0)  # [N, 3, imgsz, imgsz]


def run_dry_run(model, device: str, imgsz: int) -> List[Dict]:
    """合成随机输入跑一次前向。"""
    rows, hooks = register_hooks(model)
    with torch.no_grad():
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        _ = model(x)
    for h in hooks:
        h.remove()
    return rows


def run_real_data(model, device: str, x: torch.Tensor) -> List[Dict]:
    """真实数据模式：聚合多张图像的路由数据。"""
    rows, hooks = register_hooks(model)
    with torch.no_grad():
        x = x.to(device)
        _ = model(x)
    for h in hooks:
        h.remove()
    return rows


def main():
    parser = argparse.ArgumentParser(description="MoT 路由可解释性诊断（论文投稿增强版）")
    parser.add_argument("--model", required=True, help="YAML 或 .pt 权重")
    parser.add_argument("--dry-run", action="store_true", help="合成随机输入（无需数据集）")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0", "0"])
    parser.add_argument("--output", default="experiments_zviolin/runs/mot_routing")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--num-samples", type=int, default=20,
                        help="真实数据模式下的图像数（默认 20）")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml",
                        help="真实数据模式下的数据集配置")
    parser.add_argument("--plot", action="store_true", help="生成热力图 PNG")
    parser.add_argument("--top-k", type=int, default=2, help="MoT 实际激活的专家数")
    args = parser.parse_args()

    device = args.device
    if device == "0":
        device = "cuda:0"

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 加载模型
    is_yaml = str(args.model).lower().endswith((".yaml", ".yml"))
    if is_yaml:
        model = build_model_from_yaml(args.model)
    else:
        model = build_model_from_pt(args.model)
    model = model.to(device).eval()

    # 运行路由分析
    if args.dry_run:
        rows = run_dry_run(model, device, args.imgsz)
        suffix = "dryrun"
    else:
        x = load_real_samples(args.data, args.num_samples, args.imgsz)
        if x is None:
            print("[diagnose_mot_routing] ⚠️ 真实数据加载失败，自动回退到 dry-run")
            rows = run_dry_run(model, device, args.imgsz)
            suffix = "dryrun"
        else:
            rows = run_real_data(model, device, x)
            suffix = f"real_n{args.num_samples}"

    if not rows:
        print("[diagnose_mot_routing] ⚠️ Hook 未采集到路由数据")
        return

    # 输出文件（带后缀避免覆盖）
    csv_path = out / f"routing_summary_{suffix}.csv"
    json_path = out / f"recommendations_{suffix}.json"
    png_path = out / f"routing_heatmap_{suffix}.png"

    write_csv(rows, csv_path)
    summary = aggregate_by_expert(rows)
    recs = write_recommendations(summary, json_path)

    if args.plot:
        title = f"MoT Routing Heatmap ({suffix})"
        plot_heatmap(rows, png_path, title=title)

    # 控制台摘要
    print("\n=== 平均激活率（跨 MoT 层） ===")
    for e in EXPERT_NAMES:
        s = summary[e]
        print(f"  {e:25s}: top1={s['top1_mean']:.3f}  "
              f"top{args.top_k}={s['topk_mean']:.3f}  "
              f"weight={s['weight_mean']:.3f}")
    H = summary["__avg_routing_entropy__"]["top1_mean"]
    print(f"  {'__routing_entropy__':25s}: H_norm={H:.3f} "
          f"({'CONFIDENT' if H < 0.3 else 'BALANCED' if H < 0.7 else 'UNCERTAIN'})")

    print("\n=== 场景化推荐 ===")
    for r in recs:
        print(f"  - {r}")


if __name__ == "__main__":
    main()
=======
#!/usr/bin/env python3
"""Diagnose MoT expert routing and plot expert activation heatmaps.

The script can analyze a trained ``.pt`` checkpoint, a YAML model config, or a
directory of scene-specific images. If no image directory is provided it falls
back to synthetic scene probes so the hook and plotting pipeline can be smoke
tested without a dataset.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

EXPERT_NAMES = ("LocalConvTransformer", "WindowTransformer", "DeformableTransformer")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_torch_device(device: str) -> str:
    if not device:
        return "cpu"
    if device.isdigit():
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return device


def load_model(model_path: Path, device: str, nc: int) -> torch.nn.Module:
    # Keep analysis-only helpers importable without loading optional YOLO/SAM backends.
    if model_path.suffix.lower() in {".pt", ".pth"}:
        from ultralytics import YOLO

        model = YOLO(str(model_path)).model
    else:
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel(str(model_path), ch=3, nc=nc, verbose=False)
    model.to(torch.device(device)).eval()
    return model


def register_router_hooks(model: torch.nn.Module, records: list[dict[str, str]], current_scene: dict[str, str]):
    from ultralytics.nn.modules.mot import MoTBlock

    handles = []

    def make_hook(layer_name: str):
        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            weights = output[0].detach().float().cpu()
            if weights.ndim != 4:
                return
            top1 = weights.argmax(dim=1)
            image_ids = current_scene.get("image_ids", [])
            batch = weights.shape[0]
            for batch_idx in range(batch):
                token_count = int(top1[batch_idx].numel())
                counts = torch.bincount(top1[batch_idx].reshape(-1), minlength=len(EXPERT_NAMES)).float()
                weight_mean = weights[batch_idx].mean(dim=(1, 2))
                image_id = image_ids[batch_idx] if batch_idx < len(image_ids) else f"sample_{batch_idx}"
                for expert_id, expert_name in enumerate(EXPERT_NAMES):
                    records.append(
                        {
                            "scene": current_scene.get("name", "unknown"),
                            "image_id": str(image_id),
                            "layer": layer_name,
                            "expert_id": str(expert_id),
                            "expert": expert_name,
                            "top1_tokens": str(int(counts[expert_id].item())),
                            "total_tokens": str(token_count),
                            "top1_share": f"{(counts[expert_id].item() / max(token_count, 1)):.6f}",
                            "mean_weight": f"{float(weight_mean[expert_id].item()):.6f}",
                        }
                    )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, MoTBlock):
            handles.append(module.router.register_forward_hook(make_hook(name)))
    return handles


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_image_tensor(path: Path, imgsz: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((imgsz, imgsz))
    data = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255.0)
    return data


def image_batches(image_dir: Path, imgsz: int, batch: int, max_images: int, scene_name: str):
    if not image_dir.exists():
        raise FileNotFoundError(
            f"image directory not found: {image_dir}. "
            "Create scene folders first, e.g. "
            "`python scripts/prepare_mot_routing_scenes.py --dataset datasets/VisDrone --split val`."
        )
    scene_dirs = [p for p in sorted(image_dir.iterdir()) if p.is_dir()]
    if not scene_dirs:
        scene_dirs = [image_dir]

    for scene_dir in scene_dirs:
        scene = scene_dir.name if scene_dir != image_dir else scene_name
        paths = [p for p in sorted(scene_dir.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
        if max_images > 0:
            paths = paths[:max_images]
        tensors: list[torch.Tensor] = []
        image_ids: list[str] = []
        for path in paths:
            tensors.append(load_image_tensor(path, imgsz))
            image_ids.append(str(path.relative_to(image_dir)))
            if len(tensors) == batch:
                yield scene, torch.stack(tensors, dim=0), image_ids
                tensors = []
                image_ids = []
        if tensors:
            yield scene, torch.stack(tensors, dim=0), image_ids


def synthetic_scenes(imgsz: int, batch: int):
    generator = torch.Generator().manual_seed(0)
    scenes: list[tuple[str, torch.Tensor]] = []

    sparse = torch.rand(batch, 3, imgsz, imgsz, generator=generator) * 0.04
    for i in range(6):
        y = 8 + i * max(2, imgsz // 10)
        x = 10 + i * max(2, imgsz // 12)
        sparse[:, :, y : y + 4, x : x + 4] += 0.8
    scenes.append(("sparse_small", sparse.clamp(0, 1), [f"sparse_small_{i}" for i in range(batch)]))

    dense = torch.rand(batch, 3, imgsz, imgsz, generator=generator) * 0.10
    step = max(6, imgsz // 12)
    for y in range(4, imgsz - 4, step):
        for x in range(4, imgsz - 4, step):
            dense[:, :, y : y + 3, x : x + 3] += 0.7
    scenes.append(("dense_small", dense.clamp(0, 1), [f"dense_small_{i}" for i in range(batch)]))

    large = torch.rand(batch, 3, imgsz, imgsz, generator=generator) * 0.05
    margin = max(8, imgsz // 5)
    large[:, :, margin : imgsz - margin, margin : imgsz - margin] += 0.65
    scenes.append(("large_regular", large.clamp(0, 1), [f"large_regular_{i}" for i in range(batch)]))

    irregular = torch.rand(batch, 3, imgsz, imgsz, generator=generator) * 0.05
    yy = torch.arange(imgsz).view(-1, 1)
    xx = torch.arange(imgsz).view(1, -1)
    center = imgsz // 2
    mask = ((yy - center).abs() * 1.4 + (xx - center).abs() * 0.8) < imgsz * 0.32
    irregular[:, :, mask] += 0.75
    irregular[:, :, center - 4 : center + 4, :] *= 0.25
    irregular[:, :, :, center - 4 : center + 4] *= 0.25
    scenes.append(("irregular_occluded", irregular.clamp(0, 1), [f"irregular_occluded_{i}" for i in range(batch)]))

    return scenes


class RouterWeightSummary:
    """Per-expert summary of router weights for a single MoT layer."""

    __slots__ = ("layer", "expert", "expert_id", "active_tokens", "total_tokens", "activation_ratio", "mean_weight")

    def __init__(self, layer: str, expert: str, expert_id: int, active_tokens: int, total_tokens: int, mean_weight: float):
        self.layer = layer
        self.expert = expert
        self.expert_id = expert_id
        self.active_tokens = active_tokens
        self.total_tokens = total_tokens
        self.activation_ratio = active_tokens / max(total_tokens, 1)
        self.mean_weight = mean_weight

    def __repr__(self) -> str:
        return (
            f"RouterWeightSummary(layer={self.layer!r}, expert={self.expert!r}, "
            f"active_tokens={self.active_tokens}/{self.total_tokens}, "
            f"activation_ratio={self.activation_ratio:.4f}, mean_weight={self.mean_weight:.6f})"
        )


def summarize_router_weights(layer_name: str, weights: torch.Tensor) -> list[RouterWeightSummary]:
    """Summarize per-expert activation from a router weight tensor.

    Args:
        layer_name: Name of the MoT layer for labelling.
        weights: Router weights of shape ``[B, E, H, W]`` (or ``[B, E, N]``).

    Returns:
        One :class:`RouterWeightSummary` per expert, in expert-index order.
    """
    if weights.ndim == 4:
        B, E, H, W = weights.shape
        flat = weights.permute(1, 0, 2, 3).reshape(E, -1)  # [E, B*H*W]
    elif weights.ndim == 3:
        B, E, N = weights.shape
        flat = weights.permute(1, 0, 2).reshape(E, -1)
    else:
        raise ValueError(f"Expected weights with 3 or 4 dims, got {weights.ndim}")

    total_tokens = flat.shape[1]
    rows: list[RouterWeightSummary] = []
    for e_idx in range(E):
        expert_w = flat[e_idx]
        active = int((expert_w > 0).sum().item())
        mean_w = float(expert_w.mean().item())
        expert_name = EXPERT_NAMES[e_idx] if e_idx < len(EXPERT_NAMES) else f"Expert{e_idx}"
        rows.append(RouterWeightSummary(layer_name, expert_name, e_idx, active, total_tokens, mean_w))
    return rows


def scenario_recommendations(rows: list[RouterWeightSummary]) -> list[str]:
    """Generate data-backed routing recommendations from weight summaries.

    Returns one recommendation string per expert. Each string includes a
    numeric metric so callers can verify the advice is grounded in data.
    """
    recs: list[str] = []
    for row in rows:
        if row.activation_ratio >= 0.75:
            recs.append(
                f"{row.expert}: dominant specialist (activation {row.activation_ratio:.1%}, "
                f"mean_weight {row.mean_weight:.4f}) — retain at full capacity"
            )
        elif row.activation_ratio >= 0.25:
            recs.append(
                f"{row.expert}: active specialist (activation {row.activation_ratio:.1%}, "
                f"mean_weight {row.mean_weight:.4f}) — consider rank reduction"
            )
        else:
            recs.append(
                f"{row.expert}: underutilised (activation {row.activation_ratio:.1%}, "
                f"mean_weight {row.mean_weight:.4f}) — candidate for pruning"
            )
    return recs


def aggregate_scenarios(records: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in records:
        key = (row["scene"], row["expert_id"], row["expert"])
        grouped.setdefault(key, []).append(row)

    rows = []
    for (scene, expert_id, expert), items in sorted(grouped.items()):
        top1 = [float(item["top1_share"]) for item in items]
        weights = [float(item["mean_weight"]) for item in items]
        rows.append(
            {
                "scene": scene,
                "expert_id": expert_id,
                "expert": expert,
                "layers": str(len(items)),
                "top1_share_mean": f"{sum(top1) / len(top1):.6f}",
                "mean_weight": f"{sum(weights) / len(weights):.6f}",
            }
        )
    return rows


def is_irregular_scene(scene: str) -> bool:
    scene = scene.lower()
    return "irregular" in scene or "occluded" in scene or "occlusion" in scene


def mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def bootstrap_diff_ci(
    irregular: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if irregular.size == 0 or baseline.size == 0 or samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        a = rng.choice(irregular, size=irregular.size, replace=True)
        b = rng.choice(baseline, size=baseline.size, replace=True)
        diffs[i] = a.mean() - b.mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def permutation_p_value(
    irregular: np.ndarray,
    baseline: np.ndarray,
    permutations: int,
    seed: int,
) -> float:
    """One-sided permutation p-value for H1: irregular mean > baseline mean."""
    if irregular.size == 0 or baseline.size == 0 or permutations <= 0:
        return float("nan")
    observed = irregular.mean() - baseline.mean()
    combined = np.concatenate([irregular, baseline])
    n_irregular = irregular.size
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        perm = rng.permutation(combined)
        diff = perm[:n_irregular].mean() - perm[n_irregular:].mean()
        if diff >= observed:
            hits += 1
    return float((hits + 1) / (permutations + 1))


def deformable_values(records: list[dict[str, str]], scene_filter, metric: str) -> np.ndarray:
    """Return image-level DeformableTransformer activation values averaged across MoT layers."""
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in records:
        scene = row.get("scene", "")
        if row.get("expert") != "DeformableTransformer" or not scene_filter(scene):
            continue
        image_id = row.get("image_id") or row.get("layer", "unknown")
        grouped.setdefault((scene, image_id), []).append(float(row[metric]))
    values = [sum(items) / len(items) for items in grouped.values() if items]
    return np.asarray(values, dtype=np.float64)


def deformable_activation_checks(
    records: list[dict[str, str]],
    permutations: int,
    bootstrap_samples: int,
    alpha: float,
    seed: int,
) -> list[dict[str, str]]:
    """Test whether DeformableTransformer activation is higher in irregular/occluded scenes."""
    scenes = sorted({row["scene"] for row in records})
    irregular = [scene for scene in scenes if is_irregular_scene(scene)]
    baselines = [scene for scene in scenes if not is_irregular_scene(scene)]
    comparisons: list[tuple[str, object]] = [(scene, scene) for scene in baselines]
    if baselines:
        comparisons.append(("non_irregular_pooled", tuple(baselines)))

    rows = []
    for metric in ("top1_share", "mean_weight"):
        irregular_values = deformable_values(records, is_irregular_scene, metric)
        for baseline_name, baseline in comparisons:
            if isinstance(baseline, tuple):
                baseline_values = deformable_values(records, lambda s, allowed=baseline: s in allowed, metric)
            else:
                baseline_values = deformable_values(records, lambda s, target=baseline: s == target, metric)
            diff = mean(irregular_values) - mean(baseline_values)
            ci_low, ci_high = bootstrap_diff_ci(
                irregular_values, baseline_values, samples=bootstrap_samples, seed=seed
            )
            p_value = permutation_p_value(irregular_values, baseline_values, permutations=permutations, seed=seed)
            significant = bool(
                np.isfinite(diff)
                and diff > 0
                and np.isfinite(ci_low)
                and ci_low > 0
                and np.isfinite(p_value)
                and p_value <= alpha
            )
            baseline_mean = mean(baseline_values)
            relative_lift = diff / baseline_mean if np.isfinite(baseline_mean) and baseline_mean != 0 else float("nan")
            rows.append(
                {
                    "metric": metric,
                    "irregular_scenes": "|".join(irregular),
                    "baseline": baseline_name,
            "irregular_image_samples": str(int(irregular_values.size)),
            "baseline_image_samples": str(int(baseline_values.size)),
                    "irregular_mean": f"{mean(irregular_values):.6f}",
                    "baseline_mean": f"{baseline_mean:.6f}",
                    "mean_diff": f"{diff:.6f}",
                    "relative_lift": f"{relative_lift:.6f}",
                    "bootstrap_ci95_low": f"{ci_low:.6f}",
                    "bootstrap_ci95_high": f"{ci_high:.6f}",
                    "permutation_p_value_one_sided": f"{p_value:.6f}",
                    "alpha": f"{alpha:.4f}",
                    "deformable_significantly_higher": str(significant),
                }
            )
    return rows


def plot_heatmap(records: list[dict[str, str]], out_path: Path, value: str) -> None:
    if not records:
        return
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(records)
    df[value] = df[value].astype(float)
    df["target"] = df["scene"] + " | " + df["layer"].astype(str)
    pivot = df.pivot_table(index="target", columns="expert", values=value, aggfunc="mean")
    pivot = pivot.reindex(columns=list(EXPERT_NAMES))

    sns.set_theme(style="white", context="paper", font="Arial")
    height = max(4.0, 0.32 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(8, height))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0, linewidths=0.4, ax=ax)
    ax.set_xlabel("Transformer expert")
    ax.set_ylabel("Scene and MoTBlock layer")
    ax.set_title(f"MoT expert activation heatmap ({value})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml")
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--scene-name", default="images")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dense/sparse/occluded scene probes.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/mot_ablation/routing")
    parser.add_argument("--heatmap-value", choices=("top1_share", "mean_weight"), default="top1_share")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = normalize_torch_device(args.device)
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    project = args.project if args.project.is_absolute() else ROOT / args.project

    model = load_model(model_path, device=device, nc=args.nc)
    records: list[dict[str, str]] = []
    current_scene = {"name": "unknown"}
    handles = register_router_hooks(model, records, current_scene)
    if not handles:
        raise SystemExit("No MoTBlock modules found; provide a MoT model/config.")

    if args.synthetic or not args.image_dir:
        source = synthetic_scenes(args.imgsz, args.batch)
    else:
        image_dir = args.image_dir if args.image_dir.is_absolute() else ROOT / args.image_dir
        source = image_batches(image_dir, args.imgsz, args.batch, args.max_images, args.scene_name)

    with torch.inference_mode():
        for scene, tensor, image_ids in source:
            current_scene["name"] = scene
            current_scene["image_ids"] = image_ids
            tensor = tensor.to(torch.device(device))
            _ = model(tensor)

    for handle in handles:
        handle.remove()

    detailed_csv = project / "mot_routing_detailed.csv"
    scenario_csv = project / "mot_routing_scenarios.csv"
    deformable_check_csv = project / "mot_deformable_activation_check.csv"
    heatmap_png = project / f"mot_expert_heatmap_{args.heatmap_value}.png"
    write_csv(detailed_csv, records)
    write_csv(scenario_csv, aggregate_scenarios(records))
    write_csv(
        deformable_check_csv,
        deformable_activation_checks(
            records,
            permutations=args.permutations,
            bootstrap_samples=args.bootstrap_samples,
            alpha=args.alpha,
            seed=0,
        ),
    )
    plot_heatmap(records, heatmap_png, args.heatmap_value)

    print(f"[routing] wrote {detailed_csv}")
    print(f"[routing] wrote {scenario_csv}")
    print(f"[routing] wrote {deformable_check_csv}")
    print(f"[routing] wrote {heatmap_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
>>>>>>> origin/main
