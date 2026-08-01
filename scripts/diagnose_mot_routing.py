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
