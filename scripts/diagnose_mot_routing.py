"""MoT 路由可解释性诊断脚本（绕过 ultralytics 顶层导入）。

按 Docs/Disscusion/08-任务3-路由可解释性分析.md 的接口与输出格式：
  - --model  : 模型 YAML 或 .pt 权重
  - --dry-run: 合成随机输入（无需数据集）
  - --device : cpu / cuda
  - --output : 输出目录（默认 experiments_zviolin/runs/mot_routing）

输出：
  - routing_summary.csv   : 18 行（6 层 × 3 专家）
  - recommendations.json  : 3 条场景化推荐
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

# 解析 YOLO-Master 路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 直接加载 YAML（无需 ultralytics 顶层），避免 YOLO-Master Fork 的导入冲突
try:
    import yaml  # PyYAML
except ImportError:
    yaml = None

from ultralytics.nn.modules import MoTBlock  # noqa: E402  # 单独导入 MoTBlock 类即可
from ultralytics.nn.tasks import yaml_model_load, load_checkpoint  # noqa: E402

# MoTBlock 三专家的中文名（与文档一致）
EXPERT_NAMES = ["LocalConvTransformer", "WindowTransformer", "DeformableTransformer"]


def build_model_from_yaml(cfg_path: str):
    """从 YAML 构建模型（绕过顶层 ultralytics.YOLO）。"""
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = ROOT / cfg_file
    d = yaml_model_load(str(cfg_file))
    model, _save = parse_model(d, ch=3, verbose=False)
    return model


def build_model_from_pt(pt_path: str):
    """从训练后 .pt 权重构建模型。"""
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    # 兼容多种 ckpt 格式
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


def summarize_router_weights(name: str, weights: torch.Tensor):
    """对单个 MoTBlock 的 router 输出做激活率统计。"""
    rows = []
    top1 = weights.argmax(dim=1)  # [B, H, W]
    total_tokens = top1.numel()
    mean_weight = weights.mean(dim=(0, 2, 3))  # [E]

    for e_idx, expert_name in enumerate(EXPERT_NAMES):
        active = (top1 == e_idx).sum().item()
        rows.append({
            "layer": name,
            "expert": expert_name,
            "active_tokens": int(active),
            "activation_ratio": active / total_tokens,
            "mean_weight": float(mean_weight[e_idx]),
        })
    return rows


def diagnose(model_path: str, dry_run: bool, device: str, output_dir: Path,
             imgsz: int = 320):
    """执行路由诊断并写入 CSV / JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载模型
    is_yaml = str(model_path).lower().endswith((".yaml", ".yml"))
    if is_yaml:
        model = build_model_from_yaml(model_path)
    else:
        model = build_model_from_pt(model_path)

    model = model.to(device)
    model.eval()

    # 2) 注册 forward hook
    rows = []
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, _output):
            with torch.no_grad():
                if hasattr(module, "router") and inputs:
                    router_out = module.router(inputs[0])
                    weights = router_out[0] if isinstance(router_out, tuple) else router_out
                    rows.extend(summarize_router_weights(name, weights))
        return hook

    hooked_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MoTBlock":
            hooks.append(module.register_forward_hook(make_hook(name)))
            hooked_count += 1

    print(f"[diagnose_mot_routing] MoTBlock hooked: {hooked_count}")
    if hooked_count == 0:
        print("[diagnose_mot_routing] ⚠️ 未发现 MoTBlock，请检查 YAML 是否包含 MoT 头")
        return

    # 3) 合成随机输入（dry-run）
    with torch.no_grad():
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        _ = model(x)

    # 4) 清理 hook
    for h in hooks:
        h.remove()

    if not rows:
        print("[diagnose_mot_routing] ⚠️ Hook 未采集到路由数据")
        return

    # 5) 写入 CSV
    csv_path = output_dir / "routing_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "expert", "active_tokens",
                                               "activation_ratio", "mean_weight"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[diagnose_mot_routing] wrote {csv_path} ({len(rows)} rows)")

    # 6) 场景化推荐
    by_expert = {e: [] for e in EXPERT_NAMES}
    for r in rows:
        by_expert[r["expert"]].append(r["activation_ratio"])

    avg = {e: (sum(v) / len(v)) if v else 0.0 for e, v in by_expert.items()}

    recs = [
        (f"Dense or small-object scenes should inspect WindowTransformer first "
         f"when its activation ratio is high ({avg.get('WindowTransformer', 0):.3f})."),
        (f"Occluded or irregular-object scenes should inspect DeformableTransformer "
         f"when its activation ratio rises ({avg.get('DeformableTransformer', 0):.3f})."),
        (f"Latency-sensitive simple scenes can prefer LocalConvTransformer-heavy "
         f"routing ({avg.get('LocalConvTransformer', 0):.3f}) before enabling "
         f"deeper MoT/MoE hybrids."),
    ]

    rec_path = output_dir / "recommendations.json"
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    print(f"[diagnose_mot_routing] wrote {rec_path}")

    # 7) 控制台摘要
    print("\n=== 平均激活率（跨 MoT 层） ===")
    for e in EXPERT_NAMES:
        print(f"  {e:25s}: {avg.get(e, 0):.3f}")
    print("\n=== 场景化推荐 ===")
    for r in recs:
        print(f"  - {r}")


def main():
    parser = argparse.ArgumentParser(description="MoT 路由可解释性诊断")
    parser.add_argument("--model", required=True,
                        help="YAML 配置文件 或 .pt 训练后权重")
    parser.add_argument("--dry-run", action="store_true",
                        help="使用合成随机输入，无需数据集")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0", "0"],
                        help="推理设备（推荐 cpu，单次前向更快）")
    parser.add_argument("--output", default="experiments_zviolin/runs/mot_routing",
                        help="输出目录")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="合成输入尺寸（默认 320）")
    args = parser.parse_args()

    device = args.device
    if device in ("0",):
        device = "cuda:0"

    out = Path(args.output)
    diagnose(args.model, args.dry_run, device, out, args.imgsz)


if __name__ == "__main__":
    main()