"""测量所有 4 个混合架构的 Params 和 FLOPs"""
import torch
from ultralytics import YOLO

cfgs = {
    "v08": r"ultralytics\cfg\models\master\v0_8\det\yolo-master-n.yaml",
    "v08_mot": r"ultralytics\cfg\models\master\v0_8\det\yolo-master-mot-n.yaml",
    "v08_moe_mot": r"ultralytics\cfg\models\master\v0_8\det\yolo-master-moe-mot-hybrid-n.yaml",
    "v08_moe_mot_lite": r"ultralytics\cfg\models\master\v0_8\det\yolo-master-moe-mot-lite-n.yaml",
    "v08_moe_mot_aggr": r"ultralytics\cfg\models\master\v0_8\det\yolo-master-moe-mot-aggressive-n.yaml",
    "v08_moe_mot_moa_scene": r"ultralytics\cfg\models\master\v0_8\det\yolo-master-moe-mot-moa-scene-n.yaml",
}
print(f'{"key":30s} {"Params (M)":>12s} {"GFLOPs":>10s}')
for k, cfg in cfgs.items():
    m = YOLO(cfg)
    info = m.info()
    # m.info() returns (layers, params, gradients, GFLOPs)
    params_m = info[1] / 1e6 if info[1] > 1e6 else info[1]
    gflops = info[3] if len(info) > 3 else 0
    print(f"{k:30s} {params_m:>12.3f} {gflops:>10.3f}")
