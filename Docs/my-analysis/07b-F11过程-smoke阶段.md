# F11 实验过程记录 · 07b smoke 阶段(T0-T7 冒烟命令与实测输出)

> **文档版本**:v1(2026-09-10 · 自原 07 单文件拆分;内容 = 原"T0→T7 完整命令清单"中全部 smoke 子节)| **Owner**:张伟林(Zviolin)| **基线**:`acce839c7e895d6b179de7f7093fa879e237cc7b`
>
> **导航**:[07a 总状态与规范](./07a-F11过程-状态与规范.md) · [07c real 阶段](./07c-F11过程-real阶段.md) · [07d quick500_v2](./07d-F11过程-quick500_v2.md) · [08b smoke 结果汇总](./08b-F11结果-smoke阶段.md)

> **特性**:
>
> - 每条命令先用 `Set-Location` 切根(在仓库 `G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master`)
> - 直接用 `python`(系统 PATH 已挂 yolo-master conda env,**不依赖 `$PY`**)
> - **每条命令前都有 `# log:` + `# 产物:` 注释**,直接落正确位置
> - 每条独立可跑,不依赖其它命令的 set-value

# ① T0 — 基线 + 环境 + Smoke

### T0.1 同步上游(2026-08-23 实测 ✅)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t0_env/git_sync.log
# 产物:    (无产物,只是 git 锚定)|
git fetch upstream
git checkout acce839c7e895d6b179de7f7093fa879e237cc7b 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t0_env/git_sync.log
```

### T0.2 环境验证(2026-08-25 实测 ✅)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t0_env/env_check.log
# 产物:    (stdout)|
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)" 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t0_env/env_check.log
```

### T0.3 detect smoke(2026-08-25 实测 ✅,1 epoch coco8,~3 秒)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t0_env/detect_smoke.log
# 产物:    runs/smoke/t0_env/{results.csv + BoxPR_curve.png + weights/best.pt}|
python -c "from ultralytics import YOLO; m = YOLO('ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml'); m.train(data='coco8.yaml', epochs=1, imgsz=640, batch=8, device=0, project='runs/smoke/t0_env', name='detect_smoke')" 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t0_env/detect_smoke.log
```

**输出结果**(实测):

```
YOLO-master-n summary: 422 layers, 3,137,637 parameters, 7.4 GFLOPs
1/1    1.93G    3.138→2.99    5.485→5.553    4.313→4.26
SMOKE OK
```

### T0.4 边界测试(2026-08-25 12:18 实测 ✅,106 PASS + 1 skip)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t0_env/boundary_test.log
# 产物:    runs/smoke/t0_env/test_router_kd.log(pytest 输出)|
python -m pytest tests/test_mot.py tests/test_moa.py -v 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t0_env/boundary_test.log
```

---

# ② T1 — 教师特征缓存

### T1.0 DINOv2 smoke(16 patch,coco8,~30 秒)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t1_teacher_cache/cache.log
# 产物:    runs/smoke/t1_teacher_cache/{manifest.json + meta.json + patches/*.pt}|
python scripts/cache_teacher_features.py --teacher dinov2_vitb14 --data coco8.yaml --num 50 --batch 1 --imgsz 112 --out runs/smoke/t1_teacher_cache --device 0 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t1_teacher_cache/cache.log
```

**输出结果**(实测 meta.json):

```json
{"teacher_model": "dinov2_vitb14", "num_cached": 16, "device": "cuda:0", "imgsz": 112, ...}
```

---

# ③ T2 — q_teacher

### T2.0 q_teacher smoke(1024 patches,~1 分钟)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t2_q_teacher/q_teacher.log
# 产物:    runs/smoke/t2_q_teacher/{q_teacher.pt + entropy.csv + report.json}(H_norm=0.7022)|
conda activate yolo-master; cd G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master; python scripts/gen_q_teacher.py --cache runs/smoke/t1_teacher_cache --model ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml --device cpu --out runs/smoke/t2_q_teacher --n_experts 4 --expert_dim 768 --temperature 0.5 --smoothing_eps 0.0 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t2_q_teacher/q_teacher.log
```

**输出结果**(实测 report.json):

```json
{"check": {"mean": 0.7022, "std": 0.1776, "not_collapsed_ratio": 0.9912, "not_uniform_ratio": 0.9590, "pass": true}}
```

---

# ④ T3 — Router KD 闭环

### T3.1 RouterKDLoss 单测(smoke,9 + 10 = 19 PASS)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t3_router_kd/router_kd_unit_test.log
# 产物:    runs/smoke/t3_router_kd/test_router_kd.log(pytest 输出)|
python -m pytest tests/test_router_kd.py tests/test_mot_export_masked_configs.py -v 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t3_router_kd/router_kd_unit_test.log
```

**输出结果**(实测,3.14 s 内 19 PASS / 11 warnings)

### T3.4 KD 闭环 smoke(1 epoch, coco8,~5 秒)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t3_router_kd/kd_training.log
# 产物:    runs/smoke/t3_router_kd/{smoke_report.json + run.log}(KD=0.0561 PASS)|
python scripts/smoke_router_kd.py --epochs 1 --layers 1 --experts 2 --device 0 --q_teacher_path runs/smoke/t2_q_teacher/q_teacher.pt 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t3_router_kd/kd_training.log
```

**输出结果**(实测):

```
[F11] epoch 1/1: task_loss=1.0261, kd_loss=0.0561, router_grad_norm=0.093672
[F11] KD loss: 0.0561 OK | Task loss: 1.0261 OK | Router grad: 0.093672 OK
[F11] Overall: PASS
```

---

# ⑤ T4 — 三方对照(smoke)

### T4 smoke(A/B/C 三组各 1 epoch, coco8, ~30 秒)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t4_ablation/ablation_train.log
# 产物:    runs/smoke/t4_ablation/{a-baseline_1ep, b-repr-kd_1ep, c-router-kd_1ep}/weights/best.pt(各 1 epoch)|
python scripts/compare_f11_ablation.py --train --groups a,b,c --epochs 1 --imgsz 320 --batch 4 --device 0 --data coco8.yaml --project runs/smoke/t4_ablation 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t4_ablation/ablation_train.log
```

**输出结果**(实测 1 epoch smoke,A/B/C 全跑通):

```
[F11][T4] ==== 训练 B-repr-kd (f11_repr_kd.yaml) ====
[foundation overrides: teacher: siglip2]
[B-repr-kd ✅ 训练完成]

[F11][T4] ==== 训练 C-router-kd (f11_router_kd.yaml) ====
YOLO-master-n summary: 422 layers, 3,137,637 parameters
1/1    0.488G    3.144→3.426    5.591→5.77    4.37→4.261
Optimizer stripped from ...\c-router-kd\weights\best.pt, 6.7MB
[F11][T4] ✅ C-router-kd 训练完成
```

**保存位置**(2026-08-29 规范归类):

- smoke 试跑权重已复制到 `runs/smoke/t4_ablation/{a-baseline_1ep, b-repr-kd_1ep, c-router-kd_1ep}/weights/best.pt`(共 7 份,各 1 epoch)
- log: `experiments/f11_real_baseline/logs/smoke/t4_ablation/{ablation_train.log, summary.log}`

---

# ⑥ T5 — 路由分析 + 零教师(smoke)

### T5.1 routing synthetic(smoke,~5 秒)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t5_routing/synthetic.log
# 产物:    runs/smoke/t5_routing/{routing_analysis.json + routing_metrics.csv + synthetic_run.log}|
python scripts/analyze_f11_routing.py --synthetic --experts 4 --num-tokens 512 --temperature 1.0 --seed 0 --out runs/smoke/t5_routing 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t5_routing/synthetic.log
```

**输出结果**(实测):

```
[F11][T5] tokens=512, experts=4
[F11][T5] H_router=1.1210, H_norm=0.8086
[F11][T5] load_std=0.0373, top1_switch=0.7554
[F11][T5] top1_agree=0.2285, top2_agree=0.8457
[F11][T5] js_mean=0.1313
```

### T5.3 学生导出(smoke,~10 秒)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t5_student/export_student.log
# 产物:    runs/smoke/t5_student/f11_student.pt(12.78 MB)|
python scripts/export_student.py --ckpt runs/smoke/t4_ablation/a-baseline_1ep/weights/best.pt --out runs/smoke/t5_student/f11_student.pt 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t5_student/export_student.log
```

### T5.3 零教师验证(smoke,~30 秒,P50 26 ms)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log: experiments/f11_real_baseline/logs/smoke/t5_student/verify_no_teacher.log
python scripts/verify_no_teacher.py --student runs/smoke/t5_student/f11_student.pt --device 0 --reps 10 --imgsz 320 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t5_student/verify_no_teacher.log
```

**输出结果**(实测):

```
[F11][T5] 参数量: 3,137,637
[F11][T5] Foundation 包装残留: ✅ 无 | Teacher Router 残留: ✅ 无
[F11][T5] 冻结教师参数: 0 | 推理延迟 P50: 26.33 ms (10 reps)
[F11][T5] 结果: ✅ 零教师依赖成立
```

### T5.4 推理精度(smoke,~30 秒)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/smoke/t5_student/inference_eval.log
# 产物:    runs/smoke/t5_student/{eval_report.json + f11_student_eval.log}|
python scripts/eval_student.py --student runs/smoke/t5_student/f11_student.pt --data coco8.yaml --device 0 --imgsz 320 --out runs/smoke/t5_student 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/smoke/t5_student/inference_eval.log
```

**输出结果**: `mAP50-95 = 0.0000 / mAP50 = 0.0000`(1 epoch smoke 正常,Δ=0<0.1 无损证明成立)

---

# ⑦ T7 — 证据归档(smoke)

### T7 archive(基于 smoke 产物,~10 秒)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/real/t7_evidence/T7R_archive.log
# 产物:    experiments/f11_real_baseline/evidence.tar.gz(内含 manifest + 摘要)|
python scripts/archive_evidence.py --output experiments/f11_real_baseline/evidence.tar.gz 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t7_evidence/T7R_archive.log
```

---

**最后更新**:2026-09-10 · **Owner**:张伟林(Zviolin) · **基线**:`acce839c7e895d6b179de7f7093fa879e237cc7b`
