# F11 实验过程记录 · 07c real 阶段(T0R-T7R 命令与实测)

> **文档版本**:v2(2026-09-10 结项定稿)| **Owner**:张伟林(Zviolin)| **基线**:`acce839c7e895d6b179de7f7093fa879e237cc7b`
>
> **导航**:[07a 总状态与规范](./07a-F11过程-状态与规范.md) · [07b smoke 阶段](./07b-F11过程-smoke阶段.md) · [07d quick500_v2](./07d-F11过程-quick500_v2.md) · [08c real 结果汇总](./08c-F11结果-real阶段.md) · [08d T6R 与 quick500](./08d-F11结果-T6R与quick500.md)

## ⏸ 全量 C 组重跑命令(已暂停,保留供结项后恢复)

> 结项口径为 quick500_v2 同预算全链(见 07d);全量 C 组重跑需 ~3 天,结项后如需补跑直接执行下述命令。
> 参数:`batch=8 / workers=8`(8GB 实测稳态,与 A 组同预算);启动后前 2 epoch 确认 log 出现 `"foundation_router_distill": true` 和 `"foundation_teacher": "siglip2"`;中断续跑在命令末尾加 `--resume`。

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"
conda activate yolo-master
$env:HF_HUB_OFFLINE = "1"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"

python scripts/compare_f11_ablation.py --train --groups c --name c-router-kd-v2 --router-temperature 0.5 --epochs 100 --imgsz 640 --batch 8 --workers 8 --device 0 --data "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml" --project runs/real/t4_ablation/real_100ep 2>&1 | Tee-Object -FilePath "experiments\f11_real_baseline\logs\real\t4_ablation\T4R_c_v2_$TS.log"
```

---

# ① T0R — VisDrone 数据准备

### T0R.1 VisDrone 下载(2026-08-26 实测 ✅,7 万张,~3 小时)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/real/t0_env/visdrone_download.log
# 产物:    G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone\images\(train 6471 + val 548)|
python scripts/download_visdrone.py --root "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone" 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t0_env/visdrone_download.log
```

---

# ② T1R — 教师特征缓存

### T1R 真基线(200 patch,VisDrone,~30 分钟)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/real/t1_teacher_cache/cache_real.log
# 产物:    experiments/f11_real_baseline/T1_teacher_cache/{manifest.json + meta.json + patches/*.pt}|
python scripts/cache_teacher_features.py --teacher dinov2_vitb14 --data "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml" --num 200 --batch 1 --imgsz 224 --out experiments/f11_real_baseline/T1_teacher_cache --device 0 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t1_teacher_cache/cache_real.log
```

---

# ③ T2R — 真 q_teacher

### T2R 真基线(51200 patches,~5 分钟,VisDrone + use_real_protos)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/real/t2_q_teacher/q_teacher_real.log
# 产物:    experiments/f11_real_baseline/T2_q_teacher/{q_teacher.pt + entropy.csv + report.json}(H_norm=0.8385)|
python scripts/gen_q_teacher.py --use_real_protos --cache experiments/f11_real_baseline/T1_teacher_cache --model ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml --device cuda:0 --out experiments/f11_real_baseline/T2_q_teacher --n_experts 4 --expert_dim 768 --temperature 0.5 --smoothing_eps 0.0 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t2_q_teacher/q_teacher_real.log
```

---

# ④ T3R — Router KD 闭环真基线

### T3R 真基线(100 epoch,~2 小时,两次各跑一次)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/real/t3_router_kd/kd_real.log
# 产物:    runs/real/t3_router_kd/{smoke_report_real.json + train.log}(KD=0.0141 PASS, 100 epoch)|
python scripts/smoke_router_kd.py --epochs 100 --layers 1 --experts 2 --device 0 --q_teacher_path experiments/f11_real_baseline/T2_q_teacher/q_teacher.pt 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t3_router_kd/kd_real.log
```

**输出结果**(实测,KD 收敛轨迹):

```
epoch 1/100: task_loss=1.0267, kd_loss=0.0230, router_grad_norm=0.070798
epoch 100/100: task_loss=0.7195, kd_loss=0.0141, router_grad_norm=0.029582
[F11] Overall: PASS
```

---

# ⑤ T4R — 三方对照真基线(A/B 全量)

### T4R-A 真基线(A 组 100 epoch, ~8.7 h, 纯本地)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/real/t4_ablation/T4R_a_resume_*.log
# 产物:    runs/real/t4_ablation/real_100ep/a-baseline/weights/best.pt
python scripts/compare_f11_ablation.py --train --groups a --epochs 100 --imgsz 640 --batch 8 --device 0 --data "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml" --project runs/real/t4_ablation/real_100ep 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t4_ablation/T4R_a.log
```

### T4R-B 真基线(B 组 100 epoch, siglip2 离线)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"
$env:HF_HUB_OFFLINE = "1"

# log:     experiments/f11_real_baseline/logs/real/t4_ablation/T4R_b_b12_w16_ram_*.log
# 产物:    runs/real/t4_ablation/real_100ep/b-repr-kd/weights/best.pt
python scripts/compare_f11_ablation.py --train --groups b --epochs 100 --imgsz 640 --batch 8 --device 0 --data "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml" --project runs/real/t4_ablation/real_100ep 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t4_ablation/T4R_b.log
```

### T4R summary(三组全完后, ~1 秒)

```powershell
# log:     experiments/f11_real_baseline/logs/real/t4_ablation/T4R_summary.log
# 产物:    runs/real/t4_ablation/real_100ep/f11_ablation_summary.json
python scripts/compare_f11_ablation.py --summary-only --project runs/real/t4_ablation/real_100ep 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t4_ablation/T4R_summary.log
```

### 全量收尾命令(全量 C 组恢复执行后使用)

```powershell
# summary
python scripts/compare_f11_ablation.py --summary-only --project runs/real/t4_ablation/real_100ep

# T5R 路由分析
python scripts/analyze_f11_routing.py `
  --student-a "runs/real/t4_ablation/real_100ep/a-baseline/weights/best.pt" `
  --student-c "runs/real/t4_ablation/real_100ep/c-router-kd-v2/weights/best.pt" `
  --data "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml" --device cuda:0 `
  --out runs/real/t5_routing/routing_analysis_v2.json

# T6R.5 机制分析
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
python scripts/analyze_f11_mechanism.py `
  --summary runs/real/t4_ablation/real_100ep/f11_ablation_summary.json `
  --routing runs/real/t5_routing/routing_analysis_v2.json `
  --out runs/real/t6_extension/mechanism/t6r5_mechanism_report_v2.json 2>&1 |
  Tee-Object -FilePath "experiments\f11_real_baseline\logs\real\t6_extension\T6R_5_mechanism_v2_$TS.log"

# T5R 学生导出三件套
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
python scripts/export_student.py --ckpt runs/real/t4_ablation/real_100ep/c-router-kd-v2/weights/best.pt --out runs/real/t5_student/f11_student_v2.pt 2>&1 |
  Tee-Object -FilePath "experiments\f11_real_baseline\logs\real\t5_student\T5R_export_v2_$TS.log"
python scripts/verify_no_teacher.py --student runs/real/t5_student/f11_student_v2.pt --device 0 --reps 10 --imgsz 320 2>&1 |
  Tee-Object -FilePath "experiments\f11_real_baseline\logs\real\t5_student\T5R_verify_v2_$TS.log"
python scripts/eval_student.py --student runs/real/t5_student/f11_student_v2.pt --data "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml" --device 0 --imgsz 320 --out runs/real/t5_student 2>&1 |
  Tee-Object -FilePath "experiments\f11_real_baseline\logs\real\t5_student\T5R_eval_v2_$TS.log"
```

---

# ⑥ T5R — 路由分析 + 零教师(真基线三件套)

> 当前有效版本为 quick500_v2 版(见 07d ③⑤);以下为 real 尺度命令模板(全量 C 组恢复后以 `c-router-kd-v2` 权重执行)。

### T5R 真路由分析(~2 分钟,基于 A vs C best.pt)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/real/t5_routing/T5R_routing.log
# 产物:    runs/real/t5_routing/{routing_analysis.json + routing_metrics.csv}|
python scripts/analyze_f11_routing.py --student-a runs/real/t4_ablation/real_100ep/a-baseline/weights/best.pt --student-c runs/real/t4_ablation/real_100ep/c-router-kd-v2/weights/best.pt --data "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml" --device cpu --batch 1 --imgsz 320 --out runs/real/t5_routing 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t5_routing/T5R_routing.log
```

### T5R 学生导出 + 零教师验证 + 推理精度

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# 1) 学生导出
python scripts/export_student.py --ckpt runs/real/t4_ablation/real_100ep/c-router-kd-v2/weights/best.pt --out runs/real/t5_student/f11_student.pt 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t5_student/T5R_export.log

# 2) 零教师验证
python scripts/verify_no_teacher.py --student runs/real/t5_student/f11_student.pt --device 0 --reps 10 --imgsz 320 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t5_student/T5R_verify.log

# 3) 推理精度
python scripts/eval_student.py --student runs/real/t5_student/f11_student.pt --data "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml" --device 0 --imgsz 320 --out runs/real/t5_student 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t5_student/T5R_eval.log
```

---

# ⑦ T6 — 扩展(子方向状态总览)

> **2026-09-06 脚本扩展**:`compare_f11_ablation.py` 增 `--layers` / `--teacher` 开关;T6R.5 新建 `scripts/analyze_f11_mechanism.py`(纯 CPU,无训练)。

| 子方向 | 状态 | 结论 |
|--------|------|------|
| **T6R.1 多层 KD** | 未产出有效数据 | `--layers` 无框架消费点(全代码库无 `args["layers"]` 读取,且注入发生在模型构建后),训练与普通 C 组逐位相同;正确路线为 `foundation_multiscale=true` + `foundation_target_levels`(F10 机制) |
| **T6R.2 MoT 路由** | 未产出有效数据 | `configs/f11_mot.yaml` 的 `model:` 字段指向 422 层普通模型(非 608 层真 MoT `yolo-master-mot-n.yaml`),训练实为普通 C 组;`--cfg` 机制本身有效 |
| **T6R.3 跨域** | ⏸ 协议落盘 | v2 协议:源域 VisDrone 零重训,目标域 UAVDT(`scripts/convert_uavdt_to_yolo.py` + `classes=[3,5,8]` 只评公共类);待 UAVDT 下载后启用 |
| **T6R.4 教师对比** | ✅ 收口 | quick 同预算(500×10ep):DINOv2 vs SigLIP2 均 KD 生效,mAP 噪声内,SigLIP2 开销减半 → **维持 siglip2**(详见 07d / 08d §8.6.5) |
| **T6R.5 机制分析** | ✅ **5/5 PASS(KD 有效)** | quick500_v2 版(详见 07d ④ / 08d §8.7) |

### T6R.3 跨域 v2 协议(待 UAVDT)

**步骤 1:下载 UAVDT(hyper.ai 种子,仅检测包)**:https://hyper.ai/cn/datasets/17869 → `UAV-benchmark-M.zip`(6.32 GB)→ 解压到 `G:\Codes\OpenSource\Rhino-bird\DATASETS\UAV-benchmark-M\`

**步骤 2:转换**(脚本已落盘 `scripts/convert_uavdt_to_yolo.py`):

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
python scripts/convert_uavdt_to_yolo.py --src "G:\Codes\OpenSource\Rhino-bird\DATASETS\UAV-benchmark-M" --stride 30 2>&1 `
    | Tee-Object -FilePath "experiments\f11_real_baseline\logs\real\t6_extension\T6R_3_convert_$TS.log"
# 产物: DATASETS\UAVDT\{images,labels}\val + UAVDT.yaml(names 与 VisDrone 10 类对齐)
```

**步骤 3:零训练跨域评估**(每组 ~10 分钟):

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
& "E:\Apps\Codes\Conda\envs\yolo-master\python.exe" -c "
from ultralytics import YOLO
for g in ['a-baseline', 'b-repr-kd', 'c-router-kd-v2']:
    m = YOLO(f'runs/real/t4_ablation/real_100ep/{g}/weights/best.pt')
    r = m.val(data='G:/Codes/OpenSource/Rhino-bird/DATASETS/UAVDT.yaml', imgsz=640, batch=8,
              device=0, classes=[3, 5, 8],
              project='runs/real/t6_extension/cross_domain_v2', name=g, exist_ok=True)
    print(f'UAVDT-cross-domain {g}: mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f}')
" 2>&1 | Tee-Object -FilePath "experiments\f11_real_baseline\logs\real\t6_extension\T6R_3_cross_domain_v2_$TS.log"
```

**判据**:源域 VisDrone val 的 C−A 与目标域 UAVDT 的 C−A 对比 —— C 组退化幅度显著小于 A/B 组则 Router KD 跨域泛化增益成立;三组同幅退化则如实记录无增益。

### T6R.5 机制分析命令(quick500_v2 版,当前有效)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"
python scripts/analyze_f11_mechanism.py --summary runs/quick500_v2/t4_ablation/f11_ablation_summary.json --routing "runs/quick500_v2/t5_routing/routing_analysis_quick.json/routing_analysis.json" --out runs/quick500_v2/t6_extension/mechanism/t6r5_mechanism_quick.json
```

---

# ⑧ T7 — 证据归档

### T7 archive(基于 smoke + 真基线产物,~10 秒)

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"

# log:     experiments/f11_real_baseline/logs/real/t7_evidence/T7R_archive.log
# 产物:    experiments/f11_real_baseline/evidence.tar.gz(内含 manifest + 摘要)|
python scripts/archive_evidence.py --output experiments/f11_real_baseline/evidence.tar.gz 2>&1 | Tee-Object -FilePath experiments/f11_real_baseline/logs/real/t7_evidence/T7R_archive.log
```

---

# §A.1 P0+P1 真基线 终端实测输出片段

> 以下片段直接摘自 `experiments/f11_real_baseline/logs/real/` 下各任务 log 文件,与 08c §3.1.1 配对表数值一致。

### §A.1.1 T0R VisDrone 下载(`t0_env/visdrone_downl*.log`)
```
VisDrone2019-DET-val.zip: 100%|██████████| 81.6M/81.6M [00:10<00:00, 7.66MB/s]
Linking val images: 100%|██████████| 548/548 [00:00<00:00, 1533.24it/s]
Converting val labels: 100%|██████████| 548/548 [00:00<00:00, 1637.42it/s]
Wrote G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml
```

### §A.1.2 T1R DINOv2 教师缓存(`t1_teacher_cache/cach*.log`)
```
[F11] 收集到 7019 张候选图片(最多用 1000)
[F11] 加载 DINO 教师: dinov2_vitb14 @ cuda:0
[F11] 已缓存 200 / 1000 张
[F11] [OK] 缓存完成: 200 张
```

### §A.1.3 T2R q_teacher 构造(`t2_q_teacher/q_teache*.log`)
```
[F11] 加载 200 个 patch 特征文件
[F11] 使用结构版 prototypes: 4 experts × 768 dim
[F11] q_teacher 构造完成
[F11]   - patches: 51200, experts: 4
[F11]   - H_norm mean: 0.8385(目标 ∈ (0.3, 0.8),判读线已调为 [0.3, 1.0])
[F11]   - 非坍缩比例: 100.0% / 非均值比例: 89.9%
```

### §A.1.4 T3R KD 闭环(epoch 100/100 · `t3_router_kd/kd_real_*.log`)
```
[F11] epoch 100/100: task_loss=0.7195, kd_loss=0.0141, router_grad_norm=0.029582
[F11] KD loss: 0.0141 OK
[F11] Task loss: 0.7195 OK
[F11] Router grad: 0.029582 OK
[F11] Overall: PASS
```
> 第 2 次跑:Task=0.7150 / KD=0.0150 / Router grad=0.026710 / PASS。

### §A.1.5 T4R a-baseline 100 epoch val(`t4_ablation/T4R_a_res*.log`)
```
YOLO-master-n summary: 422 layers, 3,107,395 parameters, 3,107,379 gradients, 7.3 GFLOPs
                   all        548      38759      0.427      0.333      0.305      0.171
Speed: 0.1ms preprocess, 7.4ms inference, 0.0ms loss, 2.0ms postprocess per image
Results saved to ...\runs\real\t4_ablation\real_100ep\a-baseline
```

### §A.1.6 T4R summary 三方对照(`t4_ablation/T4R_summa*.log`)
```
run                    mAP50-95    mAP50   box_loss   cls_loss
a-baseline                0.170    0.305     1.4085     0.9613
b-repr-kd                 0.169    0.302     1.3965     0.9466
```

### §A.1.7 T5R 学生推理精度(`t5_student/T5R_eval_2*.log`)
```
YOLO-master-n summary: 309 layers, 3,096,555 parameters, 0 gradients, 7.1 GFLOPs
                   all        548      38759      0.301      0.192      0.155     0.0793
[F11][T5] mAP50-95 = 0.0793
[F11][T5] mAP50    = 0.1555
```

### §A.1.8 T5R 零教师验证(`t5_student/T5R_verify*.log`)
```
[F11][T5] Foundation 包装残留: ✓ 无
[F11][T5] Teacher Router 残留:  ✓ 无
[F11][T5] 冻结教师参数: 0
[F11][T5] 推理延迟 P50: 21.99 ms (10 reps)
[F11][T5] 结果: ✓ 零教师依赖成立
```

### §A.1.9 T5R 学生导出(`t5_student/T5R_export*.log`)
```
[F11][T5] 纯学生模型已导出: ...\runs\real\t5_student\f11_student.pt
[F11][T5] 参数量:  3,107,395
```

### §A.1.10 T5R 路由分析(`t5_routing/T5R_routin*.log`,A 侧)
```
[F11][T5] ==== A-baseline ====
[F11][T5] tokens=12, experts=2
[F11][T5] H_router=0.6397, H_norm=0.9229
[F11][T5] load_std=0.2357, top1_switch=0.6364
```

> **P0+P1 终端核对结论**:T0R/T1R/T2R/T3R/T4R/T5R 各 log 与 08c §3.1.1 + §7 数据对齐。

---

**最后更新**:2026-09-10 · **Owner**:张伟林(Zviolin) · **基线**:`acce839c7e895d6b179de7f7093fa879e237cc7b`
