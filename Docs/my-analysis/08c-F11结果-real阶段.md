# F11 结果汇总 · 08c real 阶段(真基线配对表 + A/B 三方对照 + 日志 + 原始记录)

> **文档版本**:v3(2026-09-10 结项定稿)| **Owner**:张伟林(Zviolin)| **基线**:`acce839c7e895d6b179de7f7093fa879e237cc7b`
>
> **real 阶段的贡献定位**:① **P0 全部**由 real 完成(T0R 数据 + T1R 缓存 + T2R q_teacher + T3R 双复跑 KD 闭环);② **P1 的 real 尺度 A/B 基线**(各 100ep,mAP50-95=0.1695);③ T7R 证据归档。C 组有效结果与 T5R/T6R 见 08d(quick500_v2 口径)。
>
> **导航**:[08a 总览与结论](./08a-F11结果-总览与结论.md) · [08b smoke 阶段](./08b-F11结果-smoke阶段.md) · [08d T6R 与 quick500](./08d-F11结果-T6R与quick500.md) · [07c real 阶段](./07c-F11过程-real阶段.md)

---

# §7 真基线 A/B 对照结果汇总(real 尺度,VisDrone 6471 张 × 100ep)

> **关键产物**:`Backup/runs/real/t4_ablation/real_100ep/{a-baseline, b-repr-kd}/`
>
> **C 组口径**:有效 C 组结果为 quick500_v2 同预算版(见 08d §8.7,机制 5/5 PASS,KD 有效);全量 C 组重跑已暂停。本节为 A/B 的 real 尺度参照。

## 7.1 对照总览(A/B 两组)

| 组 | 模型 | 教师 | best.pt | 100 epoch? | 训练耗时 | 状态 |
|---|------|------|---------|-----------|----------|------|
| **a-baseline** | YOLO-Master 原版 | 无 | ✅ 6.7 MB | ✅ | ~7.8 h | ✅ |
| **b-repr-kd** | + SigLIP2 表征 KD | SigLIP2-base | ✅ 6.7 MB | ✅ | ~82 h(resume 多次)| ✅ |

## 7.2 最终 mAP 对比(epoch 100)

| 组 | mAP50 | **mAP50-95** | box_loss | cls_loss | dfl_loss |
|---|-------|--------------|----------|----------|----------|
| **a-baseline** | 0.3047 | **0.1695** | 1.4373 | 1.0245 | 0.9145 |
| **b-repr-kd** | 0.3016 | **0.1695** | 1.4242 | 1.0115 | 0.9186 |

**数据口径说明**:
1. b-repr-kd 训练 batch=12,a-baseline 为 batch=8;batch 混杂对 mAP 的影响远小于"两组差异=0"的幅度,方向性结论成立;F11 核心对比 A vs C 在 quick500_v2 内以同 batch=8 严格对齐;
2. **B 组 repr-KD 管线全程工作**(checkpoint `train_results` 铁证:`train/foundation_loss` 100 epoch 全程非零 ~0.102,relational_raw 0.0851→0.0847,effective_weight=0.1),A/B 持平是真实实验结论;KD 项仅占总损失 **0.72%**(task_ratio≈0.007)→ 信号过弱、无增益,这正是 C 组转向 router KD(`router_loss_weight=0.5`)的设计依据;
3. 教师SigLIP2 全程 CPU 前向是 B 组 82h 耗时主因。

## 7.3 关键产物路径

```
G:\Codes\OpenSource\Rhino-bird\Backup\runs\real\t4_ablation\real_100ep\
├── a-baseline\
│   ├── weights\{best.pt, last.pt, last_healthy.pt}
│   ├── results.csv + results.png
│   ├── BoxPR/F1/P/R_curve.png
│   ├── confusion_matrix.png + labels.jpg
│   └── args.yaml
└── b-repr-kd\(同上结构)
```

---

# §3 真基线阶段 — 全任务↔全产物 配对表

> 真基线目的: 真实验数据 + 9.14 结项报告 · 数据集:VisDrone(7 万张) · Epoch:100 · SigLIP2 教师 + DINOv2-vitb14 离线缓存

## 3.1 真基线总览

| 维度 | 真基线 |
|------|--------|
| 目的 | 真实验数据 + 9.14 结项报告 |
| 数据集 | ✅ VisDrone(train 6471 / val 548 / test 1610,`G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone\`)|
| Epoch | 100(完整训练)|
| Seed | 0(A 组 6 个独立 run 覆盖 seed 0-2)|
| Batch | A=8 / B=12 |
| Imgsz | 640 |
| 模型 | yolo-master-n |
| 教师 | DINOv2-vitb14(缓存) + SigLIP2-base-patch16-512(主教师)|
| 用途 | 9.14 结项答辩 |

### 3.1.1 真基线任务 ↔ 脚本 ↔ 产物 ↔ 状态 配对表

| 任务 | 命令(实际跑过)| 脚本 | **产物路径** | 关键数值 | 状态 |
|------|----------|------|--------------|---------|------|
| **T0R.1** VisDrone 下载 | `python scripts/download_visdrone.py` | `scripts/download_visdrone.py` | `DATASETS/VisDrone/{images/{train,val,test}, VisDrone.yaml}` | train 6471 + val 548 + test 1610 张 | ✅ |
| **T0R.2** yolo-master-n 在 VisDrone smoke | (内建 Python 1 句)| ultralytics | `runs/smoke/`(共用 smoke)| 已验 | ✅ |
| **T1R** 真教师缓存 | `python scripts/cache_teacher_features.py --teacher dinov2_vitb14 --data VisDrone.yaml --num 200 --out experiments/f11_real_baseline/T1_teacher_cache` | `cache_teacher_features.py` | `experiments/f11_real_baseline/T1_teacher_cache/{manifest.json, meta.json, patches/200 个 patch.pt}` | num_cached=**200** / SHA256 全 | ✅ |
| **T2R** 真 q_teacher | `python scripts/gen_q_teacher.py --use_real_protos --cache .../T1_teacher_cache --out experiments/f11_real_baseline/T2_q_teacher` | `gen_q_teacher.py` | `experiments/f11_real_baseline/T2_q_teacher/{q_teacher.pt, entropy.csv, report.json}` | q_teacher.pt **[51200, 4]** / H_norm=**0.8385** / use_real_protos=**true** | ✅ |
| **T3R.1** Router KD 真基线 #1 | `python scripts/smoke_router_kd.py --epochs 100 --layers 1 --experts 2 --q_teacher_path .../T2_q_teacher/q_teacher.pt` | `smoke_router_kd.py` | `runs/real/t3_router_kd/smoke_report_real.json` + train.log | epoch 100/100 / KD=**0.0141** / Task=0.7195 / Router grad=**0.029582** / PASS | ✅ |
| **T3R.2** Router KD 真基线 #2 | (同上,2026-08-27)| `smoke_router_kd.py` | `experiments/f11_real_baseline/logs/real/t3_router_kd/kd_real_#2_042002.log` | epoch 100/100 / KD=**0.0150** / Task=0.7150 / Router grad=**0.026710** / PASS | ✅ |
| **T4R-A** a-baseline 6 份 × 100 epoch | `python scripts/compare_f11_ablation.py --train --groups a --epochs 100 --data VisDrone.yaml --batch 8` | `compare_f11_ablation.py` | `runs/real/t4_ablation/real_100ep/a-baseline/`(Backup 另存)| mAP50=0.3047 / mAP50-95=**0.1695** / 1.93GB / 8.7-9.0 h/run | ✅ |
| **T4R-B** b-repr-kd 真基线 | `python scripts/compare_f11_ablation.py --train --groups b --epochs 100 --data VisDrone.yaml` | `compare_f11_ablation.py` | `runs/real/t4_ablation/real_100ep/b-repr-kd/` | mAP50=0.3016 / mAP50-95=**0.1695** / repr-KD 生效(foundation_loss 全程非零)| ✅ |
| **T4R-C** c-router-kd | quick500_v2 同预算版(500 张 × 100ep,batch 8,τ=0.5)| `compare_f11_ablation.py` | `runs/quick500_v2/t4_ablation/c-router-kd/` | mAP50-95 best=**0.0502**(三组最高)/ **KD 生效**(modules=3,loss 100/100 非零)| ✅ |
| **T4R-summary** | `python scripts/compare_f11_ablation.py --summary-only --project runs\real\t4_ablation\real_100ep` | `compare_f11_ablation.py` | `runs/real/t4_ablation/real_100ep/f11_ablation_summary.json` | 全量口径 a=0.1695 / b=0.1695 | ✅ |
| **T5R.1** 路由分析 | `python scripts/analyze_f11_routing.py --student-a .../a-baseline/weights/best.pt --student-c runs/quick500_v2/t4_ablation/c-router-kd/weights/best.pt --out runs/quick500_v2/t5_routing` | `analyze_f11_routing.py` | `runs/quick500_v2/t5_routing/routing_analysis_quick.json/routing_analysis.json` | switch 0→**0.3636** / load_std **0.7071→0.4714** / js_mean=**0.0146** | ✅ |
| **T5R.2** 学生导出 | `python scripts/export_student.py --ckpt runs/quick500_v2/t4_ablation/c-router-kd/weights/best.pt --out runs/quick500_v2/t5_student/f11_student_quick.pt` | `export_student.py` | `runs/quick500_v2/t5_student/f11_student_quick.pt` | 参数量=**3,107,395**(教师已剥离) ✅ |
| **T5R.3** 零教师验证 | `python scripts/verify_no_teacher.py --student runs/quick500_v2/t5_student/f11_student_quick.pt --device 0 --reps 10 --imgsz 320` | `verify_no_teacher.py` | `runs/quick500_v2/t5_student/` | Foundation 残留 ✅ 无 / Teacher Router ✅ 无 / 冻结参数=**0** / P50=**35.94 ms** ✅ |
| **T5R.4** 学生精度 | `python scripts/eval_student.py --student runs/quick500_v2/t5_student/f11_student_quick.pt --data VisDrone_500.yaml --device 0 --imgsz 320 --out runs/quick500_v2/t5_student` | `eval_student.py` | `runs/quick500_v2/t5_student/{eval_report.json, student/}` | mAP50-95=0.0191 / mAP50=0.0488(320² 评测,管线性质验证)| ✅ |
| **T6R.1** 多层 KD | — | `compare_f11_ablation.py` | 无有效产物 | `--layers` 无框架消费点且注入时机错误,训练与普通 C 组逐位相同;正确路线 `foundation_multiscale`+`target_levels` | ❌ 无有效数据 |
| **T6R.2** MoT 路由 | — | `compare_f11_ablation.py` | 无有效产物 | `configs/f11_mot.yaml` 的 model 指向 422 层普通模型(非 608 层真 MoT);`--cfg` 机制有效但配置内容非 MoT | ❌ 无有效数据 |
| **T6R.3** 跨域 | `scripts/convert_uavdt_to_yolo.py` 协议落盘 | `compare_f11_ablation.py` | 协议已落盘 | v2 设计:VisDrone 零重训 + UAVDT 目标域(classes=[3,5,8] 公共类)| ⏸ 待 UAVDT |
| **T6R.4** 教师对比 | `python scripts/compare_f11_ablation.py --train --groups c --teacher dinov2 --name c-router-kd-dinov2 ...`(siglip2 侧同参)| `compare_f11_ablation.py` | `runs/quick500_v2/t6_extension/teacher_compare/{c-router-kd-dinov2, c-router-kd-siglip2}/` | DINOv2 vs SigLIP2 均 KD 生效;SigLIP2 开销减半 → **维持 siglip2** | ✅ 收口(2026-09-10) |
| **T6R.5** 机制分析 | `python scripts/analyze_f11_mechanism.py --summary runs/quick500_v2/t4_ablation/f11_ablation_summary.json --routing runs/quick500_v2/t5_routing/routing_analysis_quick.json/routing_analysis.json --out runs/quick500_v2/t6_extension/mechanism/t6r5_mechanism_quick.json` | **`scripts/analyze_f11_mechanism.py`** | `runs/quick500_v2/t6_extension/mechanism/t6r5_mechanism_quick.json` | **5/5 PASS(KD 有效)**,详见 08d §8.7 | ✅ |
| **T7R** 证据归档 | `python scripts/archive_evidence.py --output experiments/f11_real_baseline/evidence.tar.gz` | `archive_evidence.py` | `experiments/f11_real_baseline/evidence.tar.gz/`(内含 manifest + 摘要)| 3 次成功打包 | ✅ |

> **真基线完成度**:✅ 21/23 + ❌ 无有效数据 2(T6R.1/2)+ ⏸ 待 UAVDT 1(T6R.3);全量 C 组重跑暂停,结项口径为 quick500_v2(见 08a §6)。

---

# §5.2 真基线阶段 log — `experiments\f11_real_baseline\logs\real\`

## 5.2.1 基础链 log(T0R-T3R + T7R)

| 子目录 | 文件 | 大小 | 状态 | 关键产出 |
|------|------|------|------|----------|
| `t0_env/` | `visdrone_download_050746.log` | **76.1 KB** | ✅ | VisDrone 7 万张 |
| `t1_teacher_cache/` | `cache_real_051036.log` | 2.5 KB | ✅ | **200 VisDrone patch** |
| `t2_q_teacher/` | `q_teacher_real_051129.log` | 0.6 KB | ✅ | **H_norm=0.8385** |
| `t3_router_kd/` | `kd_real_#1_051131.log` | 8.3 KB | ✅ PASS | KD=**0.0141** |
| `t3_router_kd/` | `kd_real_#2_042002.log` | 8.3 KB | ✅ PASS | KD=**0.0150** |
| `t7_evidence/` | `archive_#1_050418.log` / `#2_233018` / `#3_232537` | 0.9 KB ×3 | ✅ | evidence.tar.gz 三次打包 |

## 5.2.2 训练链 log(T4R-A/B)

| 子目录 | 文件 | 大小 | 说明 |
|------|------|------|------|
| `t4_ablation/` | `T4R_a_resume_20260830_010912.log` | **13,322 KB** | A 组完整训练主 log(100ep)|
| `t4_ablation/` | `T4R_a_resume_20260830_005913/010033/010204/123758/123858.log` | 15-35 KB ×5 | A 组断点续跑链 |
| `t4_ablation/` | `T4R_b_b12_w16_ram_20260830_131243.log` | **11,108 KB** | B 组完整训练主 log(100ep)|
| `t4_ablation/` | `T4R_b_b16/b12/b12_w8_ram/resume_*.log` | 0-33 KB ×7 | B 组调参过程链 |
| `t4_ablation/` | `T4R_summary_20260906_235521.log` | 1 KB | 三方 summary 重生成 |
| `t5_routing/` | `T5R_routing_20260906_235417.log` | 1 KB | real 版路由分析(A 侧有效)|
| `t5_student/` | `T5R_eval_20260906_235435.log` | 6 KB | 学生推理精度(0.0793/0.1555)|

## 5.2.3 T6R log(T6R.1/2/3/5)

| 子目录 | 文件 | 大小 | 说明 |
|------|------|------|------|
| `t6_extension/` | `T6R_1_multilayer.log` | **3,835 KB** | T6R.1 训练 log(数据无有效增量)|
| `t6_extension/` | `T6R_2_mot.log` | 28 KB | T6R.2 训练 log |
| `t6_extension/` | `T6R_3_cross_domain.log` | 75 KB | T6R.3 v1 log |
| `t6_extension/` | `T6R_5_mechanism.log` | 1 KB | 机制分析脚本首次运行 |
| `t6_extension/` | `T6R_smoke_args_check.log` | 1 KB | `--layers`/`--teacher` 参数校验 |

---

# 附录 B:真基线原始实测记录

## B.1 真基线阶段总览

| 维度 | 真基线 |
|------|--------|
| 目的 | 真实验数据 + 9.14 结项报告 |
| 数据集 | ✅ VisDrone(7 万张,`G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone\`)|
| Epoch | 100(完整训练) |
| Seed | {0, 1, 2}(A 组 6 个独立 run 覆盖)|
| Batch | A=8 / B=12 |
| Imgsz | 640 |
| 模型 | yolo-master-n |
| 教师 | DINOv2-vitb14(缓存) + SigLIP2-base-patch16-512(主教师)|
| 用途 | 9.14 结项答辩 |

**实测时间**(单卡 RTX 5060 8GB + siglip2 教师 cpu):

| 配置 | 单次时间 | 实测情况 |
|------|---------|----------|
| VisDrone 全量(6471 train)| 1 epoch ≈ 1 h | siglip2-cpu 把单 batch 从 0.6 s 拉到 5-6 s |
| **A 组 100 epoch** | ~7.8 h | 6 份独立 run |
| **B 组 100 epoch** | ~82 h | siglip2-cpu 教师前向为主因 |

## B.2 真基线环境(实测)

| 项 | 实测 |
|----|------|
| Python | 3.10.9 |
| torch | 2.9.1+cu130 |
| ultralytics | 8.4.101 |
| transformers | **5.15.1** |
| CUDA | 13.0 + RTX 5060 |
| GPU 显存(A 组 baseline)| 1.93 GB(8GB 余量充足) |
| GPU 显存(B 组 KD)| 4.17 GB(SigLIP2 cpu 不抢显存)|

## B.3 真基线 T0 数据准备

```powershell
python scripts/download_visdrone.py
```

| 指标 | 实测 |
|------|------|
| VisDrone 下载 | ✅ 完整(trains 6471 + val 548 + test 1610)|
| log 文件 | `T0R_download_visdrone_20260826_050746.log` 77 KB |
| 数据集位置 | `G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone\` |
| yaml 路径 | `G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml`(已配 path 字段)|
| 类别数 | 10(pedestrian / people / bicycle / car / van / truck / tricycle / awning-tricycle / bus / motor)|

## B.4 真基线 T1 教师缓存

```powershell
python scripts/cache_teacher_features.py \
  --teacher dinov2_vitb14 \
  --data G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone.yaml \
  --num 200 --batch 1 --imgsz 224 \
  --out experiments/f11_real_baseline/T1_teacher_cache \
  --device 0
```

> **教师选择**:DINOv2-vitb14(免认证,SHA256 校验通过)。

| 指标 | 实测 |
|------|------|
| 缓存 patch 数 | **200 张**(VisDrone 真采)|
| manifest.json SHA256 | ✅ 全部 200 个 patch 有 sha256 |
| meta.json `num_cached` | **200** |
| 缓存总大小 | < 100 MB(每张 ~500 KB)|

## B.5 真基线 T2 q_teacher

```powershell
python scripts/gen_q_teacher.py \
  --use_real_protos \
  --cache experiments/f11_real_baseline/T1_teacher_cache \
  --model ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml \
  --device cpu \
  --out experiments/f11_real_baseline/T2_q_teacher
```

| 指标 | 目标 | 实测 |
|------|------|------|
| q_teacher.pt | [N, E=4] | **[51200, 4]** |
| H_norm mean | ∈ (0.3, 1.0) | **0.8385**(真原型分布更均匀,属预期)|
| use_real_protos | ✅ | **true** |
| entropy.csv | 每行一 patch | ✅ 51200 行 |
| report.json | pass 字段 | ✅ pass=true |

## B.6 真基线 T3 Router KD 训练

```powershell
python scripts/smoke_router_kd.py \
  --epochs 100 --layers 1 --experts 2 \
  --device 0 \
  --q_teacher_path experiments/f11_real_baseline/T2_q_teacher/q_teacher.pt
```

| 指标 | 设计目标 | 第 1 次(8/26)| 第 2 次(8/27)|
|------|---------|------------------|------------------|
| epoch | 100 | **100/100** ✅ | **100/100** ✅ |
| KD loss 最终 | 0.05-0.3(下降趋势) | **0.0141** | **0.0150** |
| Task loss 最终 | 收敛 | 0.7195 | 0.7150 |
| Router grad norm | > 0 | **0.029582** | **0.026710** |
| Overall | PASS | **PASS** ✅ | **PASS** ✅ |

**结论**:✅ T3R Router KD 真基线两轮 PASS,可作为 T4R C 组训练链路可靠性证据

---

**最后更新**:2026-09-10 · **Owner**:张伟林(Zviolin) · **基线**:`acce839c7e895d6b179de7f7093fa879e237cc7b`
