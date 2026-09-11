# F11 结果汇总 · 08d T6R 扩展与 quick500_v2(T6R 现状 + 教师对比 + 机制分析 + 日志)

> **文档版本**:v2(2026-09-10 结项定稿)| **Owner**:张伟林(Zviolin)| **基线**:`acce839c7e895d6b179de7f7093fa879e237cc7b`
>
> **导航**:[08a 总览与结论](./08a-F11结果-总览与结论.md) · [08b smoke 阶段](./08b-F11结果-smoke阶段.md) · [08c real 阶段](./08c-F11结果-real阶段.md) · [07d quick500_v2 过程](./07d-F11过程-quick500_v2.md)

# §8 T6R 扩展:现状一览 + 教师对比 + 机制分析

## 8.6 T6R 扩展实验

### 8.6.1 T6R 现状一览(2026-09-10 结项定稿)

| 子方向 | 状态 | 结论 |
|--------|------|------|
| **T6R.1** 多层 KD | ❌ 无有效数据 | `--layers` 无框架消费点(全代码库无 `args["layers"]` 读取)且注入发生在模型构建后;正确路线应为 `foundation_multiscale` + `target_levels`(F10 机制) |
| **T6R.2** MoT 路由 | ❌ 无有效数据 | `configs/f11_mot.yaml` 的 `model:` 字段指向 422 层普通模型(非 608 层真 MoT `yolo-master-mot-n.yaml`);`--cfg` 机制有效但配置内容非 MoT |
| **T6R.3** 跨域 | ⏸ 协议落盘 | v2 设计:源域 VisDrone 零重训 + 目标域 UAVDT(`classes=[3,5,8]` 公共类);待 UAVDT 下载 |
| **T6R.4** 教师对比 | ✅ 收口 | quick 同预算(500×10ep):DINOv2 vs SigLIP2 均 KD 生效、mAP 噪声内,SigLIP2 开销减半 → 维持 siglip2,见 §8.6.5 |
| **T6R.5** 机制分析 | ✅ **5/5 PASS(KD 有效)** | quick500_v2 版,见 §8.7 |

### 8.6.5 T6R.4 教师对比结果(DINOv2 vs SigLIP2,2026-09-10)

> **设计**:DINOv3 为 Meta gated 模型(镜像不可用),教师对比修正为 **DINOv2 vs SigLIP2**——`facebook/dinov2-base` 非 gated 公开权重(hf-mirror 可下 346MB,落盘 HF 缓存 + `DATASETS\foundation_teachers\dinov2-base\`);`DINOv2Teacher` 已接入框架(与 DINOv3 同构,支持本地权重注入)。

**配套代码修复**:

| # | 文件 | 修复 |
|---|---|---|
| 1 | `scripts/compare_f11_ablation.py` | `--teacher` 切换教师时 `foundation_model` 映射为目标教师默认 model_id(`TEACHER_DEFAULT_MODELS`) |
| 2 | `ultralytics/nn/foundation_distill_model.py` | 补 `DEFAULT_DINOV2_MODEL` + `DINOv2Teacher` import + dinov2 教师分支 |

**实测命令**(同 seed=0 同预算,唯一变量为教师;log: `experiments\f11_real_baseline\logs\quick_validate\t6\T6R_4_{dinov2,siglip2}_20260910_*.log`):

```powershell
python scripts/compare_f11_ablation.py --train --groups c --teacher dinov2 --name c-router-kd-dinov2 --router-temperature 0.5 --epochs 10 --imgsz 640 --batch 8 --workers 8 --device 0 --data "G:\Codes\OpenSource\Rhino-bird\DATASETS\VisDrone_500.yaml" --project runs/quick500_v2/t6_extension/teacher_compare 2>&1 | Tee-Object -FilePath "experiments\f11_real_baseline\logs\quick_validate\t6\T6R_4_dinov2_$TS.log"
# siglip2 侧:--teacher siglip2 --name c-router-kd-siglip2,其余参数相同
```

**KD 生效铁证 + 结果对比**(results.csv):

| 指标 | DINOv2 | SigLIP2 | 判读 |
|---|---|---|---|
| foundation_router_modules | 3(全程) | 3(全程) | 两教师都挂满 3 层 KD |
| router_loss ep1→ep10 | 0.1144→0.0914 | 0.0932→**0.0668** | 都收敛;SigLIP2 更快更低 |
| router_kl 非零 | ✓ | ✓ | 软目标真在传递 |
| teacher_entropy | ≈2.0015 | ≈1.9977 | SigLIP2 软目标更紧 |
| mAP50-95 @ep10(last) | 0.00323 | 0.00329 | 差 0.006 个百分点,噪声区内 |
| mAP50 @ep10(last) | 0.01108 | 0.01200 | 同上 |
| 每 epoch 耗时 | ~478s(ep9→10 实测 477.8s) | ~215s(ep9→10 实测 215.7s) | SigLIP2 教师开销减半 |
| lr 曲线 | 同 seed=0 逐 epoch 一致 | 同 | 严格同预算,唯一变量为教师 |

**结论**:

1. **KD 生效性与教师选择解耦**:两个基础模型教师(DINOv2 / SigLIP2)都能驱动 Router KD 生效;
2. quick 同预算下 mAP 无可分辨差异 → 教师对比不以 quick mAP 为敏感轴;
3. SigLIP2 软目标更紧且开销减半 → **全量 C 组维持 siglip2 教师**;
4. **T6R.4 收口**:quick 同预算对比 + 机制分析(§8.7)满足任务书"比较教师,**或**分析负结果机制"的双分支,全量教师对比不再追加。

**产物**:`runs/quick500_v2/t6_extension/teacher_compare/{c-router-kd-dinov2,c-router-kd-siglip2}/`。

## 8.7 T6R.5 机制分析报告(quick500_v2 版 · **✅ 当前有效结论**)

**输入**:`runs/quick500_v2/t4_ablation/f11_ablation_summary.json`(T4R 三组 summary)+ `runs/quick500_v2/t5_routing/routing_analysis_quick.json/routing_analysis.json`(T5R 路由分析)

**命令**:
```powershell
python scripts/analyze_f11_mechanism.py --summary runs/quick500_v2/t4_ablation/f11_ablation_summary.json --routing "runs/quick500_v2/t5_routing/routing_analysis_quick.json/routing_analysis.json" --out runs/quick500_v2/t6_extension/mechanism/t6r5_mechanism_quick.json
```

**输出报告**(2026-09-10 已落盘,5 维度全过):

| # | 维度 | 阈值 | 实测 | 判定 |
|---|------|------|------|------|
| 1 | 软目标质量(JS 距离) | < 0.05 | C 组 js_mean=**0.0146** (js_std=0.0248) | ✅ PASS |
| 2 | 同质化(load_std) | C vs A Δ < 0.05 | Δ=**-0.2357**(A=0.7071 → C=0.4714,优于 A) | ✅ PASS |
| 3 | 路由切换(switch_rate) | C vs A Δ > 0.05 | Δ=**+0.3636**(A=0.0000 → C=0.3636,路由"活了") | ✅ PASS |
| 4 | 坍塌检查(单专家 < 90%) | max_load < 0.9 | C 组 max=**0.8333**(负载 83.3% / 16.7%) | ✅ PASS(未坍塌) |
| 5 | mAP 提升 | C ≥ A - 0.5% | Δ=**-0.0001**(A=0.04866 → C=0.04858,last 口径,阈值内) | ✅ PASS |

**Verdict**:**5/5 维度通过 — KD 有效**。C 组(KD 修复后重训)路由器负载均衡改善 33%、路由切换率从完全坍塌的 0 恢复到 0.3636、软目标 JS=0.0146 真在传递。

**注意**:维度 5 的 mAP Δ=-0.0001 在 val 400 张噪声(±0.6-0.8pp)内,quick500 的 mAP 不作显著性结论;**KD 有效性的证据轴是机制 4 维(1-4)而非 quick mAP**。

**产物**:`runs/quick500_v2/t6_extension/mechanism/t6r5_mechanism_quick.json`(verdict: total=5, pass=5, fail=0, conclusion="KD 有效")。

---

# §5.2.2 T6R 补跑 log — `experiments\f11_real_baseline\logs\real\t6_extension\`

| 子目录 | 文件 | 大小 | 状态 | 关键产出 |
|------|------|------|------|----------|
| `t6_extension/` | `T6R_1_multilayer.log` | 3,835 KB | ❌ 无有效数据 | T6R.1 训练 log(`--layers` 无框架消费点)|
| `t6_extension/` | `T6R_2_mot.log` | 28 KB | ❌ 无有效数据 | T6R.2 训练 log(配置非真 MoT)|
| `t6_extension/` | `T6R_3_cross_domain.log` | 75 KB | — | T6R.3 v1 log |
| `t6_extension/` | `T6R_5_mechanism.log` | 1 KB | ✅ | `analyze_f11_mechanism.py` 脚本运行记录 |
| `t6_extension/` | `T6R_smoke_args_check.log` | 1 KB | ✅ | `--layers`/`--teacher` 参数校验 |

# §5.2.3 quick500_v2 阶段 log(17 份 — `experiments\f11_real_baseline\logs\quick_validate\`)

| 子目录 | 文件 | 大小 | 状态 | 关键产出 |
|--------|------|------|------|----------|
| (根) | `quick500_ac_20260909_0018/002229/002655.log` | 11.5 / 31.2 / 101.3 KB | ✅ | quick500 A/C 预检 |
| (根) | `T4R_main_20260909_011720.log` | 2,281.3 KB | ✅ | 一键脚本主链:T4R-A 完整 100ep(0.701h)|
| (根) | `T4R_bc_20260909_120337.log` | 23.3 KB | ✅ | `--groups b,c` 续跑启动 |
| (根) | `T4R_bc_20260909_121733.log` | 3,024.5 KB | ✅ | 续跑完成:B(6.05h)+ C(6.22h)完整 100ep |
| `t6/` | `T6R_1_20260910_005105.log` | 1,512.7 KB | — | T6R.1 100ep 训练 log(与普通 C 同,无有效增量)|
| `t6/` | `T6R_2_20260910_065025.log` | 1,512.7 KB | — | T6R.2 100ep 训练 log(实为 C 组配置)|
| `t6/` | `T6R_4_dinov3_124515.log` / `dinov2_124511.log` | 2.2 KB ×2 | — | `--teacher` 非法值快速失败 |
| `t6/` | `T6R_4_{dinov2,siglip2}_1329/1331*.log` | ~2.2-13.3 KB | — | HF 缓存补齐前的离线加载失败 |
| `t6/` | `T6R_4_dinov2_133647.log` | 16.1 KB | — | DINOv2 教师加载阶段失败(缓存补齐前)|
| `t6/` | `T6R_4_dinov2_141854.log` | 180.6 KB | ✅ | **DINOv2 教师 10ep quick 跑通**(router_modules=3) |
| `t6/` | `T6R_4_siglip2_133658.log` | 181.1 KB | ✅ | **SigLIP2 教师 10ep quick 跑通**(router_modules=3) |

> 失败 log 构成 T6R.4 代码修复的问题证据链(§8.6.5);有效 log 5 份(T4R 主链 + 续跑 + T6R.4 两组成功)。

---

**最后更新**:2026-09-10 · **Owner**:张伟林(Zviolin) · **基线**:`acce839c7e895d6b179de7f7093fa879e237cc7b`
