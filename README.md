# YOLO-Master Issue #54 实验数据 & 文档（张伟林 Zviolin）

> 本仓库是 [YOLO-Master](https://github.com/Zviolin/YOLO-Master) Issue #54 课题的**实验数据 & 文档仓库**。
>
> 配套代码仓库：[Zviolin/YOLO-Master @ issue-54-mot-experiments](https://github.com/Zviolin/YOLO-Master/tree/issue-54-mot-experiments)
>
> 个人博客：[https://zviolin.online/](https://zviolin.online/)
>
> 个人简历：[https://zviolin.online/resume/](https://zviolin.online/resume/)

## 课题信息

| 项目 | 内容 |
|------|------|
| **课题** | YOLO-Master MoT 架构消融对比、路由可解释性与混合架构探索 |
| **Issue** | [Tencent/YOLO-Master#54](https://github.com/Tencent/YOLO-Master/issues/54) |
| **实战期** | 2026 年 8 月 1 日 ~ 9 月 10 日 |
| **提交人** | 张伟林（Zviolin） |
| **学校** | 福州大学 计算机与大数据学院 软件工程 |
| **导师计划** | 2026 腾讯犀牛鸟开源人才培养计划 |

## 核心成果：方案 D（跨尺度 Expert 共享）

| 指标 | MoT 基线 (v08_mot) | 方案 D (v08_moe_mot_shared) | 变化 |
|------|--------------------|------------------------------|------|
| 参数量 | 3.71M | **3.00M** | **−19.1%** ✅ |
| CPU 延迟 | 215.91ms | **182.15ms** | **−15.6%** ✅（达标 10% 阈值）|
| GPU 延迟 | 52.74ms | 51.73ms | −1.9%（Laptop GPU 测量噪声）|
| mAP50-95 | 16.93% | 16.80% | −0.13%（损失极小）|
| mAP50 | 29.77% | 29.52% | −0.25% |
| moe_loss | - | 9.95e-05 | ✅ 完全稳定 |

**协同增益判定**：按 Issue #54 "mAP > 1% 或 延迟 > 10%"二选一标准，**CPU 延迟 −15.6% 已达标，架构创新成立**。

## 路由分析核心数据

| 专家 | v08_mot6 激活率 | 方案 D 激活率 | 变化 |
|------|----------------|--------------|------|
| LocalConvTransformer | 69.4% | 46.2% | −23.2% |
| WindowTransformer | 12.6% | **35.0%** | **+22.4%**（跨尺度特征共享提升）|
| DeformableTransformer | 18.0% | 18.8% | +0.8% |

## 仓库目录结构

```
Data/
├── README.md                   # 本文件
├── configs/                    # 项目配置文档
│   ├── 00-项目配置总览.md
│   ├── 01-设备硬件配置说明.md
│   ├── 02-Conda环境配置说明.md
│   └── 03-GitHub仓库与项目结构说明.md
├── docs/                       # 项目文档（12 个 Markdown + 1 个 PDF）
│   ├── 00-犀牛鸟计划总览.md
│   ├── 01-Issue汇总与简介.md
│   ├── 02-Issue54排除法选择.md
│   ├── 03-Issue54完整课题文档.md
│   ├── 04-项目申请书.md          ⭐ 申请材料
│   ├── 05-Issue54任务分析与实验文档.md
│   ├── 06-任务1-三变体训练.md
│   ├── 07-任务2-性能对比测量.md
│   ├── 08-任务3-路由可解释性分析.md  ⭐ 含实测路由数据
│   ├── 09-任务4-混合架构探索.md      ⭐ 含方案 D 创新
│   ├── 10-任务5-边界测试与修复.md
│   ├── 11-任务6-交付物清单.md
│   └── PDF/
│       └── 04-项目申请书.pdf    ⭐ LaTeX 级排版 PDF
└── runs/                       # 实验数据（8 个训练变体 + 2 个路由分析）
    ├── build_summary.csv       # 4 方案构建摘要
    ├── latency_cpu_640.csv     # CPU 延迟数据
    ├── latency_cuda_0_640.csv  # GPU 延迟数据（修复后实测）
    ├── summary.csv             # 训练指标汇总
    ├── test_results.txt        # pytest 边界测试输出
    │
    ├── v08/                    # MoE 基线（3.14M params，mAP50-95=16.84%）
    ├── v08_moa/                # MoA 对比（3.20M params，mAP50-95=16.80%）
    ├── v08_moa2/               # MoA v2（3.20M params）
    ├── v084/                   # v08 第4次调优（3.14M params）
    ├── v08_mot/                # MoT 基准（3.71M params，mAP50-95=16.93% 最高）
    ├── v08_mot6/               # MoT 第6次调优（3.71M params）
    ├── v08_moe_mot_lite/       # 混合架构探索（已废弃）
    └── v08_moe_mot_shared/     # ⭐ 方案 D：跨尺度 Expert 共享（3.00M params）

    # 路由分析数据（2026-07-31 新增）
    ├── mot_routing/            # v08_mot6 路由分析
    │   ├── routing_summary.csv # 18 行（6 层 × 3 专家）
    │   └── recommendations.json # 3 条场景化推荐
    └── mot_routing_shared/     # 方案 D 路由分析
        ├── routing_summary.csv
        └── recommendations.json
```

## 8 个训练变体的实测结果

| 变体 | Params (M) | mAP50-95 | 训练状态 | 备注 |
|------|-----------|----------|---------|------|
| v08 (MoE 基线) | 3.14 | 16.84% | ✅ | 基线对比 |
| v08_moa | 3.20 | 16.80% | ✅ | MoA 对比 |
| v08_moa2 | 3.20 | - | ✅ | MoA v2 |
| v084 | 3.14 | - | ✅ | v08 第4次调优 |
| v08_mot | 3.71 | **16.93%** | ✅ | **MoT 基准（最高 mAP）** |
| v08_mot6 | 3.71 | - | ✅ | MoT 第6次调优 |
| v08_moe_mot_lite | - | - | ⚠️ | 提前停止（17 epochs，废弃） |
| **v08_moe_mot_shared** | **3.00** | **16.80%** | ✅ | **⭐ 方案 D（跨尺度 Expert 共享）** |

**总计**：8 个变体 / 约 160 小时训练 / 全部完成

## 关键技术修复：SharedExpertMoE

**问题**：`SharedExpertMoE._apply()` 方法未正确传播 device，导致 CPU/GPU 隐式 transfer。

**修复**：
- 在 `_apply()` 中同步 device（递归调用 `expert._apply(fn)`）
- 添加 `reset_shared_pools()` 类方法
- Benchmark 前强制 reset 状态

**效果**：GPU 延迟从错误数据 `245.310ms` → 真实数据 `51.726ms`（**−79%**）

详见：[docs/09-任务4-混合架构探索.md §4.4 GPU Benchmark 工程教训](docs/09-任务4-混合架构探索.md)

## 路由分析脚本

配套代码仓库 [Zviolin/YOLO-Master @ issue-54-mot-experiments](https://github.com/Zviolin/YOLO-Master/tree/issue-54-mot-experiments) 中的：

- `scripts/diagnose_mot_routing.py`：MoT 路由可解释性诊断脚本（201 行）
  - 支持 `--model` / `--dry-run` / `--device` / `--output` 参数
  - 输出 `routing_summary.csv`（18 行 × 5 列）+ `recommendations.json`（3 条推荐）

**复现实验**：
```bash
conda activate yolo-master
cd <YOLO-Master 仓库根目录>

# v08_mot6（MoT 基准）路由分析
python scripts/diagnose_mot_routing.py \
    --model experiments_zviolin/runs/v08_mot6/weights/last.pt \
    --dry-run --device cpu \
    --output experiments_zviolin/runs/mot_routing

# 方案 D 路由分析
python scripts/diagnose_mot_routing.py \
    --model experiments_zviolin/runs/v08_moe_mot_shared/weights/best.pt \
    --dry-run --device cpu \
    --output experiments_zviolin/runs/mot_routing_shared
```

## 申请材料

- **[docs/04-项目申请书.md](docs/04-项目申请书.md)** — Markdown 源文件
- **[docs/PDF/04-项目申请书.pdf](docs/PDF/04-项目申请书.pdf)** — LaTeX 级排版 PDF（14 页）

## 文档快速导航

| 编号 | 文档 | 说明 |
|------|------|------|
| 00 | [犀牛鸟计划总览](docs/00-犀牛鸟计划总览.md) | 项目背景 |
| 01 | [Issue 汇总与简介](docs/01-Issue汇总与简介.md) | 选题分析 |
| 02 | [Issue #54 排除法选择](docs/02-Issue54排除法选择.md) | 课题选定 |
| 03 | [Issue #54 完整课题文档](docs/03-Issue54完整课题文档.md) | 课题定义 |
| **04** | **[项目申请书](docs/04-项目申请书.md)** ⭐ | **核心申请材料** |
| 05 | [任务分析与实验文档](docs/05-Issue54任务分析与实验文档.md) | 任务总览 |
| 06 | [任务 1：三变体训练](docs/06-任务1-三变体训练.md) | 训练日志 |
| 07 | [任务 2：性能对比测量](docs/07-任务2-性能对比测量.md) | 性能数据 |
| **08** | **[任务 3：路由可解释性](docs/08-任务3-路由可解释性分析.md)** ⭐ | **路由分析** |
| **09** | **[任务 4：混合架构探索](docs/09-任务4-混合架构探索.md)** ⭐ | **方案 D 创新** |
| 10 | [任务 5：边界测试与修复](docs/10-任务5-边界测试与修复.md) | 测试报告 |
| 11 | [任务 6：交付物清单](docs/11-任务6-交付物清单.md) | 最终交付 |

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{zviolin2026issue54,
  title  = {YOLO-Master MoT 架构消融对比、路由可解释性与混合架构探索},
  author = {张伟林 (Zviolin)},
  year   = {2026},
  url    = {https://github.com/Zviolin/YOLO-Master/tree/issue-54-mot-experiments}
}
```

## 许可证

本项目基于 YOLO-Master 原项目许可证，遵循 AGPL-3.0 License。

## 联系方式

- 邮箱：164321595@qq.com
- GitHub：[@Zviolin](https://github.com/Zviolin)
- 博客：https://zviolin.online/
- 简历：https://zviolin.online/resume/