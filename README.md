# YOLO-Master ES-MoE 自适应推理优化实验数据 & 文档（张伟林 Zviolin）

> 本仓库是 **2026 腾讯犀牛鸟实战项目一（ES-MoE 自适应推理优化）** 的实验数据 & 文档仓库。
>
> 配套代码分支：[Zviolin/YOLO-Master @ practise-1-esmoe-adaptive-inference](https://github.com/Zviolin/YOLO-Master/tree/practise-1-esmoe-adaptive-inference)
>
> 个人博客：[https://zviolin.online/](https://zviolin.online/)

## 课题信息

| 项目 | 内容 |
|------|------|
| **课题** | 基于 YOLO-Master ES-MoE 的自适应推理优化（Top-K 稀疏 + 动态路由 + 跨尺度共享）|
| **关联 Issue** | [Tencent/YOLO-Master#54](https://github.com/Tencent/YOLO-Master/issues/54) |
| **实战期** | 2026 年 8 月 1 日 ~ 9 月 10 日 |
| **提交人** | 张伟林（Zviolin）|
| **学校** | 福州大学 计算机与大数据学院 软件工程 |
| **导师计划** | 2026 腾讯犀牛鸟开源人才培养计划 |

## 核心成果：Top-K 稀疏化 + 动态路由

| 指标 | 结果 |
|------|------|
| **Top-K 权衡** | K=2 精度最优 **17.27%**（相对 MoE 基线 v084 **+0.43%**）|
| **稀疏化收益（GPU）** | K=1 **-23.7%** / K=3 **-10.5%** / Shared **-13.9%**（同权重 Hard vs Dense）|
| **稀疏化收益（CPU）** | K=1 **-18.9%** / K=2 **-13.3%** / K=3 **-15.7%** / Shared **-7.0%** |
| **动态 Top-K** | 按图像复杂度自适应，实测 avg_top_k=**1.66**（34% 触发 K=1）|
| **SharedExpertESMoE** | 跨尺度专家共享，参数 **-3.1%**、专家对象 **-25%** |

## 仓库目录结构

```
practices/DATA/
├── README.md                   # 本文件
├── configs/                    # 项目配置文档
│   ├── 00-项目配置总览.md
│   ├── 01-设备硬件配置说明.md
│   ├── 02-Conda环境配置说明.md
│   └── 03-GitHub仓库与项目结构说明.md
├── discussion/                 # GitHub Discussion 技术文章 + PR 文档
│   ├── 13-GitHub-Discussion-ESMoE.md   ⭐ Discussion 技术分享文章
│   ├── ESMoE-PR-Title-Body.md          ⭐ PR Title + Body 模板
│   └── figures/                        # Discussion 配图（4 张）
│       ├── routing_heatmap_k2.png
│       ├── routing_entropy_compare.png
│       ├── sparse_gain_gpu_cpu.png
│       └── dynamic_topk_complexity.png
├── docs/                       # 实战项目任务文档（9 篇）
│   ├── 00-实战项目原文.md
│   ├── 01-项目分析说明.md
│   ├── 02-项目结合分析.md
│   ├── 03-实战项目任务分析与实验文档.md
│   ├── 04-任务1-ES-MoE训练.md
│   ├── 05-任务2-专家利用率分析与负载均衡调优.md
│   ├── 06-任务3-性能指标验证.md
│   ├── 07-任务4-推理效率优化.md
│   └── 08-任务5-可视化分析与交付物.md
└── experiments_zviolin/        # 实验数据（4 模型权重 + runs 完整结果）
    ├── README.md               # 实验数据索引
    ├── weights/                # 6 个 best.pt 权重（约 6MB/个，随仓库上传）
    └── runs/                   # 训练日志 / 路由诊断 / benchmark / 动态 Top-K
```

## 4 个模型的实测结果（VisDrone 2019, RTX 5060 Laptop 8GB, 100 epoch）

| 变体 | top_k | Params (M) | mAP50-95 | mAP50 | CPU 延迟 (ms) | 路由熵 H_norm |
|------|-------|-----------|----------|-------|--------------|--------------|
| ES-MoE K=1 | 1 | 2.814 | 16.13% | 28.79% | 56.69 | 0.198 |
| **ES-MoE K=2（基线）** ⭐ | 2 | 2.814 | **17.27%** | **30.55%** | 59.94 | 0.490 |
| ES-MoE K=3 | 3 | 2.814 | 15.20% | 27.44% | 59.44 | 0.741 |
| **SharedExpertESMoE** | 2 | **2.727** | 15.16% | 27.30% | 59.05 | 0.430 |
| MoE 基线 v084（参考） | — | 3.14 | 16.84% | 29.79% | — | — |

**关键发现**：

- **K=2 是精度-稀疏平衡点**（17.27%，相对基线 +0.43%）；K=3 路由最均匀（熵 0.741）但精度反而最低（15.20%）——强均衡稀释了专家专门化
- **稀疏化是真实收益**（2026-08-15 GPU+CPU 交叉验证，同权重 Hard vs Dense）：K=1 GPU -23.7% 达验收 #4 优秀标准
- **动态 Top-K 更省**：平均只激活 1.66 个专家（低于 K=2 基线的 2 个）

## 关键技术修复：dynamic_threshold 二次过滤 bug

**问题**：`dynamic_threshold=0.4` 在推理时二次过滤 top-2/top-3 专家，导致 K=3 推理 mAP 崩到 1%。

**修复**：默认值 0.4 → 0.0（[ultralytics/nn/modules/moe/modules.py](../)），K=3 mAP 恢复 15.20%，无需重训。

详见：[docs/06-任务3-性能指标验证.md](docs/06-任务3-性能指标验证.md)

## 文档快速导航

| 编号 | 文档 | 说明 |
|------|------|------|
| 00 | [实战项目原文](docs/00-实战项目原文.md) | 项目要求与验收标准 |
| 01 | [项目分析说明](docs/01-项目分析说明.md) | 需求分析 |
| 02 | [项目结合分析](docs/02-项目结合分析.md) | 与 YOLO-Master 结合 |
| 03 | [任务分析与实验文档](docs/03-实战项目任务分析与实验文档.md) | 任务总览 |
| 04 | [任务 1：ES-MoE 训练](docs/04-任务1-ES-MoE训练.md) | 训练日志 |
| 05 | [任务 2：专家利用率分析](docs/05-任务2-专家利用率分析与负载均衡调优.md) | 路由/熵/负载均衡 |
| 06 | [任务 3：性能指标验证](docs/06-任务3-性能指标验证.md) | 性能数据 |
| 07 | [任务 4：推理效率优化](docs/07-任务4-推理效率优化.md) | 稀疏化/动态 Top-K |
| 08 | [任务 5：可视化分析与交付物](docs/08-任务5-可视化分析与交付物.md) | 交付物清单 |

## Discussion / PR

- **[GitHub Discussion 技术文章](discussion/13-GitHub-Discussion-ESMoE.md)** — Show and tell 分类
- **[PR Title + Body 模板](discussion/ESMoE-PR-Title-Body.md)** — 提交 PR 用

## 引用

```bibtex
@misc{zviolin2026esmoe,
  title  = {YOLO-Master ES-MoE 自适应推理优化：Top-K 稀疏化与动态路由实践},
  author = {张伟林 (Zviolin)},
  year   = {2026},
  url    = {https://github.com/Zviolin/YOLO-Master/tree/practise-1-esmoe-adaptive-inference}
}
```

## 许可证

本项目基于 YOLO-Master 原项目许可证，遵循 AGPL-3.0 License。

## 联系方式

- 邮箱：164321595@qq.com
- GitHub：[@Zviolin](https://github.com/Zviolin)
