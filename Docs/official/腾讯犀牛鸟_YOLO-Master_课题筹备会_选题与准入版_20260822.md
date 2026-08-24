## Slide 1

TENCENT RHINO-BIRD OPEN SOURCE TALENT PROGRAM

腾讯犀牛鸟开源人才计划

YOLO-Master

实战课题大纲

6 条研究主线 · 17 个课题槽位 · 8.14–9.14 开源实战

筹备会终审版 · 2026.08.21

P0 基线：YOLO-Master-v26.08@43d4011；HEAD 课题锁定 57b9ea3

指导老师： isLinXu( 林旭 )

## Slide 2

## Slide 3

## Slide 4

PROGRAM LOGIC

目标：每组带着可运行的 P0 方案离场

腾讯犀牛鸟开源人才计划 · YOLO-Master

02

01

组队与 Owner

主责 / 协作 / 导师接口

02

主选 + 备选

P0 闭环 / P1 贡献 / 降级

03

基线与资源

tag/commit / GPU / 数据 / 24h 冒烟

17 题是候选池：只开放有人、有资源、有 P0 的课题。

散会形成《课题登记表》、GPU 排期与 8.24 冒烟清单。

## Slide 5

PORTFOLIO

六条主线覆盖从算法创新到开源工程化

选题时先看“想训练什么能力”，再看算力与工程边界。 

腾讯犀牛鸟开源人才计划 · YOLO-Master

03

A

对标 YOLO26

3 题

NMS-free · STAL · Export Gap

B

开放世界

2 题

文本条件路由 · 自动标注数据引擎

C

应用竞赛式

3 题

VisDrone 擂台 · OBB × DOTA · 工业 V-PEFT

D

大模型时代

3 题

DINOv3 冻结适配 · Foundation KD · F11 路由蒸馏

E / F

社区影响力 / WebUI

3 + 3 题

社区工具与复测 · YOLO-Master Studio

\* 17 题是候选池，不要求全部开放；WebUI 复用现有 app.py 与 Agent Skill。 

## Slide 6

REPOSITORY FACTS

先冻结代码基线：Release 用于验收，HEAD 用于研究

腾讯犀牛鸟开源人才计划 · YOLO-Master

04

01

稳定基线

YOLO-Master-v26.08 @ 43d4011

Ultralytics 8.4.101 / YOLO26；377 passed · 1 xfailed

02

当前 main

main @ 57b9ea3

2026-08-21 HEAD；新增能力必须锁 commit 并重跑测试

03

成熟度边界

Release 对齐 detect/seg/pose

7 分支是代码表面；其余需配套标签；unified OBB 不支持

04

HEAD 研究能力

Foundation / F11 / one-to-one / diagnostics

可用接口≠已验证精度；无匹配 artifact 不宣称精度/延迟

05

统一规则

P0 默认稳定 tag

仅题目必须依赖 HEAD 才切换；所有组登记 tag/commit

## Slide 7

TOPIC WAVES

8.21 冻结共同底座，8.24 后才进入并行攻坚

腾讯犀牛鸟开源人才计划 · YOLO-Master

07

8.14–8.20

已过/8.21 追认：组队、仓库熟悉、数据与算力清单；缺项写入风险。

8.21–8.24

散会即锁：Owner、主/备选、tag/commit、P0、GPU、数据；完成 24h 冒烟。

8.25–8.31

已开放课题产出首个可比数字；跨组基础设施开始复用。

9.01–9.07

集中中期演示：端到端分支、路由蒸馏、量化与数据折扣统一评审。

9.08–9.12

研究课题补多 seed 与消融；工程课题补兼容性、脚本和 README；PR 进入冻结。

9.13–9.14

提交配置、复现命令、报告、Demo 与 PR；统一答辩并按 P0/P1/P2 验收。

跨组依赖：E3→B1/D1/A3；E1↔E2；A2→C1；D2→F11；B2→C1；A3↔E1。

## Slide 8

EVIDENCE CONTRACT

P0 是结项底线，P1 是优秀候选线，P2 是激励提名加分项

腾讯犀牛鸟开源人才计划 · YOLO-Master

05

P0

结项底线

跑通闭环、结果可信、证据可复现

P1

优秀候选

机制成立、指标达线、形成可评审贡献

P2

提名加分

泛化/分析/发布，形成额外影响力

内部技术分级；最终“通过/优秀”以官方评审为准

✓

复现命令 + 配置 + tag/commit

✓

≥3 seed 或单 seed 局限声明

✓

测试/消融/已知局限

✓

可评审的 PR 或正式 issue

## Slide 9

PRACTICE CALENDAR

8.21 起进入 W2：补齐开题冻结，剩余 24 天交付证据链

W1 已过，不追责但必须追认；8.24 前完成环境与最小冒烟。

腾讯犀牛鸟开源人才计划 · YOLO-Master

06

W1

8.14–8.20

资料 / 组队 / 仓库熟悉（8.21 追认）

W2

8.21–8.24

冻结题目 / commit / GPU / 数据；24h 冒烟

W3

8.25–8.31

首轮全量 / 双周详报 / 首个可比数字

W4

9.01–9.07

中期演示 / 机制分析 / 消融设计

W5

9.08–9.12

多 seed / 参数扫描 / PR 冻结

W6

9.13–9.14

结项报告 / Demo / 答辩与验收

关键门禁：8.21 冻结登记；8.24 冒烟；8.31 首轮；9.07 中期；9.12 冻结；9.14 答辩。

## Slide 10

THEME A

对标 YOLO26

把训练与部署的新机制装进 yolo-master

01

A1 NMS-free

02

A2 STAL

03

A3 Export Gap

08

## Slide 11

THEME A  /  A1

MoE × End-to-End：现有 one-to-one 能否真正免 NMS？

核心问题

MultiTaskHead 已内置 one-to-many + one-to-one；还差训练、推理/导出与 NMS-free 基准闭环。

仓库切入 / 关键提醒

不要从零造 head：从 end2end=True、one2one 分支与 NMS 开关做 2×2 消融。

数据与环境

COCO / VisDrone 子集；单卡 3090/4090；延迟固定 batch=1。

风险 / 降级

若现有分支训练不稳，优先修损失/匹配与双分支协同，禁止重复实现同类 head。

P0｜复现当前 end2end/one2one 路径；建立 NMS on/off 的 mAP 与 CPU/GPU 延迟基线。

P1｜补齐可训练、可导出、可评测闭环；mAP 掉点 ≤2，且 CPU 延迟有可测改善。

P2｜扩展至 seg/pose，或解释 TaskRouter 与 one2one 分支旁路对 MoE 结论的影响。

六周路径

W1

基线/论文

W2

最小分支

W3

全量训练

W4

2×2 中期

W5

3 seed + PR

W6

结项答辩

交付物

端到端闭环修复 PR、2×2 消融矩阵、多 seed 报告。

验收线

现有分支可复现；P1 同时满足精度护栏、导出通过与 CPU 延迟改善。

腾讯犀牛鸟开源人才计划 · YOLO-Master

09

## Slide 12

## Slide 13

THEME A  /  A3

Export Gap：稀疏路由与 dense fallback 谁更端侧友好？

核心问题

eager 可稀疏，而 ONNX/TorchScript 采用 dense fallback；INT8 后精度、时延和专家选择差多少？

仓库切入 / 关键提醒

优先复用 export_capabilities、preflight 与 validation；把“导出后变密集”列为实验变量。

数据与环境

COCO val 5000 张；TensorRT + ONNX Runtime；可选 NCNN/RKNN。

风险 / 降级

若后端不支持稀疏语义，保留 dense fallback，测清 eager↔export 的计算与精度差。

P0｜复用现有导出检查，跑通 detect + ES-MoE/MoT 的 ONNX/TRT FP16 并校准精度。

P1｜至少两类路由模块完成 INT8 PTQ；记录 dense fallback 后的精度、延迟与模型尺寸。

P2｜补齐更多路由族；分析 FP32/INT8 专家选择一致率，并尝试 QAT 挽回精度。

六周路径

W1

导出排坑

W2

FP16/敏感度

W3

两族 INT8

W4

散点图中期

W5

五族/QAT

W6

白皮书

交付物

导出语义清单、量化表、路由一致率、fallback gap 白皮书。

验收线

有可复现 INT8 流水线，并完成至少两类路由模块的完整数据。

腾讯犀牛鸟开源人才计划 · YOLO-Master

11

## Slide 14

THEME B

开放世界

让检测器理解文本，并用基础模型造数据

01

B1 文本条件路由

02

B2 Grounded-SAM 2 数据引擎

12

## Slide 15

THEME B  /  B1

文本条件路由：让“查什么”决定“用哪个专家”

核心问题

文本嵌入能否参与路由，使新类别的语义直接影响 MoT / TaskRouter 的专家选择？

仓库切入 / 关键提醒

新增 aux loss 必须注册到 collect_aux_loss 白名单，否则会被静默跳过。

数据与环境

COCO base/new 约 48/17 类；算力中高，建议 A100 或 2×4090。

风险 / 降级

尺度失配时冻结视觉主干只训投影层；失败可降级为路由可解释性研究。

P0｜跑通轻量开放词汇基线：冻结 CLIP 文本编码器，与 yolo-master 视觉特征做最简对齐。

P1｜实现视觉 + 文本双条件路由；在 COCO base/new split 上取得非随机的新类 zero-shot AP。

P2｜与 Dynamic-DINO 的细粒度 MoE 思路对比，并解释文本语义→专家偏好的映射。

六周路径

W1

OVD 基线

W2

双输入原型

W3

base/new 训练

W4

路由热图

W5

消融 + PR

W6

论文级讨论

交付物

文本路由模块 PR、zero-shot AP、文本→专家可视化、设计文档。

验收线

new 类 AP 高于随机/封闭基线；消融证明文本条件有效。

腾讯犀牛鸟开源人才计划 · YOLO-Master

13

## Slide 16

## Slide 17

THEME C

应用竞赛式

用真实榜单与真实行业数据交付可见成果

01

C1 VisDrone 擂台

02

C2 OBB × DOTA

03

C3 V-PEFT 工业缺陷

15

## Slide 18

THEME C  /  C1

VisDrone 擂台：在同一预算下只比“武器本身”

核心问题

ES-MoE、STAL、P2 高分辨率头与 Latent MoE，谁能在公平预算下稳定赢得小目标榜单？

仓库切入 / 关键提醒

以 README 的 ES-MoE-N 2.68M / 8.7G 为预算锚；统一 epoch、数据、增广与 10% held-out test。

数据与环境

统一 VisDrone2019-DET 镜像；单卡可参赛。

风险 / 降级

W4 允许一次“转会”；原武器的失败分析仍计入成果。

P0｜每组提交可复现的武器、训练配置与完整结果；榜单周更。

P1｜榜单前三名用多 seed 复核；均值最高且置信区间不重叠才是冠军。

P2｜冠军方案沉淀为官方 tutorial 与配置文件，附完整技术报告。

六 周路径

W1

抽签/基线

W2

武器接入

W3

首轮榜单

W4

交流/转会

W5

多seed决赛

W6

冠军报告

交付物

周更榜单、各组配置 PR、冠军技术报告、官方 tutorial。

验收线

可复现是入场券；排名以多 seed 均值与置信区间裁决。

腾讯犀牛鸟开源人才计划 · YOLO-Master

16

## Slide 19

THEME C  /  C2

OBB × DOTA：从兼容占位走向统一多任务分支

核心问题

仓库已有独立 OBB 实验能力，但 MultiTaskHead 将 obb 标为 compatibility-only；怎样补成可训练统一分支？

仓库切入 / 关键提醒

先区分独立 OBB 基线与 unified multitask OBB；不要把“已有 OBB”误报成零基础。

数据与环境

DOTA v1.0；裁剪与旋转框转换是关键；可降级 HRSC。

风险 / 降级

统一分支工程量过大时先用 HRSC/单类子集；联合训练崩溃则保留独立与统一单任务对照。

P0｜复现独立 OBB 基线；定位 compatibility-only 与 set_active_tasks 的边界并提交设计。

P1｜新增 unified multitask OBB 训练分支；在 DOTA/HRSC 上获得非平凡 mAP。

P2｜与 detect/seg/pose 联合训练，或完成 unified OBB × MoE 组合实验。

六 周路径

W1

拒绝点/DOTA

W2

head/loss

W3

首个数字

W4

差距分析

W5

联合训练/PR

W6

目标合并

交付物

unified OBB PR、独立/统一基线、DOTA/HRSC 差距表。

验收线

训练曲线健康；统一分支显著高于随机，并与独立 OBB 基线同口径对照。

腾讯犀牛鸟开源人才计划 · YOLO-Master

17

## Slide 20

## Slide 21

THEME D

大模型时代

复用现有 foundation stack，做冻结适配、蒸馏验证与路由 KD

01

D1 冻结 DINOv3

02

D2 Foundation 蒸馏

03

F11 路由蒸馏验证

19

## Slide 22

THEME D  /  D1

冻结 DINOv3 × LatentMixture：训练成本能降多少？

核心问题

不微调 foundation backbone，只训练 Latent MoE 适配与检测头，能保留多少精度？

仓库切入 / 关键提醒

复用 foundation/teachers 与预处理协议；latent aux 仍需显式加入 collect_aux_loss 的 include_kinds。

数据与环境

COCO + VisDrone 子集；复用 foundation teacher 管线；4090 级足够，需规划缓存磁盘。

风险 / 降级

缓存成本过高时先做 COCO-mini/VisDrone；三尺度对齐不稳时先固定单尺度接口。

P0｜复用 DINOv3 teacher adapter 抽特征，接 LatentMixture + 检测头，完成训练评测。

P1｜与同参数量从零训练检测器做精度/显存/时长对照，覆盖至少 2 个数据集。

P2｜显式注册 latent aux，扫描 balance/z-loss；或对比 DINOv3 与 SigLIP2 teacher。

六周路径

W1

特征管线

W2

适配头冒烟

W3

首数据集

W4

第二数据集

W5

尺寸/aux扫描

W6

性价比报告

交付物

训练配方 PR、三维对照、latent aux 消融与 include_kinds 修复。

验收线

至少 1 个数据集明确给出精度保留比例，且 GPU 时降低 ≥50%。

腾讯犀牛鸟开源人才计划 · YOLO-Master

20

## Slide 23

THEME D  /  D2

Foundation 蒸馏现状验证：把已实现能力跑出可信结论

核心问题

现有模型已覆盖 DINOv3/SigLIP2、多尺度、语义与路由 KD；哪些组合在同预算下真正有效？

仓库切入 / 关键提醒

不再从零实现：先做配置—代码—测试清单，再围绕现有 Foundation 功能做最小消融矩阵。

数据与环境

COCO/COCO-mini；单卡起步；teacher 缓存与在线推理分别计时。

风险 / 降级

组合爆炸时锁定 DINOv3 P4 单教师；\|ΔmAP\|\<0.3 且 CI 含 0 判 no-go。

P0｜跑通当前 foundation distill 路径；核对 teacher、tap、projector、loss 与日志。

P1｜DINOv3/SigLIP2 或 P4/multiscale 完成 2×2 中至少 3 格，同预算对照。

P2｜扩展 semantic/multi-teacher，或提交配置/测试 bugfix 与 go/no-go 建议。

六周路径

W1

文献/设计

W2

单stage原型

W3

首轮对照

W4

诚实中期

W5

来源/原因

W6

立项建议

交付物

HEAD 能力地图、复现实验、消融表、配置/测试 PR 与 go/no-go。

验收线

每个结论映射现有 config+code；无混杂变量；负结果必须有证据。

腾讯犀牛鸟开源人才计划 · YOLO-Master

21

## Slide 24

## Slide 25

THEME E

社区影响力

把研究结果沉淀为工具、基准与可视化资产

01

E1 ES-MoE 工具包

02

E2 涨点打假

03

E3 MoE 透视镜

23

## Slide 26

THEME E  /  E1

ES-MoE 工具包：把“民间涨点”做成官方能力

核心问题

能否在严格预算与多 seed 下选出真最优模块，并三行代码接入多代 Ultralytics？

仓库切入 / 关键提醒

插件化最大难点不是打包，而是让宿主项目正确接收 collect_aux_loss。

数据与环境

COCO 为主；算力需求高，建议 2–4 卡并与 E2 共享排期。

风险 / 降级

算力不足缩到两代+单主干；工程过大则先做仓库内独立模块。

P0｜以 README 的 ES-MoE-N（2.68M/8.7G/.427）为锚，3 seed 复核硬件与评测口径。

P1｜完成 pip/配置注入式插件，兼容 YOLOv8 / YOLO11 / YOLOv12，提供 README + Colab。

P2｜发布 PyPI/独立包，产出中英教程，并根据首轮社区反馈迭代。

六周路径

W1

评测脚手架

W2

五代复测

W3

选型/API

W4

v8 demo

W5

三代兼容

W6

博客/发布

交付物

插件包、多 seed 消融表、README/Colab、教程博客。

验收线

官方基线可复现；三代主干零报错，插件相对 baseline 的 3 seed 区间不重叠。

腾讯犀牛鸟开源人才计划 · YOLO-Master

24

## Slide 27

THEME E  /  E2

“涨点打假”基准：把单 seed 神话变成统计结论

核心问题

社区流行注意力、融合、卷积与 ES-MoE 模块，哪些真涨、哪些只是随机噪声？

仓库切入 / 关键提醒

报告中的每个“显著提升”都必须有配对检验数字；措辞限定在复现条件下。

数据与环境

COCO 或明示局限的统一小基准；每配置 ≥3 seed。

风险 / 降级

优先选有官方实现的模块；报告对事不对人，明确复现边界。

P0｜搭建固定 seed、统一预算与自动统计脚手架；复测 3 个流行模块。

P1｜扩展至 6–8 个模块，给出真涨 / 噪声级 / 需更多证据的分级结论。

P2｜发布报告，并加入同 FLOPs 的预算公平对照。

六周路径

W1

清单/脚手架

W2

首批3个

W3

首批判读

W4

中期/扩容

W5

全量复测

W6

公开发布

交付物

统计评测脚手架 PR、模块×seed 大表、分级结论报告。

验收线

每个结论都有配对 t 检验或 bootstrap CI 支撑。

腾讯犀牛鸟开源人才计划 · YOLO-Master

25

## Slide 28

## Slide 29

THEME F

开源工程化

连接成训练、推理、部署一体化工作台

01

F1 Studio Core

02

F2 TrainOps

03

F3 DeployOps

27

## Slide 30

THEME F  /  F1

YOLO-Master Studio：把推理 Demo 升级成任务工作台

核心问题

app.py 已覆盖五类单图推理；Agent Skill 已有异步提交、状态与取消，课题重点是产品化衔接。

仓库切入 / 关键提醒

复用 policy.async、yolo.job.status/cancel、progress.jsonl；WebUI 不再自建第二套执行器。

数据与环境

Gradio 优先；coco8 / 样例图；CPU 可冒烟，单 GPU 做完整演示。

风险 / 降级

当前终态主要按 PID 存活判断；需用 manifest/结果区分 succeeded / failed / cancelled。

P0｜保留现有推理；新增 Jobs 页，接入 doctor + predict 的提交、状态、日志与产物。

P1｜修正异步终态语义；补 job list、取消/超时、路径白名单与批量图片/视频推理。

P2｜接入 E3 路由诊断/多模态面板；可选 FastAPI + React 前后端分离。

六周路径

W1

async 考古

W2

Jobs 适配

W3

推理/状态

W4

失败/取消

W5

测试 + PR

W6

文档/答辩

交付物

Studio PR、异步任务适配与终态修复、端到端测试、2 分钟演示视频。

验收线

P0 跑通 doctor + predict；P1 区分成功/失败/取消，重启后任务与产物可追溯。

腾讯犀牛鸟开源人才计划 · YOLO-Master

28

## Slide 31

THEME F  /  F2

训练中心：从参数填表到可复现实验

核心问题

训练参数、GPU 资源与长任务生命周期必须在 WebUI 中可控，不能只把 CLI 表单化。

仓库切入 / 关键提醒

复用 yolo.train/val、pipeline.experiment 与 progress.jsonl；F2 只负责训练资源与实验。

数据与环境

coco8 / COCO-mini；单卡优先；W1 登记每组 GPU 时段与并发上限。

风险 / 降级

现有 async 可并发启动多任务；P1 补单 GPU lease / 排队与 backpressure，避免显存争抢。

P0｜可视化填写训练参数、预览 request；跑通 coco8 1 epoch，并实时展示进度。

P1｜支持停止/续训、曲线、显存、checkpoint；实现单 GPU 独占队列与双实验对比。

P2｜扩展多 GPU 调度，或增加 V-PEFT / 全量微调对比面板。

六周路径

W1

参数契约

W2

表单/dry-run

W3

任务队列

W4

曲线/续训

W5

GPU lease

W6

PR/演示

交付物

Train 页、GPU lease / 队列、任务状态存储、可复现实验包与自动化测试。

验收线

P0 完成真实训练；P1 并发提交不抢卡，UI 重启后可恢复任务与复现证据。

腾讯犀牛鸟开源人才计划 · YOLO-Master

29

## Slide 32

THEME F  /  F3

部署中心：导出、量化与 Benchmark 向导

核心问题

导出前先说明支持/拒绝与 sparse/dense fallback，再用同口径比较精度、时延和体积。

仓库切入 / 关键提醒

复用 export_preflight、能力矩阵与 yolo.export；A3 研究 Export Gap，F3 负责向导与结果固化。

数据与环境

ONNX Runtime 必做；有 NVIDIA 环境再做 TensorRT；COCO val 子集校准。

风险 / 降级

无 TensorRT 时仍须完成 ONNX Runtime 静态 INT8 PTQ；硬件专用结果只在对应运行时发布。

P0｜选择 checkpoint/后端，运行 strict preflight；展示拒绝/fallback 并导出 ONNX FP32。

P1｜ONNX Runtime INT8 PTQ 必做；有 NVIDIA 再补 TensorRT FP16/INT8 与同口径对比。

P2｜尝试 QAT、远端设备插件，或把 A3 的路由一致率分析接入面板。

六周路径

W1

能力矩阵

W2

ONNX 导出

W3

benchmark

W4

INT8/TRT

W5

护栏/测试

W6

PR/报告

交付物

Deploy 页、preflight 报告、产物/benchmark 表与自动化测试。

验收线

无效组合预检拒绝；FP32/INT8 同样本比较，产物记录源权重、后端、校准集与 checksum。

腾讯犀牛鸟开源人才计划 · YOLO-Master

30

## Slide 33

WEBUI WORK PACKAGES

不是三套 WebUI，而是一个平台的三个工作包

腾讯犀牛鸟开源人才计划 · YOLO-Master

31

F1

平台内核

job adapter / 推理 / 模型资产

F2

训练中心

配置 / 队列 / 进度 / 续训

F3

部署中心

预检 / 导出 / 量化 / benchmark

共享接口必须在 8.24 锁定

✓

Request JSON + request_id + profile/model/data

✓

policy.async + yolo.job.status / yolo.job.cancel

✓

progress.jsonl + skill_manifest.json + bounded logs

✓

路径白名单 + 脱敏 + 禁止任意 shell

## Slide 34

STUDIO ARCHITECTURE

Studio 不重写训练引擎，只补齐产品编排层

腾讯犀牛鸟开源人才计划 · YOLO-Master

32

01  WebUI Shell

Gradio：Home / Train / Predict / Deploy / Jobs

02  Request Gateway

JSON schema / dry-run / request_id / 路径白名单

03  Agent Dispatcher

yolo.train / val / predict / export / benchmark / doctor

04  Async Jobs

policy.async + status/cancel；补终态语义与 GPU lease

05  Evidence Stream

progress.jsonl + stdout/stderr + metrics + recovery

06  Artifact Registry

skill_manifest + checkpoints + exports + checksum

统一 smoke：doctor → coco8 train → predict → preflight → ONNX/INT8 → benchmark

## Slide 35

DEPENDENCIES

课题联动的价值，是把一次开发变成多组共同基础设施

腾讯犀牛鸟开源人才计划 · YOLO-Master

33

E3 → F1

路由诊断 schema 与空间热图直接进入 Studio 推理分析面板

E1 ↔ E2

共享多 seed 统计脚手架、GPU 排期与 ES-MoE 复测数据

A2 → C1

STAL 是擂台合法武器；A2 组可直接提供实现与参数经验

D2 → F11

现有 Foundation 蒸馏基线与 teacher adapter 直接支撑路由 KD

B2 → C1

自动标注数据可成为擂台加赛数据集，验证噪声鲁棒性

A3 → F3

量化、fallback 与导出实验沉淀为 DeployOps 的预检与 benchmark 工作流

公共执行底座：Agent Skill → F1/F2/F3；WebUI 不直接拼接 shell 命令。

## Slide 36

TOPIC SELECTION

A–E 看矩阵；F 组三题按“一个平台、三个工作包”组队

腾讯犀牛鸟开源人才计划 · YOLO-Master

34

课题

主能力

算力

研究不确定性

最适合

A1

端到端闭环 / 分配

中

高

算法研究型

A2

标签分配

低–中

低

稳健涨点型

A3

部署 / 量化

中

中

端侧工程型

B1

开放词汇 / 路由

高

高

前沿研究型

B2

数据闭环

中

中

数据工程型

C1

公平竞赛 / 调参

低–中

中

竞赛协作型

C2

unified OBB 分支

中

中–高

强工程型

C3

PEFT / 工业

低

低–中

应用落地型

D1

基础模型适配

中

中

性价比研究型

D2

蒸馏验证 / 消融

中

中

实验复核型

F11

路由 KD 验证

中–高

中

机制验证型

E1

插件 / 社区

高

中

开源产品型

E2

统计复测

高

中

严谨评测型

E3

诊断 / 可视化

低

低–中

工程展示型

短周期优先：A2/A3/C3/E3/F1；F2/F3 仅在 GPU 与后端环境齐备后开放。

## Slide 37

ACCEPTANCE REDLINES

提交开题与结项前，只检查这七件事

腾讯犀牛鸟开源人才计划 · YOLO-Master

35

01  复现命令与配置文件路径齐全

02  基线固定到 tag/commit；禁止只写 main/HEAD

03  “提升/下降”有 ≥3 seed，或明确声明单 seed 局限

04  对照实验预算、数据、增广与评测口径一致

05  负结果预先有判读标准，结论有原因证据链

06  PR 描述包含：改动摘要 / 测试证据 / 消融数据 / 已知局限

07  README 与源码成熟度差异已标注；禁止沿用历史行号和旧能力判断

## Slide 38

TOPIC MATCHING

选题分两阶段：个人报三志愿，组队后锁定一主一备

腾讯犀牛鸟开源人才计划 · YOLO-Master

36

个人填报

提交第1/第2/第3志愿；同步登记技能、每周投入、GPU与数据条件。

导师撮合

综合志愿顺序、技能匹配、课题容量、算力资源和团队结构进行撮合。

团队确认

每组锁定1个主选 + 1个备选/降级方向；明确Owner、成员与分工。

准入准备

登记tag/commit、P0验收线、复现命令、GPU时段和数据路径。

正式开题

8.24准入通过后启动完整实验；未通过则切备选或降级。

结项验收

9.13–9.14按P0/P1/P2内部技术分级验收；官方等级以计划评审为准。

原则：尊重志愿，但不按“先到先得”机械分配；避免同题拥挤与公共课题无人负责。

## Slide 39

8.24 ADMISSION GATE

8.24 检查的是能否开工，不是最终是否结项

腾讯犀牛鸟开源人才计划 · YOLO-Master

37

01  Owner、成员与职责已确认；主选和备选均可落地

02  固定 tag/commit；环境可启动，数据样例可读取

03  baseline 或最小代码链路可执行，并留存日志/测试结果

04  GPU时段、P0指标、复现命令与失败降级方案已登记

05  通过：主选课题正式进入全量实验

06  条件通过：限时补齐缺项，先按缩小范围启动

07  未通过：切换备选课题，或降低数据/模型/任务范围

## Slide 40

NEXT STEP

选一个真问题，

交付一条可复现的证据链

散会前完成

① 确认 Owner 与主/备选题   ② 记录 tag/commit、GPU、数据   ③ 约定 8.24 冒烟证据

YOLO-Master · 8.21 筹备会 → 9.14 结项

38

## Slide 41

NEXT STEP

T hanks

YOLO-Master · 8.21 筹备会 → 9.14 结项

38

【第一步】本周内完成三件事：(大家注意修改群名称中带上 GitHub ID 以便交流协作)

加入会议：08月21日 20:00 首次全员线上会，讲解任务与课题池，务必出席；

跑通基线：按  README  搭好环境，跑通  detect / seg / pose  任一任务训练；

填写志愿：会后 24 h  内在共享表格填课题志愿。

【日常节奏】每周一在群里发"上周进展 + 本周计划 + 卡点"三段式打卡；每周 1-2 次线上答疑；每双周交一次阶段详报（模板会上发）。导师每天 22:00 前后集中回复群内问题，急事可 @班长 ( tomlrh ) 。

【组织】将选 1 名班长协助管理，各组设 1 名  Owner。 表现优秀的同学有机会获得导师提名与主办方礼品激励
