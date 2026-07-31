# YOLO-Master Conda 环境配置说明

> 本文件基于 **yolo-master** Conda 环境实际包列表（`pip list` 输出）编写，与 YOLO-Master 仓库 [pyproject.toml](../../Codes/YOLO-Master/pyproject.toml) 和 [requirements.txt](../../Codes/YOLO-Master/requirements.txt) 保持一致
>
> 设备：NVIDIA RTX 5060 Laptop (8GB)，Windows 11，PowerShell

---

## 一、环境基本信息

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **Conda 安装路径** | `E:\Apps\Codes\Conda\` | 自定义安装路径 |
| **环境名称** | `yolo-master` | 用于 YOLO-Master 项目 |
| **Python 版本** | **3.10.9** | 与 ultralytics 全平台兼容性最好 |
| **PyTorch 版本** | **2.9.1+cu130** | 支持 CUDA 13.0 |
| **CUDA 版本** | **13.0** | 匹配 RTX 5060（Blackwell 架构） |
| **ultralytics 版本** | **8.3.240**（editable）| YOLO-Master Fork 版本，通过 `pip install -e .` 安装 |
| **ultralytics 路径** | `G:\Codes\OpenSource\Rhino-bird\Codes\YOLO-Master` | 可编辑安装位置 |

> **重要约束**（来自 [pyproject.toml#L72](../../Codes/YOLO-Master/pyproject.toml#L72)）：
> - Windows 下**不要使用 PyTorch 2.4.0**（已知 CPU 报错，参考 [ultralytics#15049](https://github.com/ultralytics/ultralytics/issues/15049)）

---

## 二、深度学习框架栈（实际安装版本）

### 2.1 PyTorch + CUDA

| 包名 | 版本 | 说明 |
|------|------|------|
| `torch` | **2.9.1+cu130** | PyTorch 深度学习框架（CUDA 13.0 构建版） |
| `torchaudio` | **2.9.1+cu130** | 音频处理 |
| `torchvision` | **0.24.1+cu130** | 计算机视觉工具 |
| `torch-geometric` | **2.7.0** | 图神经网络（可选，用于图数据） |
| `xformers` | **0.0.33.post2** | 内存高效注意力（可选，加速 Transformer） |

> **版本一致性**：torch / torchaudio / torchvision 均使用 `+cu130` 后缀，确保 CUDA 13.0 兼容。

### 2.2 Transformer 生态（HuggingFace + PEFT）

| 包名 | 版本 | 用途 |
|------|------|------|
| `transformers` | **4.30.2** | HuggingFace Transformers（预训练模型库） |
| `tokenizers` | **0.13.3** | 快速分词器 |
| `huggingface-hub` | **0.36.0** | HF 模型下载中心 |
| `hf-xet` | **1.2.0** | HF Xet 高效下载协议 |
| `datasets` | **4.4.1** | HF 数据集加载 |
| `sentence-transformers` | **5.1.2** | 句向量模型（可选） |
| `sentencepiece` | **0.2.1** | 子词分词器 |
| `peft` | **0.18.1** | LoRA / Prefix Tuning 微调（YOLO-Master Issue #49 用） |
| `accelerate` | **1.12.0** | HuggingFace 训练加速 |
| `safetensors` | **0.7.0** | 安全模型权重格式 |
| `bitsandbytes` | **0.49.0** | 8-bit/4-bit 量化（低显存训练） |

### 2.3 实验追踪与可视化

| 包名 | 版本 | 用途 |
|------|------|------|
| `wandb` | **0.28.0** | Weights & Biases 实验追踪 |
| `tensorboard` | **2.21.0** | TensorBoard 可视化 |
| `tensorboard-data-server` | **0.7.2** | TensorBoard 数据后端 |
| `ml-dtypes` | **0.5.4** | TensorFlow 类型支持（间接依赖） |

### 2.4 模型导出与部署

| 包名 | 版本 | 用途 |
|------|------|------|
| `onnx` | **1.22.0** | ONNX 模型导出 |
| `onnxsim` | **0.6.5** | ONNX 模型简化 |
| `graphviz` | **0.21** | 模型结构可视化 |

---

## 三、计算机视觉与图像处理

| 包名 | 版本 | 用途 |
|------|------|------|
| `opencv-python` | **4.12.0.88** | OpenCV 图像处理 |
| `pillow` | **12.0.0** | PIL 图像库 |
| `scikit-image` | **0.25.2** | 高级图像处理 |
| `tifffile` | **2025.5.10** | TIFF 图像读写 |
| `ImageIO` | **2.37.2** | 多格式图像 IO |

---

## 四、数据处理与科学计算

| 包名 | 版本 | 用途 |
|------|------|------|
| `numpy` | **2.2.6** | 数值计算 |
| `pandas` | **2.3.3** | DataFrame 数据分析 |
| `polars` | **1.35.2** | 高性能 DataFrame（YAML 解析加速） |
| `polars-runtime-32` | **1.35.2** | Polars 32-bit 运行时 |
| `scipy` | **1.15.3** | 科学计算 |
| `scikit-learn` | **1.7.2** | 机器学习工具 |
| `networkx` | **3.4.2** | 图算法 |
| `pyarrow` | **22.0.0** | Arrow 列式存储 |
| `joblib` | **1.5.2** | 并行计算 |

---

## 五、可视化与绘图

| 包名 | 版本 | 用途 |
|------|------|------|
| `matplotlib` | **3.10.7** | 绘图基础 |
| `seaborn` | **0.13.2** | 统计绘图 |
| `contourpy` | **1.3.2** | matplotlib 等高线 |
| `kiwisolver` | **1.4.9** | matplotlib 布局求解 |
| `fonttools` | **4.61.0** | 字体处理 |
| `Pillow` | **12.0.0** | （同时属于图像处理）|

---

## 六、网络与 HTTP

| 包名 | 版本 | 用途 |
|------|------|------|
| `requests` | **2.32.5** | HTTP 客户端 |
| `urllib3` | **2.5.0** | HTTP 底层 |
| `certifi` | **2025.11.12** | CA 证书 |
| `httpx` | **0.28.1** | 异步 HTTP |
| `httpcore` | **1.0.9** | httpx 底层 |
| `h11` | **0.16.0** | HTTP/1.1 协议 |
| `charset-normalizer` | **3.4.4** | 字符编码 |
| `idna` | **3.11** | 国际化域名 |
| `aiohttp` | **3.13.2** | 异步 HTTP 客户端 |
| `aiosignal` | **1.4.0** | aiohttp 信号 |
| `aiohappyeyeballs` | **2.6.1** | aiohttp 连接优化 |
| `frozenlist` | **1.8.0** | aiohttp 数据结构 |
| `multidict` | **6.7.0** | aiohttp 多键字典 |
| `yarl` | **1.22.0** | URL 解析 |
| `propcache` | **0.4.1** | 属性缓存 |
| `fastapi` | **0.123.5** | Web API（可选） |
| `starlette` | **0.50.0** | FastAPI 底层 |
| `uvicorn` | **0.38.0** | ASGI 服务器 |
| `anyio` | **4.12.0** | 异步兼容层 |

---

## 七、模型服务与 LLM 工具

| 包名 | 版本 | 用途 |
|------|------|------|
| `ollama` | **0.6.1** | Ollama LLM 本地推理（可选） |
| `modelscope` | **1.10.0** | ModelScope 模型下载（备选 HF） |
| `oss2` | **2.19.1** | 阿里云 OSS（ModelScope 依赖） |
| `aliyun-python-sdk-core` | **2.16.0** | 阿里云 SDK（oss2 依赖） |
| `aliyun-python-sdk-kms` | **2.16.5** | 阿里云 KMS |
| `crcmod` | **1.7** | CRC 校验（oss2 依赖） |
| `jmespath` | **0.10.0** | JSON 查询（oss2 依赖） |

---

## 八、Ultralytics 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| `ultralytics` | **8.3.240**（editable）| YOLO-Master Fork 版本（`pip install -e .`） |
| `ultralytics-thop` | **2.0.18** | FLOPs 计算 |
| `pyyaml` | **6.0.3** | YAML 解析 |
| `addict` | **2.4.0** | dict 子类（ultralytics 配置） |

---

## 九、配置文件与解析

| 包名 | 版本 | 用途 |
|------|------|------|
| `PyYAML` | **6.0.3** | YAML 解析 |
| `Markdown` | **3.10.2** | Markdown 解析 |
| `markdown-it-py` | **4.2.0** | Markdown 解析器 |
| `mdurl` | **0.1.2** | markdown-it-py URL 工具 |
| `Pygments` | **2.20.0** | 代码高亮 |
| `rich` | **15.0.0** | 富文本终端输出 |
| `typer-slim` | **0.20.0** | CLI 框架 |
| `shellingham` | **1.5.4** | shell 检测 |
| `click` | **8.3.1** | CLI 工具 |
| `colorama` | **0.4.6** | 跨平台彩色终端 |
| `Jinja2` | **3.1.6** | 模板引擎 |
| `MarkupSafe` | **2.1.5** | Jinja2 安全转义 |

---

## 十、Git 与版本控制

| 包名 | 版本 | 用途 |
|------|------|------|
| `GitPython` | **3.1.50** | Git 操作库 |
| `gitdb` | **4.0.12** | GitPython 底层 |
| `smmap` | **5.0.3** | GitPython 内存映射 |
| `setuptools` | **80.9.0** | 打包工具 |
| `wheel` | **0.45.1** | 打包格式 |
| `packaging` | **25.0** | 版本处理 |
| `yapf` | **0.43.0** | Google 代码格式化 |

---

## 十一、测试与质量保证

| 包名 | 版本 | 用途 |
|------|------|------|
| `pytest` | **9.1.1** | 单元测试框架 |
| `pytest-cov` | **7.1.0** | 测试覆盖率 |
| `coverage` | **7.15.2** | 覆盖率核心 |
| `pluggy` | **1.6.0** | pytest 插件机制 |
| `iniconfig` | **2.3.0** | INI 配置 |
| `tomli` | **2.3.0** | TOML 解析（pytest 依赖） |
| `exceptiongroup` | **1.3.1** | 异常组（pytest 依赖） |

---

## 十二、系统监控与工具

| 包名 | 版本 | 用途 |
|------|------|------|
| `psutil` | **7.1.3** | 系统资源监控 |
| `filelock` | **3.20.0** | 文件锁 |
| `platformdirs` | **4.5.0** | 跨平台路径 |
| `typing_extensions` | **4.15.0** | 类型注解扩展 |
| `typing-inspection` | **0.4.2** | 类型检查 |
| `annotated-doc` | **0.0.4** | 注解文档 |
| `annotated-types` | **0.7.0** | 注解类型 |
| `pydantic` | **2.12.5** | 数据验证 |
| `pydantic_core` | **2.41.5** | pydantic 核心 |
| `attrs` | **25.4.0** | 类属性简化 |
| `sortedcontainers` | **2.4.0** | 排序容器 |
| `threadpoolctl` | **3.6.0** | 线程池控制 |
| `mpmath` | **1.3.0** | 任意精度数学 |
| `sympy` | **1.14.0** | 符号数学 |
| `fsspec` | **2025.10.0** | 文件系统接口 |
| `xxhash` | **3.6.0** | 高速哈希 |
| `six` | **1.17.0** | Python 2/3 兼容 |

---

## 十三、时间与日期

| 包名 | 版本 | 用途 |
|------|------|------|
| `python-dateutil` | **2.9.0.post0** | 日期处理 |
| `pytz` | **2025.2** | 时区 |
| `tzdata` | **2025.2** | 时区数据 |

---

## 十四、安全与加密

| 包名 | 版本 | 用途 |
|------|------|------|
| `cryptography` | **46.0.3** | 加密算法 |
| `pycparser` | **2.23** | C 语言解析 |
| `cffi` | **2.0.0** | C 语言调用 |
| `pycryptodome` | **3.23.0** | 加密原语 |

---

## 十五、其他工具

| 包名 | 版本 | 用途 |
|------|------|------|
| `tqdm` | **4.67.1** | 进度条 |
| `regex` | **2025.11.3** | 正则表达式 |
| `dill` | **0.4.0** | 序列化扩展 |
| `multiprocess` | **0.70.18** | 多进程 |
| `simplejson` | **3.20.2** | JSON 解析 |
| `et_xmlfile` | **2.0.0** | XML 解析 |
| `openpyxl` | **3.1.5** | Excel 读写 |
| `protobuf` | **7.35.1** | 协议缓冲 |
| `grpcio` | **1.81.1** | gRPC |
| `sentry-sdk` | **2.64.0** | 错误监控 |
| `absl-py` | **2.5.0** | Google 通用库 |
| `lazy_loader` | **0.4** | 懒加载 |
| `gast` | **0.7.0** | AST 扩展 |
| `pyparsing` | **3.2.5** | 语法解析 |
| `einops` | **0.8.1** | 张量操作 |
| `pip` | **25.3** | 包管理器 |

---

## 十六、环境创建过程

### 16.1 环境克隆（推荐方式）

从已有的 `mamba_sm120` 环境克隆，避免重新安装 PyTorch：

```powershell
# 1. 加载 conda hook（每次打开新终端都需要执行）
& "E:\Apps\Codes\Conda\Scripts\conda.exe" "shell.powershell" "hook" | Out-String | Invoke-Expression

# 2. 克隆环境
conda create --name yolo-master --clone mamba_sm120

# 3. 激活环境
conda activate yolo-master
```

**注意**：克隆后 `ultralytics` 包可能不是 YOLO-Master Fork 版本，需执行 §16.3 的可编辑安装。

### 16.2 手动创建（备选方案）

如果需要从头创建环境：

```powershell
# 1. 创建新环境（Python 3.10 与 YOLO-Master 兼容性最好）
conda create --name yolo-master python=3.10 -y

# 2. 激活环境
conda activate yolo-master

# 3. 安装 PyTorch（CUDA 13.0，需要较长时间）
conda install pytorch torchvision torchaudio pytorch-cuda=13.0 -c pytorch -c nvidia -y

# 4. 安装基础依赖（必需依赖部分）
pip install "numpy>=1.23.0" "matplotlib>=3.3.0" "opencv-python>=4.6.0" "pillow>=7.1.2" "pyyaml>=5.3.1" "requests>=2.23.0" "scipy>=1.4.1" "psutil>=5.8.0" "polars>=0.20.0" "ultralytics-thop>=2.0.18"

# 5. 安装 YOLO-Master 项目（开发模式，见 §16.3）
```

### 16.3 安装 YOLO-Master（开发模式）

YOLO-Master 的包名是 `ultralytics`（基于 [pyproject.toml#L27](../../Codes/YOLO-Master/pyproject.toml#L27)），**必须**通过 `pip install -e .` 以可编辑模式安装，才能使用 Fork 版本的所有自定义代码（MoT/MoA/MoE 增强、SharedExpertMoE 等）。

```powershell
# 1. 激活环境
conda activate yolo-master

# 2. 进入项目目录
cd g:\Codes\OpenSource\Rhino-bird\Codes\YOLO-Master

# 3. 安装 YOLO-Master（开发模式）
pip install -e .

# 4. 安装项目额外依赖（来自 requirements.txt）
pip install -r requirements.txt

# 5. 安装测试和实验依赖
pip install pytest pytest-cov "coverage[toml]"

# 6. 安装训练加速与 LoRA 依赖
pip install accelerate peft "bitsandbytes"
```

验证安装：
```powershell
pip show ultralytics
# 应显示：
# Name: ultralytics
# Version: 8.3.240
# Location: G:\Codes\OpenSource\Rhino-bird\Codes\YOLO-Master
# Editable project location: ...
```

---

## 十七、CUDA 依赖修复（Windows 特有）

### 17.1 问题描述

导入 torch 时报错：
```
OSError: [WinError 126] 找不到指定的模块
```

### 17.2 解决方案

复制 `mamba_sm120` 的 Library 目录到 `yolo-master` 环境：

```powershell
# 源目录
$source = "E:\Apps\Codes\Conda\envs\mamba_sm120\Library"

# 目标目录
$target = "E:\Apps\Codes\Conda\envs\yolo-master\Library"

# 复制（如果目标已存在，先备份）
Copy-Item -Path $source -Destination $target -Recurse -Force
```

**原因分析**：
- CUDA 相关 DLL 依赖未正确安装
- 克隆环境时部分系统库未完全复制
- 手动复制 Library 目录可解决依赖缺失问题

---

## 十八、环境验证

### 18.1 验证 PyTorch 和 CUDA

```python
import torch
import torchvision
import torchaudio

print(f"PyTorch 版本: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"cuDNN: {torch.backends.cudnn.version()}")
```

**预期输出**（基于 yolo-master 实际环境）：
```
PyTorch 版本: 2.9.1+cu130
TorchVision: 0.24.1+cu130
CUDA 可用: True
CUDA 版本: 13.0
GPU 设备: NVIDIA GeForce RTX 5060 Laptop GPU
显存总量: 8.00 GB
cuDNN: <version>
```

### 18.2 验证 YOLO-Master Fork 安装

```python
from ultralytics import YOLO

# 验证是 Fork 版本（应能加载 v0_8 配置）
model = YOLO('ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml')

# 查看模型信息（参数、FLOPs）
model.info()
```

**预期输出**：
```
YOLO-Master v0.8 baseline summary: 225 layers, 3137637 parameters, 0 gradients, 8.2 GFLOPs
```

### 18.3 验证 MoT/MoE 自定义模块

```python
# 验证 MoT 自定义模块可用
from ultralytics.nn.modules.mot import MoTBlock, _LocalConvTransformerExpert
from ultralytics.nn.modules.moe import SharedExpertMoE

print("✅ YOLO-Master Fork 模块加载成功")
```

### 18.4 验证完整包统计

```powershell
conda activate yolo-master
pip list | wc -l
```

**预期输出**：`127`（共 127 个包，含 ultralytics 等）

---

## 十九、环境管理命令

### 19.1 常用命令

```powershell
# 查看所有环境
conda env list

# 激活环境
conda activate yolo-master

# 退出环境
conda deactivate

# 查看已安装包
conda list
pip list

# 导出环境配置
conda env export > environment.yml

# 从配置文件创建环境
conda env create -f environment.yml

# 删除环境
conda remove --name yolo-master --all
```

### 19.2 环境备份

```powershell
# 导出完整环境（不含 build 号，便于跨平台迁移）
conda env export --no-builds > yolo-master-env.yml

# 仅导出 pip 安装的包
pip freeze > requirements-frozen.txt
```

---

## 二十、常见问题

### 20.1 显存不足（OOM）

**解决方案**：
```python
# 1. 减小 batch size
batch=2  # 从 4 降到 2

# 2. 减小图像尺寸
imgsz=512  # 从 640 降到 512

# 3. 启用梯度检查点
model.train(data='coco8.yaml', epochs=300, batch=2, imgsz=512, amp=True)

# 4. 清理显存
import torch
torch.cuda.empty_cache()
```

### 20.2 Windows 下 PyTorch 2.4.0 CPU 错误

**问题**：Windows + Python 3.10/3.11 下 PyTorch 2.4.0 触发 CPU 段错误。

**解决方案**：
- 升级到 PyTorch >= 2.5.0
- 或降级到 PyTorch == 2.3.x
- 参考 [ultralytics#15049](https://github.com/ultralytics/ultralytics/issues/15049)

### 20.3 `ultralytics` 包冲突

**问题**：使用 `pip install ultralytics`（官方包）会覆盖 YOLO-Master Fork 版本。

**解决方案**：
- **不要**执行 `pip install ultralytics`
- **必须**使用 `pip install -e .`（在 YOLO-Master 目录下）安装 Fork 版本
- 验证方式：`pip show ultralytics` 应显示本地路径而非 PyPI

### 20.4 依赖冲突

**解决方案**：
```powershell
# 1. 创建干净环境
conda create --name yolo-master-clean python=3.10 -y

# 2. 按顺序安装依赖
conda activate yolo-master-clean
conda install pytorch torchvision torchaudio pytorch-cuda=13.0 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -e .
```

### 20.5 环境变量问题

**解决方案**：
```powershell
# 检查 CUDA 路径
$env:CUDA_PATH

# 设置环境变量（临时）
$env:CUDA_PATH = "E:\Apps\Codes\Conda\envs\yolo-master\Library"

# 永久设置（系统环境变量）
[System.Environment]::SetEnvironmentVariable('CUDA_PATH', 'E:\Apps\Codes\Conda\envs\yolo-master\Library', 'User')
```

---

## 二十一、性能优化建议

### 21.1 训练加速

```python
# 1. 启用混合精度训练（FP16）
model.train(amp=True)

# 2. 使用多进程数据加载
model.train(workers=4)

# 3. 启用梯度累积（等效大 batch）
model.train(batch=2, accumulate=2)  # 等效 batch=4

# 4. 使用 bitsandbytes 8-bit 优化器（节省显存）
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)

# 5. 使用 xformers 加速注意力
# torch 2.9+ 已支持 scaled_dot_product_attention，无需额外配置
```

### 21.2 显存监控

```powershell
# 实时监控显存
nvidia-smi

# 或使用 PowerShell 循环
while ($true) { nvidia-smi; Start-Sleep -Seconds 1; Clear-Host }
```

### 21.3 训练中断恢复

```python
# 从 checkpoint 恢复训练
model.train(resume=True)
```

---

## 二十二、总结

✅ **环境配置完成**：`yolo-master` 环境已正确配置。

**关键统计**：
- **总包数**：127 个
- **Python**：3.10.9
- **PyTorch**：2.9.1+cu130
- **CUDA**：13.0
- **ultralytics**：8.3.240（editable，YOLO-Master Fork）

**关键要点**：
- 使用 `pip install -e .` 安装 YOLO-Master Fork 版本，**不要**用 `pip install ultralytics`
- 避免 PyTorch 2.4.0（Windows 下有 CPU 报错）
- 通过 PowerShell profile 自动加载 conda hook
- 复制 Library 目录解决 CUDA 依赖问题
- 采用混合精度和小 batch 策略优化显存使用

**下一步**：
- 验证环境配置（见 §18）
- 开始 Issue #54 实验任务（详见 [Disscusion/05-任务1-三变体训练.md](../Disscusion/05-任务1-三变体训练.md)）
- 定期备份环境配置（见 §19.2）

---

## 附：依赖来源参考

- `Codes/YOLO-Master/pyproject.toml`：YOLO-Master 官方包定义（[project] dependencies + [project.optional-dependencies]）
- `Codes/YOLO-Master/requirements.txt`：pip 安装用精简清单
- `Codes/YOLO-Master/setup.py`：不存在，已迁移到 pyproject.toml（PEP 621）
- 实际环境：`conda activate yolo-master && pip list`（共 127 个包）

> **文档维护**：本文件 §二~§十五 包列表来自 `pip list` 实际输出，与 `requirements.txt` + `pyproject.toml` 一致。每次环境变更后请更新。