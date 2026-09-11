# F11 · 使用 AutoDL 云端算力跑 c-router-kd 真基线 —— 完整命令行记录

> 适用场景:F11 课题 `c-router-kd` 真基线训练。本地 5060 Laptop 跑得太慢(45 min/epoch),改用 AutoDL RTX 5090 实例(90 GB 内存 / 25 核 Xeon Platinum 8470Q / 32 GB 显存)。
>
> 文档位置:`Practice/Files/Docs/my-analysis/09-AutoDL云端部署c-router-kd完整命令行.md`
>
> 创建时间:2026-09-03 ~ 2026-09-04(本轮部署)
>
> 相关文档:`06-F11全部任务分析与设计.md`(任务定义)、`07a-F11过程-状态与规范.md`(→ 07b/07c/07d,过程记录)、`08a-F11结果-总览与结论.md`(→ 08b/08c/08d,结果汇总)

---

## 1. 机器规格

| 项 | 值 |
|----|----|
| 平台 | AutoDL |
| 镜像 | Ultralytics YOLO11(自带 Python 3.8,**需要新建 conda env**) |
| GPU | RTX 5090(实际配置,虽然选的是 4090 但分配到了 5090) |
| 显存 | 32 GB |
| CPU | 25 核 Xeon Platinum 8470Q |
| 内存 | 90 GB |
| 系统盘 | 50 GB 数据盘 |
| SSH | `ssh -p 26076 root@connect.westc.seetacloud.com`,密码 `<见 AutoDL 控制台>` |
| 数据盘路径 | `/root/autodl-tmp/`(主用) |

---

## 2. 完整命令行(从 0 到训练启动)

### 2.1 SSH 登录 + 接受主机密钥

```bash
# 本地 PowerShell 第一次连会提示主机验证
ssh -p 26076 root@connect.westc.seetacloud.com
# 看到 "Are you sure you want to continue connecting" → 输入 yes

# 或者一次性接受
ssh -p 26076 -o StrictHostKeyChecking=accept-new root@connect.westc.seetacloud.com
```

### 2.2 初始化 conda shell(关键,镜像自带的 Python 3.8 不是我们要的)

```bash
# 镜像默认 conda 在 /root/miniconda3
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo-master 2>/dev/null || conda create -n yolo-master python=3.10 -y
conda activate yolo-master
python -V   # 期望:Python 3.10.x
```

### 2.3 装 PyTorch + CUDA 13.0 + ultralytics

```bash
# PyTorch CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# ultralytics(固定到 F11 兼容版本)
pip install ultralytics==8.4.101
```

### 2.4 上传 F11 代码(本地打包 + SCP 上传)

**本地 PowerShell 打包**:

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\Practice\Code\YOLO-Master"
# 排除 runs/、__pycache__/、.git/ 等冗余
tar -czf YOLO-Master_f11.tar.gz --exclude=./runs --exclude=./.git --exclude=./__pycache__ .
Get-Item YOLO-Master_f11.tar.gz | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
```

**上传 + 解压(SSH)**:

```bash
# 本地 PowerShell
scp -P 26076 -C "YOLO-Master_f11.tar.gz" root@connect.westc.seetacloud.com:/root/autodl-tmp/

# SSH
cd /root/autodl-tmp
tar -xzf YOLO-Master_f11.tar.gz
ls   # 看到 ultralytics/ scripts/ configs/ 等就是对的
```

### 2.5 验证 F11 模块导入

```bash
cd /root/autodl-tmp
python -c "
import sys
sys.path.insert(0, '/root/autodl-tmp')
from ultralytics.nn.foundation.routing import FoundationTeacherRouter
from ultralytics.nn.foundation.losses import RouterKDLoss
from ultralytics.nn.foundation.teachers import SigLIP2Teacher
print('✅ FoundationTeacherRouter, RouterKDLoss, SigLIP2Teacher 全部 OK')
"
```

**期望输出**:`✅ FoundationTeacherRouter, RouterKDLoss, SigLIP2Teacher 全部 OK`

### 2.6 下载 SigLIP2 教师模型(关键,云端直连 HF 经常卡)

```bash
export HF_ENDPOINT=https://hf-mirror.com
source /etc/network_turbo 2>/dev/null   # AutoDL 学术加速

# 先下 AutoModel
python -c "from transformers import AutoModel; AutoModel.from_pretrained('google/siglip2-base-patch16-512'); print('Model OK')"

# 再下 AutoProcessor
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('google/siglip2-base-patch16-512'); print('Processor OK')"

# 验证(应该 6 个文件,~1.5 GB)
ls -lh ~/.cache/huggingface/hub/models--google--siglip2-base-patch16-512/snapshots/*/
# 期望看到:
#   config.json
#   model.safetensors          (~1.4G)
#   preprocessor_config.json
#   tokenizer.json
#   tokenizer_config.json
#   special_tokens_map.json
```

**踩坑记录**:
- `HF_ENDPOINT` 必须在 SSH 这一层 `export`,**不能写在 `python -c "..."` 里**,否则下载器拿不到
- `nohup` 后台下载时,`os.environ['HF_ENDPOINT']='...'` 在 `from transformers` **之后**赋值 = 镜像不生效
- 第一次下载如果 `peer closed`,是因为被分配到 `us.aws.cdn.hf.co` 美国边缘节点限速,**配 hf-mirror + 网络加速**解决

### 2.7 准备 VisDrone 数据集(本地过滤 + zip 上传)

**本地 PowerShell 过滤(去掉冗余的 .npy)**:

```powershell
Set-Location "G:\Codes\OpenSource\Rhino-bird\DATASETS"

New-Item -ItemType Directory -Force "VisDrone_min" | Out-Null
Copy-Item -Recurse -Force "VisDrone\labels" "VisDrone_min\labels"
Get-ChildItem "VisDrone\images" -Directory | ForEach-Object {
    $dst = "VisDrone_min\images\$($_.Name)"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item "$($_.FullName)\*.jpg" $dst
}

# 看结果
$total = (Get-ChildItem VisDrone_min -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host "VisDrone_min total: $([math]::Round($total/1GB, 2)) GB"
Write-Host "Files: $((Get-ChildItem VisDrone_min -Recurse -File | Measure-Object).Count)"
# 期望:~1.5 GB,14040 文件(7019 jpg + 7019 txt + 2 cache)

# 打 zip
Compress-Archive -Path VisDrone_min -DestinationPath VisDrone_min.zip -Force
Get-Item VisDrone_min.zip | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
```

**上传 + 解压(SSH)**:

```bash
# 本地 PowerShell
scp -P 26076 -C "VisDrone_min.zip" root@connect.westc.seetacloud.com:/root/autodl-tmp/

# SSH
cd /root/autodl-tmp
unzip -t VisDrone_min.zip | tail -3   # 测试完整性
unzip -q VisDrone_min.zip -d /root/autodl-tmp/
# 注意:解压到 autodl-tmp/ 顶层,会生成 VisDrone_min/,然后手动挪到 data/
mv /root/autodl-tmp/VisDrone_min /root/autodl-tmp/data/VisDrone_min
rm VisDrone_min.zip

# 验证
ls /root/autodl-tmp/data/VisDrone_min/
ls /root/autodl-tmp/data/VisDrone_min/images/
ls /root/autodl-tmp/data/VisDrone_min/labels/
# 期望文件数
echo "jpg train: $(ls /root/autodl-tmp/data/VisDrone_min/images/train/*.jpg 2>/dev/null | wc -l)"   # 6471
echo "jpg val:   $(ls /root/autodl-tmp/data/VisDrone_min/images/val/*.jpg 2>/dev/null | wc -l)"     # 548
echo "npy (应该 0): $(find /root/autodl-tmp/data/VisDrone_min -name '*.npy' | wc -l)"               # 0
```

### 2.8 写 VisDrone.yaml + 启动训练

```bash
# 写 yaml
cat > /root/autodl-tmp/data/VisDrone.yaml <<'EOF'
path: /root/autodl-tmp/data/VisDrone_min
train: images/train
val: images/val
test: images/val
nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
EOF

# 测试 yaml 解析
python -c "
from ultralytics.utils import YAML
yaml = YAML.load('/root/autodl-tmp/data/VisDrone.yaml')
print('YAML OK:')
for k, v in yaml.items():
    print(f'  {k}: {v}')
"
```

### 2.9 启动 c-router-kd 真基线 100 epoch(b12 + w24 + ram)

```bash
cd /root/autodl-tmp
mkdir -p runs/real/t4_ablation/real_100ep logs/real/t4_ablation

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
TS=$(date +%Y%m%d_%H%M%S)

nohup python scripts/compare_f11_ablation.py --train \
    --groups c --epochs 100 --imgsz 640 \
    --batch 12 --device 0 \
    --data /root/autodl-tmp/data/VisDrone.yaml \
    --project /root/autodl-tmp/runs/real/t4_ablation/real_100ep \
    --name c-router-kd_b12_w24_ram \
    --workers 24 --cache ram \
    --seed 42 \
    > /root/autodl-tmp/logs/real/t4_ablation/T4R_c_b12_w24_ram_${TS}.log 2>&1 &

echo "训练 PID: $!"
echo "TS=$TS"
sleep 30
tail -25 /root/autodl-tmp/logs/real/t4_ablation/T4R_c_b12_w24_ram_${TS}.log
```

**关键参数解释**:

| 参数 | 值 | 说明 |
|------|---|------|
| `--groups c` | 只跑 c-router-kd(对照实验只做 1 组) | |
| `--epochs 100` | 真基线 100 epoch | |
| `--imgsz 640` | 标准输入尺寸 | |
| `--batch 12` | 5090 32G 显存可吃满 | |
| `--workers 24` | 25 核 CPU 几乎跑满 | |
| `--cache ram` | 数据集 1.6 GB 进 RAM,后续 epoch 不再读盘 | |
| `--seed 42` | 固定随机种子,结果可复现 | |
| `HF_HUB_OFFLINE=1` | 训练时不再访问 HF,只读本地缓存 | |
| `PYTHONUNBUFFERED=1` | 日志实时输出,不缓冲 | |

---

## 3. 监控命令(训练过程中随时跑)

```bash
# 1) 最新日志
ls -t /root/autodl-tmp/logs/real/t4_ablation/*.log | head -1 | xargs tail -f

# 2) GPU 状态
nvidia-smi

# 3) 训练进程
ps aux | grep compare_f11 | grep -v grep

# 4) 只看关键行(epoch / metrics / errors)
grep -E "Epoch|all\s+[0-9]|Error|Traceback" /root/autodl-tmp/logs/real/t4_ablation/*.log | tail -20
```

---

## 4. 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `/root/autodl-tmp/logs/real/t4_ablation/T4R_c_b12_w24_ram_<TS>.log` |
| 权重 | `/root/autodl-tmp/runs/real/t4_ablation/real_100ep/c-router-kd_b12_w24_ram/weights/` |
| best | `.../weights/best.pt` |
| last | `.../weights/last.pt` |
| 结果 CSV | `.../results.csv` |
| 训练曲线 | `.../results.png` |

---

## 5. 常见踩坑与修复

### 5.1 SSH 主机验证失败

```
Host key verification failed.
```

**修复**:
```bash
# 第一次连会问,输入 yes 接受
# 或者
ssh -p 26076 -o StrictHostKeyChecking=accept-new root@connect.westc.seetacloud.com
```

### 5.2 conda activate 失败

```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
```

**修复**:
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo-master
```

### 5.3 tar.gz 传输损坏

```
gzip: stdin: unexpected end of file
tar: Unexpected EOF in archive
```

**修复**:重传(确认本地大小 + SSH 上大小一致)

```bash
# 本地
Get-Item YOLO-Master_f11.tar.gz | Select-Object Length
# SSH
ls -l /root/autodl-tmp/YOLO-Master_f11.tar.gz
# 两者应一致
```

### 5.4 SigLIP2 下载被 `peer closed` 切断

```
peer closed connection without sending complete message body
```

**修复**:`HF_ENDPOINT` 必须在外层 `export`,**不能写在 `python -c "..."` 里**

```bash
# ❌ 错的(镜像不生效)
nohup python -c "import os; os.environ['HF_ENDPOINT']='...'; from transformers import ..."

# ✅ 对的(SSH 这一层就 export)
export HF_ENDPOINT=https://hf-mirror.com
source /etc/network_turbo 2>/dev/null
nohup python -c "from transformers import AutoModel; AutoModel.from_pretrained('...')"
```

### 5.5 解压路径错(VisDrone_min 跑到 autodl-tmp 顶层)

```bash
# 解压到 /root/autodl-tmp/(而不是 data/),需要手动挪
mv /root/autodl-tmp/VisDrone_min /root/autodl-tmp/data/VisDrone_min
```

### 5.6 PowerShell / Git Bash 混用路径

```powershell
# ❌ PowerShell 里跑 Git Bash 风格
Set-Location /g/Codes/...

# ✅ PowerShell 用 Windows 路径
Set-Location "G:\Codes\OpenSource\Rhino-bird\DATASETS"
```

---

## 6. 数据集上传对比(选哪种)

| 方式 | 命令 | 优点 | 缺点 |
|------|------|------|------|
| **zip + SCP** | `scp -P 26076 -C VisDrone_min.zip ...` | **稳定,断点可重传** | 打包占空间 |
| SCP 散文件 | `scp -P 26076 -r -C VisDrone ...` | 直接 | 1.5 GB 散文件慢,断了全重传 |
| rsync | `rsync -avz --progress ...` | 断点续传 | Windows 默认没装 |
| AutoDL OSS 网页 | 浏览器拖拽 | 简单 | 不支持文件夹,文件数有限 |
| JupyterLab | 浏览器拖拽 | 直观 | 大量文件慢 |

**推荐**:**zip + SCP**,本次部署用的就是这个,稳。

---

## 7. 时间预估(本次部署实测)

| 步骤 | 时间 |
|------|------|
| SSH 登录 + conda 创建 + 装包 | 5~10 min |
| F11 代码 tar 上传 + 解压 | 2~3 min |
| SigLIP2 下载 | 1~3 min |
| VisDrone 过滤 + zip + 上传 | 10~15 min |
| yaml 写入 + 启动训练 | < 1 min |
| **总部署时间** | **20~30 min** |
| c-router-kd 100 epoch 训练 | **~4~6 小时**(5090,12 batch,w24) |

vs 本地 5060 Laptop:**3 天 → 1 天**(提速 ~3 倍)。

---

## 8. 后续:A / B 两组对照(已在本地完成)

A / B 两组 100 epoch 已在本地 RTX 5060 完成(见 08c §7),云端方案保留作为全量 C 组恢复的备用算力:

| 组 | 配置 | 训练命令差异 |
|----|------|-------------|
| **A-baseline** | 不加 KD | `--groups a` |
| **B-repr-kd** | 加表征 KD | `--groups b` |
| **C-router-kd** | 加 Router KD | `--groups c` |

每组用同样的 VisDrone 数据集 + 同样的 batch / workers / seed,只换 yaml 中的 KD 模块开关。

---

## 10. 三组最大性能跑(A / B / C 串行,5090 单卡)

> 单卡只能串行跑三组。同一台机器(25 核 / 90 GB / 5090 32G)上,每次用同一份 `compare_f11_ablation.py --train` 改 `--groups` 即可。

### 10.1 一次性启动 a/b/c 三组(后台串行,关 SSH 也不停)

```bash
cd /root/autodl-tmp
mkdir -p runs/real/t4_ablation/real_100ep logs/real/t4_ablation

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cat > /root/autodl-tmp/scripts/run_abc_100ep.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd /root/autodl-tmp

DATA=/root/autodl-tmp/data/VisDrone.yaml
PROJ=/root/autodl-tmp/runs/real/t4_ablation/real_100ep
LOGD=/root/autodl-tmp/logs/real/t4_ablation
mkdir -p "$PROJ" "$LOGD"

for GRP in a b c; do
    TS=$(date +%Y%m%d_%H%M%S)
    NAME="${GRP}-f11_b12_w24_ram"
    LOG="$LOGD/T4R_${GRP}_b12_w24_ram_${TS}.log"
    echo "=========================================="
    echo "[$(date)] 启动 ${GRP} → ${LOG}"
    echo "=========================================="

    python scripts/compare_f11_ablation.py --train \
        --groups ${GRP} --epochs 100 --imgsz 640 \
        --batch 12 --device 0 \
        --data ${DATA} \
        --project ${PROJ} \
        --name ${NAME} \
        --workers 24 --cache ram \
        --seed 42 \
        > ${LOG} 2>&1

    echo "[$(date)] ${GRP} 完成 → ${LOG}"
done

echo "[$(date)] 三组全部完成"
EOF
chmod +x /root/autodl-tmp/scripts/run_abc_100ep.sh

# 后台启动,关 SSH 也不停
nohup bash /root/autodl-tmp/scripts/run_abc_100ep.sh \
    > /root/autodl-tmp/logs/real/t4_ablation/abc_runner.log 2>&1 &

echo "abc runner PID: $!"
sleep 30
tail -25 /root/autodl-tmp/logs/real/t4_ablation/abc_runner.log
```

**关键**:`--groups a b c` 改成循环变量,三组自动跑完一组再下一组。

### 10.2 单组跑(只看其中一组)

```bash
cd /root/autodl-tmp

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
TS=$(date +%Y%m%d_%H%M%S)

# 替换 GROUP 为 a / b / c 其中一个
GROUP=c

nohup python scripts/compare_f11_ablation.py --train \
    --groups ${GROUP} --epochs 100 --imgsz 640 \
    --batch 12 --device 0 \
    --data /root/autodl-tmp/data/VisDrone.yaml \
    --project /root/autodl-tmp/runs/real/t4_ablation/real_100ep \
    --name ${GROUP}-f11_b12_w24_ram \
    --workers 24 --cache ram \
    --seed 42 \
    > /root/autodl-tmp/logs/real/t4_ablation/T4R_${GROUP}_b12_w24_ram_${TS}.log 2>&1 &

echo "PID: $!  GROUP=${GROUP}  TS=${TS}"
sleep 30
tail -25 /root/autodl-tmp/logs/real/t4_ablation/T4R_${GROUP}_b12_w24_ram_${TS}.log
```

### 10.3 最大性能参数(为什么是 batch=12 / workers=24 / cache=ram)

| 参数 | 值 | 依据 |
|------|----|------|
| `--batch 12` | 5090 32G 显存饱和点 | 8 batch 浪费、16 batch 容易 OOM(SigLIP2 teacher 额外占显存) |
| `--workers 24` | 25 核 CPU 满载 | 25 核机留 1 核给主进程;16 也稳但稍慢,24 接近上限 |
| `--cache ram` | 数据集 ~5.3 GB 进 RAM | 90 GB 内存充足,后续 epoch 不再读盘,提速 20~30% |
| `--device 0` | 单卡 5090 | 只有 1 张卡 |
| `--imgsz 640` | VisDrone 标准 | 不动 |
| `--seed 42` | 固定 | a/b/c 三组可比 |
| `--epochs 100` | 真基线 | 不动 |

**预期耗时**:5090 上每组 100 epoch 约 4~6 小时,a+b+c 共约 **12~18 小时**。

### 10.4 三组进度监控(实时看 a/b/c 当前跑到第几组)

```bash
# 1) 哪个组在跑(从 abc_runner.log 看)
tail -5 /root/autodl-tmp/logs/real/t4_ablation/abc_runner.log

# 2) 当前组 epoch 进度
LATEST=$(ls -t /root/autodl-tmp/logs/real/t4_ablation/T4R_*.log | head -1)
echo ">>>当前日志: $LATEST"
grep -E "Epoch|all\s+[0-9]|Speed:|Results saved" "$LATEST" | tail -10

# 3) 三组是否都已完成
ls /root/autodl-tmp/runs/real/t4_ablation/real_100ep/

# 4) 实时滚动
tail -f "$LATEST"
# Ctrl+C 退出(abc runner 不受影响)
```

### 10.5 三组都跑完后,统一汇总 mAP

```bash
echo "===== A / B / C 最终 mAP ====="
for GRP in a b c; do
    CSV="/root/autodl-tmp/runs/real/t4_ablation/real_100ep/${GRP}-f11_b12_w24_ram/results.csv"
    if [ -f "$CSV" ]; then
        echo "----- $GRP -----"
        tail -1 "$CSV"
    else
        echo "----- $GRP : 未找到 $CSV -----"
    fi
done
```

---

## 11. 文档版本

- v1:2026-09-04,首次部署完整命令行记录

