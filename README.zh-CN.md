# HAE-Net

论文：[HAE-Net: Haze-Aware Remote Sensing Object Detection With Local–Global Defogging](https://doi.org/10.1109/JSTARS.2026.3676569)，IEEE JSTARS，2026。

流程：**加雾合成数据 → 去雾 → YOLO 检测**。三个项目的源码原样保留，空目录用 `.gitkeep` 保留。

## 准备环境与数据

```bash
git clone https://github.com/iliaohai/HAE-Net.git
cd HAE-Net
conda create -n hae-net python=3.12 -y
conda activate hae-net
pip install torch torchvision
pip install -r DehazeXL-main/requirements.txt
pip install -e ./yolo12
```

按本机 CUDA 安装对应的 PyTorch。到 [Release](https://github.com/iliaohai/HAE-Net/releases/tag/hae-dior-v1) 下载全部 `HAE-DIOR-part*.tar.gz`、`HAE-Net-weights.tar.gz`、`SHA256SUMS`，放到仓库根目录执行：

```bash
sha256sum -c SHA256SUMS
for f in HAE-DIOR-part*.tar.gz HAE-Net-weights.tar.gz; do tar -xzf "$f"; done
```

HAE-DIOR 共 23,463 张图像、20 类、192,472 个目标；训练/验证/测试分别为 5,862 / 5,863 / 11,738 张。解压后位于 `yolo12/datasets/HAE-DIOR/`，包含图像和 YOLO 标签。各分卷独立解压，数据统计与校验值见 `DATASET_MANIFEST.json`。

## 1. 合成数据集制作：加雾

将清晰 DIOR 图像放入 `add_hazy/input/`：

```bash
cd add_hazy
python add_hazy.py
# 可选：python add_hazy_level.py，生成六档雾浓度
cd ..
```

结果保存在 `add_hazy/output/`。随机加雾参数为 β∈[0.06, 0.10]、大气光∈[0.6, 1.0]。完整数据集按 train/val/test 分别处理，保留原文件名和标签，整理为 `images/{train,val,test}` 与 `labels/{train,val,test}`。已有 HAE-DIOR 可直接使用，无需重新生成。

## 2. 二阶段模型

### 2.1 去雾

可直接使用 Release 中的 `DehazeXL-main/checkpoints/best.pth`。对三个划分分别去雾：

```bash
cd DehazeXL-main
for split in train val test; do
  python test_all.py --test_img ../yolo12/datasets/HAE-DIOR/images/$split \
    --model_path ./checkpoints/best.pth \
    --save_dir ../yolo12/datasets/HAE-DIOR-Dehazing/images/$split \
    --uni_size_w 768 --uni_size_h 768 --nor
done
cd ..
cp -r yolo12/datasets/HAE-DIOR/labels/. yolo12/datasets/HAE-DIOR-Dehazing/labels/
```

若需训练去雾模型，另备配对数据到 `DehazeXL-main/datasets/8kdehaze_mini/{clear,cloud_L1,cloud_L2,cloud_L3,cloud_L4}/`，对应文件同名，图像尺寸须满足 2048 像素裁剪。此训练数据不包含在 HAE-DIOR 发布包中。

```bash
cd DehazeXL-main
python train.py --data_dir ./datasets/8kdehaze_mini --save_dir ./checkpoints \
  --lr 0.000005 --batch_size 1 --epochs 30240 --nor
cd ..
```

### 2.2 YOLO 检测

在仓库根目录生成单独的数据配置，不改原文件：

```bash
python - <<'PY'
from pathlib import Path
import yaml
c = yaml.safe_load(Path('yolo12/data.yaml').read_text())
c['path'] = str(Path('yolo12/datasets/HAE-DIOR-Dehazing').resolve())
Path('hae-dior-dehazed.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY
cd yolo12
yolo detect train model=ultralytics/cfg/models/11/yolo11s.yaml \
  pretrained=pre-models/yolo11s.pt data=../hae-dior-dehazed.yaml \
  imgsz=640 epochs=2000 batch=8 workers=2 device=0 optimizer=SGD amp=False
yolo detect val model=runs/detect/train/weights/best.pt \
  data=../hae-dior-dehazed.yaml split=test imgsz=640
yolo detect predict model=runs/detect/train/weights/best.pt \
  source=datasets/HAE-DIOR-Dehazing/images/test imgsz=640
```

按显存调整 batch，验证/推理使用训练输出的实际权重路径。当前代码启用 SDIoU，论文描述为 NWD；本次按原代码发布，未替换损失函数，未验证论文指标复现。

数据与权重通过 Release 提供；其他数据集、训练日志、缓存和崩溃转储不进入 Git 仓库。保留原项目授权声明，检测代码许可见 `yolo12/LICENSE`，衍生数据遵守原 DIOR 数据集条款。完整论文引用见 [README](README.md#citation)。
