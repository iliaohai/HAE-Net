# HAE-Net

Code and HAE-DIOR dataset for **HAE-Net: Haze-Aware Remote Sensing Object Detection With Local–Global Defogging**.

[Paper (IEEE JSTARS, 2026)](https://doi.org/10.1109/JSTARS.2026.3676569) · [Dataset & weights](https://github.com/iliaohai/HAE-Net/releases/tag/hae-dior-v1)

[简洁中文教程](README.zh-CN.md)

## Setup

```bash
git clone https://github.com/iliaohai/HAE-Net.git
cd HAE-Net
conda create -n hae-net python=3.12 -y
conda activate hae-net
pip install torch torchvision
pip install -r DehazeXL-main/requirements.txt
pip install -e ./yolo12
```

Install a PyTorch build matching your CUDA environment. Run commands below from the repository root unless a different directory is shown.

The original three projects are preserved: `add_hazy/` (haze synthesis), `DehazeXL-main/` (dehazing), and `yolo12/` (detection). Source code is released unchanged. Empty directories are retained with `.gitkeep`. Large datasets and weights are distributed through Releases; experiment outputs and unrelated datasets are not included in the Git checkout.

## 1. Synthetic dataset generation

Put clean DIOR images in `add_hazy/input/`, then run:

```bash
cd add_hazy
python add_hazy.py
# Optional: generate six haze levels instead.
# python add_hazy_level.py
cd ..
```

Hazy images are written to `add_hazy/output/`. The random haze script samples beta in `[0.06, 0.10]` and atmospheric light in `[0.6, 1.0]`. Keep the original image names and detection annotations. For a full dataset, process each split separately and organize results as `images/{train,val,test}` and `labels/{train,val,test}`.

### Download the released HAE-DIOR dataset

Download **all** `HAE-DIOR-part*.tar.gz`, `HAE-Net-weights.tar.gz`, and `SHA256SUMS` from [Releases](https://github.com/iliaohai/HAE-Net/releases/tag/hae-dior-v1) into the repository root. With GitHub CLI:

```bash
gh release download hae-dior-v1 -R iliaohai/HAE-Net \
  -p 'HAE-DIOR-part*.tar.gz' -p 'HAE-Net-weights.tar.gz' -p SHA256SUMS
sha256sum -c SHA256SUMS
for archive in HAE-DIOR-part*.tar.gz HAE-Net-weights.tar.gz; do
  tar -xzf "$archive"
done
```

The independent archives extract into the existing project layout. HAE-DIOR contains 23,463 images across 20 classes: 5,862 train, 5,863 validation, and 11,738 test images, with matching YOLO labels. See `DATASET_MANIFEST.json` for audited counts and checksums. Derived imagery remains subject to the original DIOR dataset terms.

## 2. Two-stage model

### Stage 1: Dehazing

Optional training: prepare paired, same-name images in `DehazeXL-main/datasets/8kdehaze_mini/{clear,cloud_L1,cloud_L2,cloud_L3,cloud_L4}/`. This training dataset is not part of HAE-DIOR; supply it separately. The existing training script uses 2048-pixel crops, so training images must be sufficiently large.

```bash
cd DehazeXL-main
python train.py --data_dir ./datasets/8kdehaze_mini \
  --save_dir ./checkpoints --lr 0.000005 --batch_size 1 --epochs 30240 --nor
cd ..
```

For inference, use the released `DehazeXL-main/checkpoints/best.pth`. Dehaze each detection split:

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

### Stage 2: YOLO detection

Create a separate dataset configuration with an absolute path; the original `data.yaml` and class order remain unchanged:

```bash
python - <<'PY'
from pathlib import Path
import yaml
config = yaml.safe_load(Path('yolo12/data.yaml').read_text())
config['path'] = str(Path('yolo12/datasets/HAE-DIOR-Dehazing').resolve())
Path('hae-dior-dehazed.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
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

Adjust `batch` to GPU memory and use the actual run directory printed by training. To train on hazy images directly, set the generated configuration's `path` to `yolo12/datasets/HAE-DIOR` instead.

**Implementation note:** this release preserves the supplied source exactly. Its active bounding-box loss in `yolo12/ultralytics/utils/loss.py` uses SDIoU; the paper describes NWD. No loss implementation has been substituted. Commands describe the released code and do not establish reproduction of the paper's reported scores.

## Citation

```bibtex
@article{liao2026haenet,
  author={Liao, Hai and Chen, Song and Liang, Yan and Zhou, Hang and Min, Fan},
  title={HAE-Net: Haze-Aware Remote Sensing Object Detection With Local--Global Defogging},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2026}, volume={19}, pages={14077--14094},
  doi={10.1109/JSTARS.2026.3676569}
}
```

Acknowledgments: the original DehazeXL, Ultralytics/YOLO, and DIOR projects. Existing third-party license notices are retained; see `yolo12/LICENSE` for the detector's AGPL-3.0 license.
