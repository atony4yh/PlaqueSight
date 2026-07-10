# PlaqueSight

**PlaqueSight: A data-efficient foundation model framework for precise segmentation and characterization of plaques**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

PlaqueSight fuses DINOv3's dense visual features with SAM's segmentation capability via a lightweight ConvPromptAdapter that bridges two frozen foundation models. Only ~330K trainable parameters are needed for high-precision plaque segmentation.

## Architecture

```
Input Image (1024×1024)
    │
    ├─→ DINOv3 (frozen) ──→ [1280, 64, 64]
    │                              │
    │                     ConvPromptAdapter (trainable)
    │                              │
    │                     [256, 64, 64] dense prompt
    │                              │
    └─→ SAM Encoder (frozen) ──→ SAM Decoder ──→ Segmentation Mask
```

## Quick Start

### Installation

```bash
git clone --recursive https://github.com/atony4yh/PlaqueSight.git
cd PlaqueSight
pip install -r requirements.txt
```

### Download Weights

See `weights/README.md` for instructions to download SAM, DINOv3, and PlaqueSight adapter weights into the `weights/` directory.

### Demo Inference

```bash
python demo/demo_infer.py --image path/to/image.jpg
```

Output: mask overlay visualization + plaque statistics.

## Few-Shot Training

```bash
python experiments/run_all.py --shots 1,5,10,20,100,160
```

## Data Format

YOLO segmentation format. Images and corresponding `.txt` label files reside in the same directory:

```
data/train/
├── image001.jpg
├── image001.txt    # class_id x1 y1 x2 y2 ... (normalized polygon coordinates)
├── image002.jpg
├── image002.txt
└── ...
```

## License

This project is licensed under Apache 2.0. SAM (Apache 2.0) and DINOv3 (Meta proprietary) retain their original licenses.
