# PlaqueSight

**DINOv3 + SAM 融合的 Few-Shot 菌斑分割模型**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

PlaqueSight 将 DINOv3 的密集视觉特征与 SAM 的分割能力融合，通过一个轻量级 ConvPromptAdapter 桥接两个冻结的基础模型。仅需训练 ~330K 参数即可实现高精度菌斑分割。

## 架构

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

## 快速开始

### 安装

```bash
git clone --recursive https://github.com/atony4yh/PlaqueSight.git
cd PlaqueSight
pip install -r requirements.txt
```

### 下载权重

参考 `weights/README.md` 下载 SAM、DINOv3 和 PlaqueSight Adapter 权重到 `weights/` 目录。

### Demo 推理

```bash
python demo/demo_infer.py --image path/to/image.jpg
```

输出: 覆盖了 mask 的可视化图 + 菌斑统计信息。

## Few-Shot 实验

### PlaqueSight

```bash
python experiments/run_all.py --shots 1,5,10,20,100,160
```

### U-Net Baseline

```bash
python baselines/unet/train.py --shots 1,5,10,20,100,160
```

### YOLO Baseline

```bash
python baselines/yolo/train.py --shots 1,5,10,20,100,160
```

## 数据格式

使用 YOLO 分割格式。图片和对应的 `.txt` 标签文件放在同一目录：

```
data/train/
├── image001.jpg
├── image001.txt    # class_id x1 y1 x2 y2 ... (归一化多边形坐标)
├── image002.jpg
├── image002.txt
└── ...
```

## 实验结果

| Shot | IoU | Dice | mAP@50 | mAP@50:95 |
|------|-----|------|--------|-----------|
| 1 | 0.796 | 0.846 | - | - |
| 5 | 0.826 | 0.876 | - | - |
| 10 | 0.850 | 0.893 | - | - |
| 20 | 0.858 | 0.912 | - | - |
| 100 | 0.930 | 0.961 | - | - |
| 160 | - | - | - | - |

## 引用

```bibtex
@misc{plaquesight2025,
  title={PlaqueSight: Few-Shot Plaque Segmentation with DINOv3 and SAM},
  author={},
  year={2025},
}
```

## 许可证

本项目代码采用 Apache 2.0 许可证。SAM（Apache 2.0）和 DINOv3（Meta 专有许可）各保留其原始许可。
