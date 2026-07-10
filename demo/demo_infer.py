#!/usr/bin/env python3
"""PlaqueSight Demo：单张图片推理示例。

用法:
    python demo_infer.py --image path/to/image.jpg
    python demo_infer.py --image path/to/image.jpg --output result.jpg --threshold 0.5

前提:
    1. SAM 权重: weights/sam_vit_h_4b8939.pth (下载链接见 weights/README.md)
    2. DINOv3 权重: weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
    3. PlaqueSight adapter: weights/adapter_weights.pth (从 GitHub Release 下载)
"""
import argparse
import sys
import os
from pathlib import Path

import cv2
import torch
import numpy as np

# 将项目根目录加入 Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 确保能找到 sam 和 dinov3
sys.path.insert(0, str(ROOT / "sam"))
sys.path.insert(0, str(ROOT / "dinov3"))

from plaquesight.model import PlaqueSightModel, ConvPromptAdapter
from plaquesight.infer import predict_mask, analyze_plaques, draw_overlay


def load_model(adapter_path, sam_checkpoint, dino_weights, device="cuda"):
    """加载 PlaqueSight 模型"""
    from segment_anything import sam_model_registry

    print(f"[1/4] 加载 SAM (vit_h) ...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device).eval()
    for p in sam.parameters():
        p.requires_grad = False
    print("  ✓ SAM 加载完成")

    print(f"[2/4] 加载 DINOv3 ...")
    dinov3 = torch.hub.load(str(ROOT / "dinov3"), "dinov3_vith16plus",
                            pretrained=False, source="local")
    state = torch.load(dino_weights, map_location=device)
    dinov3.load_state_dict(state, strict=False)
    dinov3.to(device).eval()
    for p in dinov3.parameters():
        p.requires_grad = False
    print("  ✓ DINOv3 加载完成")

    print(f"[3/4] 加载 PlaqueSight Adapter ...")
    adapter = ConvPromptAdapter(dino_embed_dim=dinov3.embed_dim).to(device)
    adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    print("  ✓ Adapter 加载完成")

    print(f"[4/4] 组装模型 ...")
    model = PlaqueSightModel(dinov3, sam, adapter).to(device).eval()
    print(f"  ✓ 模型就绪，设备: {device}")
    return model


def main():
    parser = argparse.ArgumentParser(description="PlaqueSight 单张图片推理")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--output", default=None, help="输出图片路径（默认: 输入名_overlay.jpg）")
    parser.add_argument("--threshold", type=float, default=0.5, help="分割阈值 (0.1-0.9)")
    parser.add_argument("--min-area", type=int, default=50, help="最小菌斑面积 (px)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sam-checkpoint", default=str(ROOT / "weights/sam_vit_h_4b8939.pth"))
    parser.add_argument("--dino-weights", default=str(ROOT / "weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"))
    parser.add_argument("--adapter", default=str(ROOT / "weights/adapter_weights.pth"))
    args = parser.parse_args()

    # 检查输入
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"错误: 图片不存在: {img_path}")
        sys.exit(1)

    # 检查权重
    for p, name in [(args.sam_checkpoint, "SAM"), (args.dino_weights, "DINOv3"), (args.adapter, "Adapter")]:
        if not Path(p).exists():
            print(f"错误: {name} 权重不存在: {p}")
            print(f"  请参考 weights/README.md 下载")
            sys.exit(1)

    if args.output is None:
        args.output = str(img_path.parent / f"{img_path.stem}_overlay.jpg")

    print(f"图片: {img_path}")
    print(f"输出: {args.output}")
    print(f"设备: {args.device}")

    model = load_model(args.adapter, args.sam_checkpoint, args.dino_weights, args.device)

    print(f"\n推理中 ...")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"错误: 无法读取图片 {img_path}")
        sys.exit(1)

    mask = predict_mask(model, img_bgr, threshold=args.threshold, device=args.device)
    plaques = analyze_plaques(mask, min_area=args.min_area)
    overlay = draw_overlay(img_bgr, mask, plaques)

    cv2.imwrite(args.output, overlay)
    print(f"\n{'='*50}")
    print(f"检测到 {len(plaques)} 个菌斑")
    for p in plaques:
        print(f"  #{p['id']}: 面积={p['area_pixels']:.0f}px  "
              f"圆度={p['circularity']:.4f}  中心=({p['centroid']['x']:.0f},{p['centroid']['y']:.0f})")
    print(f"结果已保存: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
