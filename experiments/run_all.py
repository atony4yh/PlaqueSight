#!/usr/bin/env python3
"""PlaqueSight Few-Shot 批量实验

用法:
    python run_all.py                          # 默认 1/5/10/20/100/160 shot
    python run_all.py --shots 1,5              # 指定 shot 数量
    python run_all.py --shots 5 --epochs 50    # 自定义 epoch
"""
import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sam"))
sys.path.insert(0, str(ROOT / "dinov3"))

import torch
from segment_anything import sam_model_registry
from plaquesight.model import PlaqueSightModel, ConvPromptAdapter
from plaquesight.dataset import FewShotPlaqueDataset
from plaquesight.train import train_plaquesight
from plaquesight.evaluate import evaluate_plaquesight


def run_experiment(train_dir, test_dir, num_samples, output_base,
                   sam_ckpt, dino_weights, dino_local_dir,
                   image_size=1024, batch_size=2, num_epochs=100,
                   learning_rate=1e-4, random_seed=42, device="cuda"):
    """运行单个 few-shot 实验"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"plaquesight_{num_samples}shot_{ts}"
    exp_dir = Path(output_base) / exp_name
    ckpt_dir = exp_dir / "checkpoints"
    test_dir_out = exp_dir / "test_results"
    for d in [exp_dir, ckpt_dir, test_dir_out]:
        d.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config = {
        "train_data_dir": train_dir, "test_data_dir": test_dir,
        "num_samples": num_samples, "image_size": image_size,
        "batch_size": batch_size, "num_epochs": num_epochs,
        "learning_rate": learning_rate, "random_seed": random_seed,
        "timestamp": ts,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}\n  实验: {exp_name}\n{'='*60}", flush=True)

    print("[1/4] 加载 DINOv3 ...", flush=True)
    dinov3 = torch.hub.load(str(dino_local_dir), "dinov3_vith16plus",
                            pretrained=False, source="local")
    dinov3.load_state_dict(torch.load(dino_weights, map_location=device), strict=False)
    dinov3.to(device).eval()
    for p in dinov3.parameters():
        p.requires_grad = False

    print("[2/4] 加载 SAM ...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    sam.to(device).eval()
    for p in sam.parameters():
        p.requires_grad = False

    print("[3/4] 创建模型 ...", flush=True)
    adapter = ConvPromptAdapter(dino_embed_dim=dinov3.embed_dim).to(device)
    model = PlaqueSightModel(dinov3, sam, adapter)

    print(f"[4/4] 训练 ({num_samples} samples) ...", flush=True)
    train_dataset = FewShotPlaqueDataset(
        data_dir=train_dir, image_size=image_size,
        num_samples=num_samples, random_seed=random_seed)

    history, best_dice = train_plaquesight(
        model=model, train_dataset=train_dataset, device=device,
        batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)

    torch.save(model.adapter.state_dict(), ckpt_dir / "adapter_weights.pth")
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n评估 ...", flush=True)
    test_results = evaluate_plaquesight(
        model=model, test_data_dir=test_dir, device=device,
        output_dir=str(test_dir_out), image_size=image_size)

    return {
        "experiment": exp_name,
        "iou": test_results["pixel_metrics"]["iou"],
        "dice": test_results["pixel_metrics"]["dice"],
        "mAP50": test_results["instance_metrics"]["mAP@50"],
        "mAP": test_results["instance_metrics"]["mAP@50:95"],
    }


def main():
    parser = argparse.ArgumentParser(description="PlaqueSight Few-Shot 批量实验")
    parser.add_argument("--shots", default="1,5,10,20,100,160",
                        help="shot 数量列表（逗号分隔）")
    parser.add_argument("--train-dir", default=str(ROOT / "data/train"))
    parser.add_argument("--test-dir", default=str(ROOT / "data/test"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/output"))
    parser.add_argument("--sam-ckpt", default=str(ROOT / "weights/sam_vit_h_4b8939.pth"))
    parser.add_argument("--dino-weights", default=str(ROOT / "weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"))
    parser.add_argument("--dino-local-dir", default=str(ROOT / "dinov3"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    shot_list = [int(s.strip()) for s in args.shots.split(",")]

    print(f"PlaqueSight Few-Shot 批量实验")
    print(f"  Shots: {shot_list}")
    print(f"  训练数据: {args.train_dir}")
    print(f"  测试数据: {args.test_dir}")
    print(f"  设备: {args.device}")

    results = {}
    for n_shot in shot_list:
        try:
            r = run_experiment(
                train_dir=args.train_dir, test_dir=args.test_dir,
                num_samples=n_shot, output_base=args.output_dir,
                sam_ckpt=args.sam_ckpt, dino_weights=args.dino_weights,
                dino_local_dir=args.dino_local_dir,
                image_size=args.image_size, batch_size=args.batch_size,
                num_epochs=args.epochs, learning_rate=args.lr,
                random_seed=args.seed, device=args.device)
            results[n_shot] = r
        except Exception as e:
            print(f"  ✗ {n_shot}-shot 失败: {e}", flush=True)
            results[n_shot] = {"error": str(e)}

    # 汇总
    print(f"\n{'='*80}")
    print(f"{'Shot':<8} {'IoU':<10} {'Dice':<10} {'mAP@50':<10} {'mAP':<10}")
    print("-" * 80)
    for n in shot_list:
        r = results.get(n, {})
        if "iou" in r:
            print(f"{n:<8} {r['iou']:<10.4f} {r['dice']:<10.4f} {r['mAP50']:<10.4f} {r['mAP']:<10.4f}")
        else:
            print(f"{n:<8} {'FAILED':<10}")
    print(f"{'='*80}")

    with open(Path(args.output_dir) / "experiment_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
