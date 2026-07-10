#!/usr/bin/env python3
"""U-Net Few-Shot 训练 + 评估（Baseline）

用法:
    python train.py --shots 1,5,10,20,100,160 --train-dir ../../data/train --test-dir ../../data/test
"""
import sys, os, argparse, json, random
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm


class YOLODataset(Dataset):
    def __init__(self, data_dir, img_size=512, num_samples=None, seed=42):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
        files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(exts)])
        valid = [f for f in files if (self.data_dir / (Path(f).stem + ".txt")).exists()]
        print(f"[U-Net Dataset] {len(valid)} 有效对", flush=True)
        if num_samples and num_samples < len(valid):
            random.seed(seed); torch.manual_seed(seed)
            self.files = random.sample(valid, num_samples)
        else:
            self.files = valid
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img = cv2.imread(str(self.data_dir / fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        txt = self.data_dir / (Path(fn).stem + ".txt")
        if txt.exists():
            with open(txt) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) < 3: continue
                    coords = np.array([float(x) for x in p[1:]])
                    coords[0::2] *= w; coords[1::2] *= h
                    cv2.fillPoly(mask, [coords.reshape(-1,2).astype(np.int32)], color=255)
        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].float().unsqueeze(0) / 255.0


def train_unet(train_dir, test_dir, num_samples, output_base="unet_experiments",
               img_size=512, batch_size=4, num_epochs=100, lr=1e-4, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_base) / f"unet_{num_samples}shot_{ts}"
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ds = YOLODataset(train_dir, img_size=img_size, num_samples=num_samples, seed=seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"[U-Net] 创建模型 (ResNet34), {num_samples} 样本, {img_size}x{img_size}", flush=True)
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_dice = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 简单验证：在训练集上算 Dice
        model.eval()
        dice_sum = 0; n = 0
        with torch.no_grad():
            for images, masks in dl:
                pred = (torch.sigmoid(model(images.to(device))) > 0.5).float()
                inter = (pred * masks.to(device)).sum()
                d = (2*inter+1e-6)/(pred.sum()+masks.sum()+1e-6)
                dice_sum += d.item(); n += 1
        avg_dice = dice_sum / n if n > 0 else 0
        model.train()

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), ckpt_dir / "best_model_dice.pth")

        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(dl):.4f} dice={avg_dice:.4f}", flush=True)

    torch.save(model.state_dict(), ckpt_dir / "final_model.pth")
    print(f"[U-Net {num_samples}shot] 完成! best_dice={best_dice:.4f}", flush=True)
    return {"best_dice": best_dice, "exp_dir": str(exp_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", default="1,5,10,20,100,160")
    parser.add_argument("--train-dir", default=str(ROOT / "data/train"))
    parser.add_argument("--test-dir", default=str(ROOT / "data/test"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/output/unet"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    shots = [int(s.strip()) for s in args.shots.split(",")]
    print(f"U-Net Few-Shot: {shots}")

    for n in shots:
        try:
            train_unet(args.train_dir, args.test_dir, n, args.output_dir,
                       img_size=args.img_size, batch_size=args.batch_size,
                       num_epochs=args.epochs, lr=args.lr)
        except Exception as e:
            print(f"  ✗ {n}-shot 失败: {e}", flush=True)


if __name__ == "__main__":
    main()
