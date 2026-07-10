"""PlaqueSight 训练函数"""
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import FewShotPlaqueDataset


class DiceBCELoss(nn.Module):
    """Dice + Binary Cross-Entropy 组合损失"""

    def forward(self, inputs, targets, smooth=1.0):
        inputs = torch.sigmoid(inputs)
        bce = nn.BCELoss()(inputs, targets)
        intersection = (inputs * targets).sum()
        dice = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce + dice


def train_plaquesight(model, train_dataset, device,
                      batch_size=2, num_epochs=100, learning_rate=1e-4):
    """训练 PlaqueSight 模型（仅训练 adapter）。

    返回:
        history: 包含 train_loss / train_dice / train_iou 的字典
        best_dice: 最佳 Dice 分数
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    optimizer = AdamW(model.adapter.parameters(), lr=learning_rate)
    loss_fn = DiceBCELoss()

    print(f"\n{'='*60}", flush=True)
    print(f"开始训练 | 样本数: {len(train_dataset)} | Epochs: {num_epochs} | 设备: {device}", flush=True)
    print(f"{'='*60}\n", flush=True)

    history = {'train_loss': [], 'train_dice': [], 'train_iou': []}
    best_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = epoch_dice = epoch_iou = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            predicted_masks, _ = model(images)
            loss = loss_fn(predicted_masks, masks)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_bin = (torch.sigmoid(predicted_masks) > 0.5).float()
                inter = (pred_bin * masks).sum()
                union = pred_bin.sum() + masks.sum() - inter
                iou = (inter + 1e-6) / (union + 1e-6)
                dice = (2 * inter + 1e-6) / (pred_bin.sum() + masks.sum() + 1e-6)

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            epoch_iou += iou.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item(), dice=dice.item())

        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_iou = epoch_iou / num_batches

        history['train_loss'].append(avg_loss)
        history['train_dice'].append(avg_dice)
        history['train_iou'].append(avg_iou)

        if avg_dice > best_dice:
            best_dice = avg_dice

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
                  f"Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}", flush=True)

    print(f"\n训练完成！最佳 Dice: {best_dice:.4f}", flush=True)
    return history, best_dice
