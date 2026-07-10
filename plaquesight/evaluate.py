"""PlaqueSight 评估函数"""
import os
import json
import time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import TestDataset


def calculate_pixel_metrics(pred_mask, gt_mask):
    """计算像素级指标: IoU, Dice, Precision, Recall"""
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    iou = inter / union if union > 0 else 0.0
    dice = (2.0 * inter) / (pred_bool.sum() + gt_bool.sum()) if (pred_bool.sum() + gt_bool.sum()) > 0 else 0.0
    precision = inter / pred_bool.sum() if pred_bool.sum() > 0 else 0.0
    recall = inter / gt_bool.sum() if gt_bool.sum() > 0 else 0.0
    return iou, dice, precision, recall


def calculate_instance_metrics(pred_mask, gt_mask, iou_thresholds=None):
    """计算实例级 AP/AR 指标（COCO 风格）。

    返回:
        ap_at_thresholds: 每个 IoU 阈值的 AP
        ar_at_thresholds: 每个 IoU 阈值的 AR
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 1.00, 0.05)

    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_pred = len(pred_contours)
    num_gt = len(gt_contours)

    if num_gt == 0:
        return np.zeros(len(iou_thresholds)), np.zeros(len(iou_thresholds))
    if num_pred == 0:
        return np.zeros(len(iou_thresholds)), np.zeros(len(iou_thresholds))

    # 计算 IoU 矩阵
    iou_matrix = np.zeros((num_pred, num_gt))
    for i, pc in enumerate(pred_contours):
        pm = np.zeros_like(pred_mask)
        cv2.drawContours(pm, [pc], -1, 1, thickness=cv2.FILLED)
        for j, gc in enumerate(gt_contours):
            gm = np.zeros_like(gt_mask)
            cv2.drawContours(gm, [gc], -1, 1, thickness=cv2.FILLED)
            inter = np.logical_and(pm.astype(bool), gm.astype(bool)).sum()
            union = np.logical_or(pm.astype(bool), gm.astype(bool)).sum()
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    aps, ars = [], []
    for thresh in iou_thresholds:
        gt_matched = np.zeros(num_gt, dtype=bool)
        tp = 0
        for i in range(num_pred):
            best_j, best_iou = -1, thresh
            for j in range(num_gt):
                if not gt_matched[j] and iou_matrix[i, j] > best_iou:
                    best_iou, best_j = iou_matrix[i, j], j
            if best_j != -1:
                tp += 1
                gt_matched[best_j] = True
        fp = num_pred - tp
        fn = num_gt - tp
        aps.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        ars.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    return np.array(aps), np.array(ars)


def evaluate_plaquesight(model, test_data_dir, device, output_dir, image_size=1024):
    """评估 PlaqueSight 模型并保存结果。

    返回:
        results: 包含 pixel_metrics / instance_metrics / efficiency 的字典
    """
    print(f"\n{'='*60}", flush=True)
    print(f"评估模型 on {test_data_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = TestDataset(data_dir=test_data_dir, image_size=image_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_iou, all_dice, all_precision, all_recall = [], [], [], []
    all_aps, all_ars = [], []
    inference_times = []

    with torch.no_grad():
        for images, gt_masks, stems, orig_images, gt_masks_orig in tqdm(test_loader, desc="评估中"):
            images = images.to(device)

            t0 = time.perf_counter()
            pred_masks, _ = model(images)
            elapsed = time.perf_counter() - t0
            inference_times.append(elapsed / images.size(0))

            pred_bin = (torch.sigmoid(pred_masks) > 0.5).float()

            for i in range(images.size(0)):
                pred_np = pred_bin[i].squeeze().cpu().numpy().astype(np.uint8)
                gt_np = gt_masks_orig[i].numpy() if torch.is_tensor(gt_masks_orig[i]) else gt_masks_orig[i]

                h, w = gt_np.shape
                pred_resized = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_NEAREST)

                iou, dice, precision, recall = calculate_pixel_metrics(pred_resized, gt_np)
                all_iou.append(iou); all_dice.append(dice)
                all_precision.append(precision); all_recall.append(recall)

                ap_arr, ar_arr = calculate_instance_metrics(pred_resized, gt_np)
                all_aps.append(ap_arr); all_ars.append(ar_arr)

    # 汇总
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    aps_stack = np.stack(all_aps)
    ars_stack = np.stack(all_ars)

    results = {
        'pixel_metrics': {
            'iou': float(np.mean(all_iou)),
            'dice': float(np.mean(all_dice)),
            'precision': float(np.mean(all_precision)),
            'recall': float(np.mean(all_recall)),
        },
        'instance_metrics': {
            'mAP@50': float(aps_stack[:, 0].mean()),
            'mAP@75': float(aps_stack[:, 5].mean()) if len(iou_thresholds) > 5 else 0,
            'mAP@50:95': float(aps_stack.mean()),
            'mAR@50:95': float(ars_stack.mean()),
        },
        'efficiency': {
            'avg_inference_time_ms': float(np.mean(inference_times)) * 1000,
            'fps': 1.0 / float(np.mean(inference_times)),
        }
    }

    # 打印
    print("\n" + "=" * 60)
    print("PlaqueSight Evaluation Results")
    print("=" * 60)
    print(f"  IoU:       {results['pixel_metrics']['iou']:.4f}")
    print(f"  Dice:      {results['pixel_metrics']['dice']:.4f}")
    print(f"  mAP@50:    {results['instance_metrics']['mAP@50']:.4f}")
    print(f"  mAP@50:95: {results['instance_metrics']['mAP@50:95']:.4f}")
    print(f"  FPS:       {results['efficiency']['fps']:.2f}")
    print("=" * 60)

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
