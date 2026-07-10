"""Few-shot 菌斑分割数据集（YOLO 分割格式）"""
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


class FewShotPlaqueDataset(Dataset):
    """训练数据集，支持 few-shot 随机采样。

    数据格式: YOLO 分割格式，每张图片对应一个同名 .txt 标签文件。
    .txt 每行: class_id x1 y1 x2 y2 ... (归一化到 [0,1] 的多边形坐标)

    参数:
        data_dir: 包含图片和 .txt 标签的目录
        image_size: 模型输入尺寸（默认 1024）
        num_samples: 随机采样数量，None 表示使用全部
        random_seed: 随机种子（默认 42）
    """

    def __init__(self, data_dir, image_size=1024, num_samples=5, random_seed=42):
        self.data_dir = data_dir
        self.image_size = image_size

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        all_files = [f for f in os.listdir(data_dir) if f.lower().endswith(image_extensions)]

        # 只保留有对应标签文件的图片
        valid_files = []
        for f in all_files:
            base = os.path.splitext(f)[0]
            if os.path.exists(os.path.join(data_dir, base + ".txt")):
                valid_files.append(f)

        print(f"[Dataset] {data_dir}: {len(valid_files)} 有效图片-标签对", flush=True)

        if num_samples is not None and num_samples > 0:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            if num_samples > len(valid_files):
                print(f"[Dataset] 警告: 请求样本数 ({num_samples}) > 可用数 ({len(valid_files)})，使用全部", flush=True)
                self.image_files = valid_files
            else:
                self.image_files = random.sample(valid_files, num_samples)
                print(f"[Dataset] Few-shot 模式: {len(valid_files)} → {num_samples} 样本", flush=True)
        else:
            self.image_files = valid_files

        print(f"[Dataset] 最终数据集大小: {len(self.image_files)}", flush=True)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        original_w, original_h = image.size

        # 解析 YOLO 标签 → mask
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(self.data_dir, base_name + ".txt")
        mask = np.zeros((original_h, original_w), dtype=np.uint8)
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    coords = np.array([float(p) for p in parts[1:]])
                    coords[0::2] *= original_w
                    coords[1::2] *= original_h
                    polygon_pts = coords.reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [polygon_pts], color=255)

        mask = Image.fromarray(mask)
        image_tensor = self.transform(image)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask_tensor = F.to_tensor(mask)
        return image_tensor, mask_tensor


class TestDataset(Dataset):
    """测试数据集，保留原始图片和 GT mask 用于评估。

    参数:
        data_dir: 包含图片和 .txt 标签的目录
        image_size: 模型输入尺寸（默认 1024）
    """

    def __init__(self, data_dir, image_size=1024):
        self.data_dir = data_dir
        self.image_size = image_size

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        all_files = [f for f in os.listdir(data_dir) if f.lower().endswith(image_extensions)]

        self.valid_items = []
        for f in all_files:
            base = os.path.splitext(f)[0]
            txt_path = os.path.join(data_dir, base + ".txt")
            if os.path.exists(txt_path):
                self.valid_items.append({'image': f, 'label': txt_path, 'stem': base})

        print(f"[TestDataset] {len(self.valid_items)} 有效样本", flush=True)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.valid_items)

    def __getitem__(self, idx):
        item = self.valid_items[idx]
        img_path = os.path.join(self.data_dir, item['image'])
        image = Image.open(img_path).convert("RGB")
        original_w, original_h = image.size

        # GT mask
        gt_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        with open(item['label'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                coords = np.array([float(p) for p in parts[1:]])
                coords[0::2] *= original_w
                coords[1::2] *= original_h
                polygon_pts = coords.reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt_mask, [polygon_pts], color=1)

        image_tensor = self.transform(image)
        gt_resized = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        gt_tensor = torch.from_numpy(gt_resized).float().unsqueeze(0)

        return image_tensor, gt_tensor, item['stem'], np.array(image), gt_mask
