#!/usr/bin/env python3
"""YOLO Few-Shot 训练（Baseline）

用法:
    python train.py --shots 1,5,10,20,100,160 --train-dir ../../data/train --test-dir ../../data/test

前提: pip install ultralytics
"""
import sys, os, argparse, shutil, yaml, random
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def prepare_yolo_dataset(source_dir, output_dir, num_samples=None, seed=42):
    """准备 YOLO 格式数据集"""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    img_dir = output_dir / "images"; lab_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lab_dir.mkdir(parents=True, exist_ok=True)

    exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
    files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(exts)])
    valid = [f for f in files if (source_dir / (Path(f).stem + ".txt")).exists()]

    if num_samples and num_samples < len(valid):
        random.seed(seed)
        valid = random.sample(valid, num_samples)

    for f in valid:
        shutil.copy(str(source_dir / f), str(img_dir / f))
        src_txt = source_dir / (Path(f).stem + ".txt")
        if src_txt.exists():
            shutil.copy(str(src_txt), str(lab_dir / (Path(f).stem + ".txt")))

    print(f"[YOLO] {len(valid)} 样本已准备好: {output_dir}", flush=True)
    return output_dir


def train_yolo(train_dir, test_dir, num_samples, output_base="yolo_experiments",
               img_size=1024, num_epochs=100, seed=42):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_base) / f"yolo_{num_samples}shot_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 准备训练数据
    train_out = exp_dir / "train_dataset"
    prepare_yolo_dataset(train_dir, train_out, num_samples=num_samples, seed=seed)

    # 准备测试数据
    test_out = exp_dir / "test_dataset"
    prepare_yolo_dataset(test_dir, test_out, num_samples=None, seed=seed)

    # 创建 YOLO 配置文件
    data_yaml = exp_dir / "data.yaml"
    data_config = {
        "path": str(exp_dir),
        "train": "train_dataset/images",
        "val": "test_dataset/images",
        "nc": 1,
        "names": ["plaque"]
    }
    with open(data_yaml, "w") as f:
        yaml.dump(data_config, f)

    # 训练
    print(f"[YOLO {num_samples}shot] 开始训练 ...", flush=True)
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=str(data_yaml),
        epochs=num_epochs,
        imgsz=img_size,
        batch=4,
        workers=1,
        device=0 if sys.platform != "darwin" else "mps",
        project=str(exp_dir),
        name="train",
        exist_ok=True,
        verbose=False,
    )

    print(f"[YOLO {num_samples}shot] 完成!", flush=True)
    return str(exp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", default="1,5,10,20,100,160")
    parser.add_argument("--train-dir", default=str(ROOT / "data/train"))
    parser.add_argument("--test-dir", default=str(ROOT / "data/test"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/output/yolo"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=1024)
    args = parser.parse_args()

    shots = [int(s.strip()) for s in args.shots.split(",")]
    print(f"YOLO Few-Shot: {shots}")

    for n in shots:
        try:
            train_yolo(args.train_dir, args.test_dir, n, args.output_dir,
                       img_size=args.img_size, num_epochs=args.epochs)
        except Exception as e:
            print(f"  ✗ {n}-shot 失败: {e}", flush=True)


if __name__ == "__main__":
    main()
