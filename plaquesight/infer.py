"""PlaqueSight 推理函数"""
import cv2
import numpy as np
import torch


def preprocess_image(img_bgr, image_size=1024, device="cuda"):
    """预处理图片: 解码 → 缩放 → 归一化 → tensor"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (image_size, image_size))
    normed = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normed).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, h, w


def predict_mask(model, img_bgr, threshold=0.5, image_size=1024, device="cuda"):
    """推理：输入 BGR 图片，返回二值 mask（0/1）"""
    model.eval()
    tensor, h, w = preprocess_image(img_bgr, image_size, device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output[0] if isinstance(output, tuple) else output)
        prob = prob.squeeze().cpu().numpy()
    prob_orig = cv2.resize(prob, (w, h))
    return (prob_orig > threshold).astype(np.uint8)


def analyze_plaques(mask_bin, min_area=50):
    """分析二值 mask，返回每个菌斑的统计信息"""
    contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plaques = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, closed=True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        x, y, bw, bh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0.0
        cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0.0

        major, minor = 0.0, 0.0
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            major, minor = max(ellipse[1]), min(ellipse[1])

        plaques.append({
            "area_pixels": area,
            "perimeter_pixels": perimeter,
            "circularity": circularity,
            "bbox": {"x": x, "y": y, "w": bw, "h": bh},
            "centroid": {"x": cx, "y": cy},
            "major_axis": major,
            "minor_axis": minor,
        })

    plaques.sort(key=lambda p: p["area_pixels"], reverse=True)
    for i, p in enumerate(plaques):
        p["id"] = i + 1
    return plaques


def draw_overlay(img_bgr, mask_bin, plaques=None):
    """绘制覆盖了 mask 的可视化图"""
    vis = img_bgr.copy()
    mc = np.zeros_like(vis)
    mc[mask_bin > 0] = [0, 0, 255]
    vis = cv2.addWeighted(vis, 0.7, mc, 0.3, 0)

    if plaques:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                  (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for p in plaques:
            c = colors[(p["id"] - 1) % len(colors)]
            b = p["bbox"]
            cv2.rectangle(vis, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), c, 2)
            cv2.putText(vis, f"#{p['id']} {p['area_pixels']:.0f}px",
                        (b["x"], max(b["y"] - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return vis
