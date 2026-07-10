"""PlaqueSight 模型：DINOv3 + Adapter + SAM 融合架构"""
import torch
import torch.nn as nn


class ConvPromptAdapter(nn.Module):
    """轻量级适配器，将 DINOv3 特征投影到 SAM prompt embeddings。

    参数:
        dino_embed_dim: DINOv3 输出通道数（vit_h=1280, vit_l=1024）
        sam_prompt_dim: SAM prompt embedding 维度（默认 256）
    """
    def __init__(self, dino_embed_dim, sam_prompt_dim=256):
        super().__init__()
        self.adapter_layers = nn.Sequential(
            nn.Conv2d(dino_embed_dim, sam_prompt_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(sam_prompt_dim, sam_prompt_dim, kernel_size=3, padding=1),
        )

    def forward(self, dino_features):
        return self.adapter_layers(dino_features)


class PlaqueSightModel(nn.Module):
    """PlaqueSight 模型：使用 DINOv3 特征作为 SAM 的 dense prompt。

    架构:
        Input → DINOv3 (frozen) → adapter (trainable) → SAM prompt encoder
              → SAM image encoder (frozen) → SAM mask decoder → mask

    只训练 adapter 参数，SAM 和 DINOv3 的权重均被冻结。
    """

    def __init__(self, dino, sam_model, adapter):
        super().__init__()
        self.dino = dino
        self.sam_enc = sam_model.image_encoder
        self.sam_prompt = sam_model.prompt_encoder
        self.sam_dec = sam_model.mask_decoder
        self.adapter = adapter

    def forward(self, x):
        with torch.no_grad():
            df = self.dino.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
            se = self.sam_enc(x)
        dp = self.adapter(df)
        sp, _ = self.sam_prompt(points=None, boxes=None, masks=None)
        masks, iou_pred = self.sam_dec(
            image_embeddings=se,
            image_pe=self.sam_prompt.get_dense_pe(),
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dp,
            multimask_output=False,
        )
        masks = nn.functional.interpolate(
            masks, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False
        )
        return masks, iou_pred
