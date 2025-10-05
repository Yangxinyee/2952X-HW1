from typing import Tuple

import torch
from torch import nn
import timm


def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, C = x.shape
    num_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :num_keep]
    x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C))
    mask = torch.ones(B, N, device=x.device)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_keep, mask, ids_restore


class MAEModel(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, encoder_dim: int = 768, decoder_dim: int = 512, decoder_depth: int = 4, mask_ratio: float = 0.75) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        self.encoder = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.encoder.patch_embed.proj.bias = None if getattr(self.encoder.patch_embed.proj, 'bias', None) is None else self.encoder.patch_embed.proj.bias
        self.encoder.reset_classifier(0)

        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=8, dim_feedforward=decoder_dim * 4, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * 3)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p = self.patch_size
        h = H // p
        w = W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor, img_size: int = 224) -> torch.Tensor:
        B, N, D = x.shape
        p = self.patch_size
        h = w = int((N) ** 0.5)
        x = x.reshape(B, h, w, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, h * p, w * p)
        return x

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get patch embeddings from encoder's patch_embed
        x = self.encoder.patch_embed(images)  # (B, N, C)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = x + self.encoder.pos_embed[:, 1:, :]

        x_keep, mask, ids_restore = random_masking(x, self.mask_ratio)
        x_keep = torch.cat([cls_token, x_keep], dim=1)
        for blk in self.encoder.blocks:
            x_keep = blk(x_keep)
        x_keep = self.encoder.norm(x_keep)
        enc_tokens = x_keep[:, 1:, :]  # remove cls

        B, N, C = x.shape
        num_mask = N - enc_tokens.shape[1]
        mask_tokens = self.mask_token.repeat(B, num_mask, 1)
        dec_tokens = torch.cat([enc_tokens, mask_tokens], dim=1)
        dec_tokens = torch.gather(dec_tokens, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, dec_tokens.shape[-1]))
        dec_tokens = self.decoder_embed(dec_tokens) + self.decoder_pos_embed
        dec_tokens = self.decoder(dec_tokens)
        pred = self.decoder_pred(dec_tokens)
        recons = self.unpatchify(pred)
        return recons, mask, enc_tokens



