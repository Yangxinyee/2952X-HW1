from typing import Literal, Tuple

import torch
from torch import nn
import timm


class VitFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = False) -> None:
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(0)
        self.feature_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use timm's forward_features to get representation
        feats = self.model.forward_features(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        # Many ViTs return (B, C) pooled features
        return feats


def get_vit_encoder(kind: Literal["random", "imagenet"]) -> Tuple[nn.Module, int]:
    pretrained = kind == "imagenet"
    enc = VitFeatureExtractor(pretrained=pretrained)
    return enc, enc.feature_dim


