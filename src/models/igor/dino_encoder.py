import kornia
import torch
import torch.nn as nn

from .backbones.dinov2.hub.backbones import *

MODEL_SIZES = {
    "small": dinov2_vits14_reg,
    "base": dinov2_vitb14_reg,
    "large": dinov2_vitl14_reg,
    "giant": dinov2_vitg14_reg,
}

class FrozenDinov2ImageEmbedder(nn.Module):
    """
    Uses the Dinov2 Vit encoder for images
    """

    def __init__(
        self,     
        pretrained=True, 
        freeze=True, 
        antialias=True,
        size="small",
    ):
        super().__init__()
        dino_model = MODEL_SIZES[size]
        self.model = dino_model(pretrained=pretrained)
        if freeze:
            self.freeze()

        self.antialias = antialias

        # dinov2/train/train.py (make_dataset)
        self.register_buffer(
            "mean", torch.Tensor([0.485, 0.456, 0.406]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.229, 0.224, 0.225]), persistent=False
        )

    def preprocess(self, x):
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        z, tokens = self.encode_with_vision_transformer(image)  # NOTE: torch's implementation of upsample_bicubic2d doesn't support bf16
        z = z.to(image.dtype)
        tokens = tokens.to(image.dtype)
        return tokens, z

    def encode_with_vision_transformer(self, img):
        ret = self.model(img)
        return ret['x_norm_clstoken'], ret['x_norm_patchtokens'] 