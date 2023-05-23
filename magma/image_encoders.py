import torch
import torch.nn as nn
from typing import Callable, Union
from torchtyping import patch_typeguard
from einops import rearrange
import timm
import open_clip
from functools import partial

# ----------------------------- Utils --------------------------------------

# clip.model.LayerNorm = (
#     nn.LayerNorm
# )  # we need to patch this for clip to work with deepspeed
# patch_typeguard()  # needed for torchtyping typechecks to work


class Lambda(torch.nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        assert hasattr(fn, "__call__")
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# ------------------------- Image encoders ----------------------------------


def nfresnet50(
    device: Union[torch.device, str] = None, pretrained: bool = True
) -> nn.Module:
    """
    Loads nfresnet50 model, removing the pooling layer and replacing it with
    an adaptive pooling layer.
    """
    encoder = torch.nn.Sequential(
        *list(timm.create_model("nf_resnet50", pretrained=pretrained).children())[:-1]
    )
    pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
    encoder = torch.nn.Sequential(encoder, pooling)
    if device is not None:
        encoder = encoder.to(device)
    return encoder


def clip_encoder(
    device: Union[torch.device, str] = None, name: str = "clip",
) -> nn.Module:
    """
    Loads clip's image encoder module, discarding the lm component.

    If the variant is a resnet model, we also remove the attention pooling.
    """
    if name in ["clip", "ViT-B/32"]:
        name, pretrained = "ViT-B-32", "openai"
    elif name in ["clip_resnet", "RN50x4"]:
        name, pretrained = "RN50x4", "openai"
    elif name in ["clip_resnet_large", "RN50x16"]:
        name, pretrained = "RN50x16", "openai"
    elif "openclip" in name:
        if "H" in name:
            name, pretrained = "ViT-H-14", "laion2b_s32b_b79k"
        elif "B" in name and "32" in name:
            name, pretrained = "ViT-B-32", "laion2b_s34b_b79k"
        else:
            raise NotImplementedError(f"Encoder {name} not recognized")
    else:
        raise NotImplementedError(f"Encoder {name} not recognized")

    # TODO better internet connection
    encoder = open_clip.create_model(name, device=device, precision="fp16" if "cuda" in str(device) else "fp32", pretrained=pretrained).visual

    if "RN" in name:
        # remove attention pooling
        encoder.attnpool = Lambda(
            partial(rearrange, pattern="b d h w -> b (h w) d")
        )  # remove attn pooling, just use reshaped features

    if False and hasattr(encoder, "transformer"):  # TODO when do we want to disable pooling?
        def forward(self, x: torch.Tensor):
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)

            ## a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            # x = self.patch_dropout(x)
            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = self.ln_post(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            return x
        encoder.forward = partial(forward, encoder)


    if device is not None:
        encoder = encoder.to(device)

    return encoder


def get_image_encoder(
    name: str, device: Union[torch.device, str] = None, pretrained: bool = False
) -> torch.nn.Module:
    """
    Loads image encoder module
    """
    if name == "nfresnet50":
        encoder = nfresnet50(device=device, pretrained=pretrained)
    elif "clip" in name:
        encoder = clip_encoder(device=device, name=name)
    else:
        raise ValueError(f"image encoder {name} not recognized")
    return encoder
