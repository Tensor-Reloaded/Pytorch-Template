import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import models


__all__ = ["ViT"]

# helpers

def pair(t):
    if isinstance(t, tuple):
        return t
    elif(isinstance(t, list)):
        return tuple(t)
    else:
        return (t, t)

# classes

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer_name, transformer_params, pool = 'cls', channels = 3, emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        try:
            transformer = getattr(models, transformer_name)
        except:
            print(f"This transformer is not implemented ({transformer_name}), go ahead and commit it")
            exit()

        transformer_params = dict((key,d[key]) for d in transformer_params for key in d)
        self.transformer = transformer(**transformer_params)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)