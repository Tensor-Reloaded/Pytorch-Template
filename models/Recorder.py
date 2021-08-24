from functools import wraps
import torch
from torch import nn
import models

from functools import partial
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


from models.PerceiverIO import Attention as Attention # Make this switch dynamically based on the model used



__all__ = ['Recorder']


def find_modules(nn_module, attention_class):
    return [module for module in nn_module.modules() if isinstance(module, attention_class)]

class Recorder(nn.Module):
    def __init__(self, transformer_name, transformer_params, discard_ratio=0.9, head_fusion='max', device = None):
        super().__init__()

        try:
            transformer = getattr(models, transformer_name)
        except:
            print(f"This transformer is not implemented ({transformer_name}), go ahead and commit it")
            exit()

        transformer_params = dict((key,d[key]) for d in transformer_params for key in d)
        self.vit = transformer(**transformer_params)

        self.attention_class = Attention # TODO find another way to dinamically get the Attention class used in the model

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output)

    def _register_hook(self):
        modules = find_modules(self.vit, self.attention_class)
        for module in modules:
            handle = module.record.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.recordings.clear()

    def rollout_attention(self, attentions):
        result = torch.eye(attentions[0].size(-1), device=self.device)

        for attention in attentions:
            if self.head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif self.head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif self.head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*self.discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1), device=self.device)
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=0)

            result = torch.matmul(a, result)
        
        # Look at the total attention between the class token,
        # and the image patches
        # mask = result#[0, 0 , 1 :]
        # In case of 224x224 image, this brings us from 196 to 14
        
        mask = (result - result.min(dim=0)[0]) / (result.max(dim=0)[0] - result.min(dim=0)[0]+1e-5)
        return mask    


    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))
        # What we have: 32, 128, 8, 8
        # We want: 32, 6, 288, 288

        recording = torch.stack(recordings, 0)
        mask = self.rollout_attention(recording)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=img.size()[-2:], mode='bilinear', align_corners=False).squeeze(1)

        return pred, mask