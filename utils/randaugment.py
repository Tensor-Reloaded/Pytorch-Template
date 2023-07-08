# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# TODO: Use library
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

__all__ = ["RandAugment"]


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    img = img.unsqueeze(0)
    grid = F.affine_grid(torch.tensor([1, v, 0, 0, 1, 0], device=img.device, dtype=img.dtype).view(1, 2, 3), img.size(),
                         align_corners=False)
    return F.grid_sample(img, grid, align_corners=False).squeeze(0)


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    img = img.unsqueeze(0)
    grid = F.affine_grid(torch.tensor([1, 0, 0, v, 1, 0], device=img.device, dtype=img.dtype).view(1, 2, 3), img.size(),
                         align_corners=False)
    return F.grid_sample(img, grid, align_corners=False).squeeze(0)


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size(1)
    img = img.unsqueeze(0)
    grid = F.affine_grid(torch.tensor([1, 0, v, 0, 1, 0], device=img.device, dtype=img.dtype).view(1, 2, 3), img.size(),
                         align_corners=False)
    return F.grid_sample(img, grid, align_corners=False).squeeze(0)


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 150
    if random.random() > 0.5:
        v = -v
    img = img.unsqueeze(0)
    grid = F.affine_grid(torch.tensor([1, 0, v, 0, 1, 0], device=img.device, dtype=img.dtype).view(1, 2, 3), img.size(),
                         align_corners=False)
    return F.grid_sample(img, grid, align_corners=False).squeeze(0)


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size(2)
    img = img.unsqueeze(0)
    grid = F.affine_grid(torch.tensor([1, 0, 0, 0, 1, v], device=img.device, dtype=img.dtype).view(1, 2, 3), img.size(),
                         align_corners=False)
    return F.grid_sample(img, grid, align_corners=False).squeeze(0)


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 150
    if random.random() > 0.5:
        v = -v
    img = img.unsqueeze(0)
    grid = F.affine_grid(torch.tensor([1, 0, 0, 0, 1, v], device=img.device, dtype=img.dtype).view(1, 2, 3), img.size(),
                         align_corners=False)
    return F.grid_sample(img, grid, align_corners=False).squeeze(0)


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return TF.rotate(img, v)


def AutoContrast(img, _):
    bound = 1.0 if img.is_floating_point() else 255.0
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32

    minimum = img.amin(dim=(-2, -1), keepdim=True).to(dtype)
    maximum = img.amax(dim=(-2, -1), keepdim=True).to(dtype)
    eq_idxs = torch.where(minimum == maximum)[0]
    minimum[eq_idxs] = 0
    maximum[eq_idxs] = bound
    scale = bound / (maximum - minimum)

    return ((img - minimum) * scale).clamp(0, bound).to(img.dtype)


def Invert(img, _):
    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    return bound - img


def Equalize(img, _):
    def _scale_channel(img_chan):
        img_chan = img_chan * 255.0

        hist = torch.histc(img_chan.to(torch.float32), bins=256, min=0, max=255)

        nonzero_hist = hist[hist != 0]
        step = torch.div(nonzero_hist[:-1].sum(), 255, rounding_mode='floor')
        if step == 0:
            return img_chan

        lut = torch.div(
            torch.cumsum(hist, 0) + torch.div(step, 2, rounding_mode='floor'),
            step, rounding_mode='floor')
        lut = torch.nn.functional.pad(lut, [1, 0])[:-1].clamp(0, 255)

        return lut[img_chan.to(torch.int64)]

    return torch.stack([_scale_channel(img[c]) / 255.0 for c in range(img.size(0))])


def Flip(img, _):  # not from the paper
    return img.flip(-1)


def Solarize(img, v):  # [0, 256]
    v = v / 255.0
    assert 0 <= v <= 1
    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    inverted_img = bound - img
    return torch.where(img >= v, inverted_img, img)


def SolarizeAdd(img, addition=0, threshold=128):
    img = img + (addition / 255.0)
    img = torch.clip(img, 0, 1)

    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    inverted_img = bound - img
    return torch.where(img >= threshold, inverted_img, img)


def Brightness(img, v):
    def _blend(img1, img2, ratio):
        ratio = float(ratio)
        bound = 1.0 if img1.is_floating_point() else 255.0
        return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)

    assert 0.1 <= v <= 1.9
    return _blend(img, torch.zeros_like(img), v)


def Sharpness(img, v):
    def _cast_squeeze_in(img, req_dtypes):
        need_squeeze = False
        # make image NCHW
        if img.ndim < 4:
            img = img.unsqueeze(dim=0)
            need_squeeze = True

        out_dtype = img.dtype
        need_cast = False
        if out_dtype not in req_dtypes:
            need_cast = True
            req_dtype = req_dtypes[0]
            img = img.to(req_dtype)
        return img, need_cast, need_squeeze, out_dtype

    def _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype):
        if need_squeeze:
            img = img.squeeze(dim=0)

        if need_cast:
            if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                # it is better to round before cast
                img = torch.round(img)
            img = img.to(out_dtype)

        return img

    def _blurred_degenerate_image(img):
        dtype = img.dtype if torch.is_floating_point(img) else torch.float32

        kernel = torch.ones((3, 3), dtype=dtype, device=img.device)
        kernel[1, 1] = 5.0
        kernel /= kernel.sum()
        kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

        result_tmp, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype, ])
        result_tmp = F.conv2d(result_tmp, kernel, groups=result_tmp.shape[-3])
        result_tmp = _cast_squeeze_out(result_tmp, need_cast, need_squeeze, out_dtype)

        result = img.clone()
        result[..., 1:-1, 1:-1] = result_tmp

        return result

    def _blend(img1, img2, ratio):
        ratio = float(ratio)
        bound = 1.0 if img1.is_floating_point() else 255.0
        return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)

    return _blend(img, _blurred_degenerate_image(img), v)


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    augment = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (TranslateX, 0., 0.33),  # 2
        (TranslateY, 0., 0.33),  # 3
        (TranslateXabs, 0., 150.0),
        (TranslateYabs, 0., 150.0),
    ]

    return augment


class RandAugment:
    def __init__(self, N, M, std=0.0):
        self.n = N
        self.m = M  # [0, 30]
        self.std = std
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            m = self.m
            if self.std > 0.0:
                m = random.gauss(self.m, self.std)
            val = (float(m) / 30.0) * float(maxval - minval) + minval
            val = np.clip(val, minval, maxval)
            if op in [ShearX, ShearY, TranslateX, TranslateY, TranslateXabs, TranslateYabs]:
                if len(img.shape) == 4:
                    for idx in range(img.size(1)):
                        img[:, idx, :, :] = op(img[:, idx, :, :], val)
                else:
                    img = op(img, val)
            else:
                img = op(img, val)

        return img
