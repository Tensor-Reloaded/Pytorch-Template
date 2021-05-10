import random
import math 
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.functional import conv2d
from torchvision import transforms as transforms

from .randaugment import RandAugment


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class OneHot(object):
    def __init__(self, num_classes, on_value=1., off_value=0., device='cuda'):
        self.num_classes = num_classes
        self.on_value = on_value
        self.off_value = off_value
        self.device = 'cuda'

    def __call__(self, x):
        x = torch.LongTensor([x]).long().view(-1, 1).to(self.device)
        return torch.full((x.size(0), self.num_classes), self.off_value, device=self.device).scatter_(1, x,
                                                                                                      self.on_value).squeeze(
            0)


class LambdaTransform(object):
    def __init__(self, lambda_string, params):
        self.lambda_func = lambda X, params=params: eval(lambda_string)

    def __call__(self, data):
        return self.lambda_func(data)


class Resize(object):
    def __init__(self, size, interpolation):
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, data):
        return torch.nn.functional.interpolate(data.unsqueeze(0), size=self.size, scale_factor=None,
                                               mode=self.interpolation, align_corners=None,
                                               recompute_scale_factor=None).squeeze(0)


class ShearX(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.1):
        assert -0.3 <= v <= 0.3
        if random.random() > 0.5:
            v = -v
        img = img.unsqueeze(0)
        grid = F.affine_grid(torch.Tensor([1, v, 0, 0, 1, 0]).view(1, 2, 3), img.size())
        return F.grid_sample(img, grid).squeeze(0)


class ShearY(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.1):
        assert -0.3 <= v <= 0.3
        if random.random() > 0.5:
            v = -v
        img = img.unsqueeze(0)
        grid = F.affine_grid(torch.Tensor([1, 0, 0, v, 1, 0]).view(1, 2, 3), img.size())
        return F.grid_sample(img, grid).squeeze(0)


class TranslateX(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.1):
        assert -0.45 <= v <= 0.45
        if random.random() > 0.5:
            v = -v
        v = v * img.size(1)
        img = img.unsqueeze(0)
        grid = F.affine_grid(torch.Tensor([1, 0, v, 0, 1, 0]).view(1, 2, 3), img.size())
        return F.grid_sample(img, grid).squeeze(0)


class TranslateXabs(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.1):
        assert 0 <= v
        if random.random() > 0.5:
            v = -v
        img = img.unsqueeze(0)
        grid = F.affine_grid(torch.Tensor([1, 0, v, 0, 1, 0]).view(1, 2, 3), img.size())
        return F.grid_sample(img, grid).squeeze(0)


class TranslateY(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.1):
        assert -0.45 <= v <= 0.45
        if random.random() > 0.5:
            v = -v
        v = v * img.size(2)
        img = img.unsqueeze(0)
        grid = F.affine_grid(torch.Tensor([1, 0, 0, 0, 1, v]).view(1, 2, 3), img.size())
        return F.grid_sample(img, grid).squeeze(0)


class TranslateYabs(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.1):
        assert 0 <= v
        if random.random() > 0.5:
            v = -v
        img = img.unsqueeze(0)
        grid = F.affine_grid(torch.Tensor([1, 0, 0, 0, 1, v]).view(1, 2, 3), img.size())
        return F.grid_sample(img, grid).squeeze(0)


class Autocontrast(object):
    def __init__(self):
        pass

    def __call__(self, img):
        bound = 1.0 if img.is_floating_point() else 255.0
        dtype = img.dtype if torch.is_floating_point(img) else torch.float32

        minimum = img.amin(dim=(-2, -1), keepdim=True).to(dtype)
        maximum = img.amax(dim=(-2, -1), keepdim=True).to(dtype)
        eq_idxs = torch.where(minimum == maximum)[0]
        minimum[eq_idxs] = 0
        maximum[eq_idxs] = bound
        scale = bound / (maximum - minimum)

        return ((img - minimum) * scale).clamp(0, bound).to(img.dtype)


class Rotate(object):
    def __init__(self):
        pass

    def __call__(self, img, v=10):
        assert -30 <= v <= 30
        if random.random() > 0.5:
            v = -v
        return TF.rotate(img, v)


class Invert(object):
    def __init__(self):
        pass

    def __call__(self, img):
        bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
        return bound - img


class Solarize(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.1):
        assert 0 <= v <= 1
        bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
        inverted_img = bound - img
        return torch.where(img >= v, inverted_img, img)


class Equalize(object):
    def __init__(self):
        pass

    @staticmethod
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

    def __call__(self, img):
        return torch.stack([self._scale_channel(img[c]) / 255.0 for c in range(img.size(0))])


class Flip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img.flip(-1)


class SolarizeAdd(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.1, threshold=0.5):
        img = img + (v / 255.0)
        img = torch.clip(img, 0, 1)

        bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
        inverted_img = bound - img
        return torch.where(img >= threshold, inverted_img, img)


def _blend(img1, img2, ratio):
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


class Brightness(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.4):
        assert 0.1 <= v <= 1.9
        return _blend(img, torch.zeros_like(img), v)


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
    result_tmp = conv2d(result_tmp, kernel, groups=result_tmp.shape[-3])
    result_tmp = _cast_squeeze_out(result_tmp, need_cast, need_squeeze, out_dtype)

    result = img.clone()
    result[..., 1:-1, 1:-1] = result_tmp

    return result


class Sharpness(object):
    def __init__(self):
        pass

    def __call__(self, img, v=0.3):
        return _blend(img, _blurred_degenerate_image(img), v)


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img



def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break
    
    def _erase3D(self, img, chan, img_d, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_d * img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                d = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if d < img_d and w < img_w and h < img_h:
                    up = random.randint(0, img_d - d)
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, up:up+d, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, d, h, w),
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, tensor):
        if len(tensor.size()) == 3:
            self._erase(tensor, *tensor.size(), tensor.dtype)
        elif len(tensor.size()) == 4:
            batch_size, chan, img_h, img_w = tensor.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(tensor[i], chan, img_h, img_w, tensor.dtype)
        elif len(tensor.size()) == 5:
            batch_size, chan, img_d, img_h, img_w = tensor.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase3D(tensor[i], chan, img_d, img_h, img_w, tensor.dtype)
        return tensor

class ImageRandomResizedCrop(object):
    def __init__(self, size, scale):
        if isinstance(size, list):
            size = tuple(size)

        self.fn = transforms.RandomResizedCrop(size, tuple(scale))

    def __call__(self, tensor):
        return self.fn(tensor)


transformations = {
    'ImageNormalize': transforms.Normalize,
    'ImageRandomCrop': transforms.RandomCrop,
    'ImageRandomResizedCrop': ImageRandomResizedCrop,
    'ImageRandomHorizontalFlip': transforms.RandomHorizontalFlip,
    'ImageToTensor': transforms.ToTensor,
    'ImageRandomRotation': transforms.RandomRotation,
    'ImageColorJitter': transforms.ColorJitter,
    'ImageLighting': Lighting,
    'ImageResize': transforms.Resize,
    'ImageCenterCrop': transforms.CenterCrop,
    'Resize': Resize,
    'LambdaTransform': LambdaTransform,
    'OneHot': OneHot,
    'RandAugment': RandAugment,
    'RandomErasing': RandomErasing,
}
