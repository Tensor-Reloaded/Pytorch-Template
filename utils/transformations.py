import random
import math 
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import albumentations.augmentations.transforms as ATF

from torch.nn.functional import conv2d
from torchvision import transforms as transforms

from scipy.fftpack import dct, idct

import numpy as np
import skimage.morphology
from skimage.morphology import disk
import cv2 as cv

from .randaugment import RandAugment

class TransformWrapper(object):
    def __init__(self, transform, apply_to=None):
        self.transform = transform
        self.apply_to = apply_to

    def __call__(self, data):
        if self.apply_to is None:
            return self.transform(data)
        else:
            if self.apply_to == 'input':
                return self.transform(data[0]), data[1]
            elif self.apply_to == 'target':
                return data[0], self.transform(data[1])
            else:
                raise ValueError("apply_to must be 'input' or 'target'")



class LightingNoise(object):
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
        x = torch.LongTensor([x]).long().view(-1, 1)
        return torch.full((x.size(0), self.num_classes), self.off_value).scatter_(1, x, self.on_value).squeeze(0)


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
        if isinstance(data, np.ndarray):
            return cv.resize(data, dsize=self.size)

        return torch.nn.functional.interpolate(data.unsqueeze(0), size=self.size, scale_factor=None,
                                               mode=self.interpolation, align_corners=None,
                                               recompute_scale_factor=None).squeeze(0)


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
            probability=0.5, min_area=0.02, max_area=0.2, min_aspect=0.3, max_aspect=None,
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


class Unsqueeze(object):
    def __init__(self, dimension):
        self.dimension = dimension

    def __call__(self, tensor):
        return tensor.unsqueeze(self.dimension)

        
class Squeeze(object):
    def __init__(self, dimension):
        self.dimension = dimension

    def __call__(self, tensor):
        return tensor.squeeze(self.dimension)
        
class Half(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return tensor.half()

class TensorType(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.type(self.dtype)

class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return torch.tensor(tensor)

class Normalize(object):
    def __init__(self, maximum=None, minimum=None):
        self.maximum = maximum
        self.minimum = minimum

    def __call__(self, tensor):
        if self.maximum is None:
            self.maximum = tensor.max()
        if self.minimum is None:
            self.minimum = tensor.min()
        return (tensor - self.minimum) / (self.maximum - self.minimum)


transformations = {
    'ImageNormalize': {
        'constructor': transforms.Normalize,
        'cacheable': True,
    },
    'ImageRandomCrop': {
        'constructor': transforms.RandomCrop,
        'cacheable': False,
    },
    'ImageRandomResizedCrop': {
        'constructor': ImageRandomResizedCrop,
        'cacheable': False,
    },
    'ImageRandomHorizontalFlip': {
        'constructor': transforms.RandomHorizontalFlip,
        'cacheable': False,
    },
    'ImageToTensor': {
        'constructor': transforms.ToTensor,
        'cacheable': True,
    },
    'ToTensor': {
        'constructor': ToTensor,
        'cacheable': True,
    },
    'ImageRandomRotation': {
        'constructor': transforms.RandomRotation,
        'cacheable': False,
    },
    'ImageColorJitter': {
        'constructor': transforms.ColorJitter,
        'cacheable': False,
    },
    'ImageLightingNoise': {
        'constructor': LightingNoise,
        'cacheable': False,
    },
    'ImageResize': {
        'constructor': transforms.Resize,
        'cacheable': True,
    },
    'ImageCenterCrop': {
        'constructor': transforms.CenterCrop,
        'cacheable': True,
    },
    'Resize': {
        'constructor': Resize,
        'cacheable': True,
    },
    'LambdaTransform': {
        'constructor': LambdaTransform,
        'cacheable': True,
    },
    'OneHot': {
        'constructor': OneHot,
        'cacheable': True,
    },
    'RandAugment': {
        'constructor': RandAugment,
        'cacheable': False,
    },
    'RandomErasing': {
        'constructor': RandomErasing,
        'cacheable': False,
    },
    'Unsqueeze': {
        'constructor': Unsqueeze,
        'cacheable': True,
    },
    'Squeeze': {
        'constructor': Squeeze,
        'cacheable': True,
    },
    'Half': {
        'constructor': Half,
        'cacheable': True,
    },
    'TensorType': {
        'constructor': TensorType,
        'cacheable': True,
    },
    'Normalize': {
        'constructor': Normalize,
        'cacheable': True,
    },
    'TensorType': {
        'constructor': TensorType,
        'cacheable': True,
    },
}


if __name__ == '__main__':
    pass