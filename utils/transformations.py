from torchvision import transforms as transforms


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



transformations = {
    'Normalize': transforms.Normalize,
    'RandomCrop': transforms.RandomCrop,
    'RandomResizedCrop': transforms.RandomResizedCrop,
    'RandomHorizontalFlip': transforms.RandomHorizontalFlip,
    'ToTensor': transforms.ToTensor,
    'ColorJitter': transforms.ColorJitter,
    'Lighting': Lighting,
    'Resize': transforms.Resize,
    'CenterCrop': transforms.CenterCrop,
}

