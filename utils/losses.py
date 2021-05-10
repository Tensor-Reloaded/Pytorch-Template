import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch import nn
try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


class TorchDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True,
                 per_image=False, logits=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.logits = logits

    def forward(self, outputs, targets):
        if self.logits:
            outputs = torch.sigmoid(outputs)
        
        batch_size = outputs.size()[0]
        eps = 1e-5
        if not self.per_image:
            batch_size = 1
        dice_target = targets.contiguous().view(batch_size, -1).float()
        dice_output = outputs.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
        loss = (1 - (2 * intersection + eps) / union).mean()
        return loss


class TorchFocalLoss(nn.Module):
    """Implementation of Focal Loss[1]_ modified from Catalyst [2]_ .
    Arguments
    ---------
    gamma : :class:`int` or :class:`float`
        Focusing parameter. See [1]_ .
    alpha : :class:`int` or :class:`float`
        Normalization factor. See [1]_ .
    References
    ----------
    .. [1] https://arxiv.org/pdf/1708.02002.pdf
    .. [2] https://catalyst-team.github.io/catalyst/
    """

    def __init__(self, gamma=2, reduce=True, logits=False):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce
        self.logits = logits

    def forward(self, outputs, targets):
        """Calculate the loss function between `outputs` and `targets`.
        Arguments
        ---------
        outputs : :class:`torch.Tensor`
            The output tensor from a model.
        targets : :class:`torch.Tensor`
            The training target.
        Returns
        -------
        loss : :class:`torch.Variable`
            The loss value.
        """

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets,
                                                          reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(outputs, targets,
                                              reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class TorchJaccardLoss(torch.nn.modules.Module):
    # modified from XD_XD's implementation
    def __init__(self):
        super(TorchJaccardLoss, self).__init__()

    def forward(self, outputs, targets):
        eps = 1e-15

        jaccard_target = (targets == 1).float()
        jaccard_output = torch.sigmoid(outputs)
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        jaccard_score = ((intersection + eps) / (union - intersection + eps))
        self._stash_jaccard = jaccard_score
        loss = 1. - jaccard_score

        return loss


class TorchStableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(TorchStableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()



class SoftTargetCrossEntropy(torch.nn.modules.Module):

    def __init__(self, class_weights = None, device='cuda'):
        super(SoftTargetCrossEntropy, self).__init__()
        self.device = device
        self.class_weights = None
        if class_weights != None:
            self.class_weights = torch.Tensor(class_weights).to(self.device)

    def forward(self, x, target):
        lsm = F.log_softmax(x, dim=-1)
        if self.class_weights != None:
            lsm = lsm * self.class_weights
        loss = torch.sum(-target * lsm, dim=-1)
        return loss.mean()




class CrossEntropyLoss(torch.nn.modules.Module):

    def __init__(self, class_weights = None, device='cuda', half=True):
        super(CrossEntropyLoss, self).__init__()
        self.device = device
        if class_weights != None:
            w = torch.Tensor(class_weights).to(self.device)
            if half:
                w = w.half()
            self.fnc = nn.CrossEntropyLoss(weight=w)
        else:
            self.fnc = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        return self.fnc(prediction,target.argmax(-1))



losses = {
    'l1loss': {
        'constructor': nn.L1Loss,
        'higher_is_better': False
    },
    'smoothl1loss': {
        'constructor': nn.SmoothL1Loss,
        'higher_is_better': False
    },
    'mseloss': {
        'constructor': nn.MSELoss,
        'higher_is_better': False
    },
    'crossentropyloss': {
        'constructor': CrossEntropyLoss,
        'higher_is_better': False
    },
    'nllloss': {
        'constructor': nn.NLLLoss,
        'higher_is_better': False
    },
    'poisson_negative_log_likelihood': {
        'constructor': nn.PoissonNLLLoss,
        'higher_is_better': False
    },
    'kullback_leibler_divergence': {
        'constructor': nn.KLDivLoss,
        'higher_is_better': False
    },
    'binary_crossentropy': {
        'constructor': nn.BCELoss,
        'higher_is_better': False
    },
    'bcewithlogits': {
        'constructor': nn.BCEWithLogitsLoss,
        'higher_is_better': False
    },
    'hinge': {
        'constructor': nn.HingeEmbeddingLoss,
        'higher_is_better': False
    },
    'multiclass_hinge': {
        'constructor': nn.MultiMarginLoss,
        'higher_is_better': False
    },
    'softmarginloss': {
        'constructor': nn.SoftMarginLoss,
        'higher_is_better': False
    },
    'multiclass_softmargin': {
        'constructor': nn.MultiLabelSoftMarginLoss,
        'higher_is_better': False
    },
    'cosineloss': {
        'constructor': nn.CosineEmbeddingLoss,
        'higher_is_better': False
    },
    'focal': {
        'constructor': TorchFocalLoss,
        'higher_is_better': False
    },
    'jaccard': {
        'constructor': TorchJaccardLoss,
        'higher_is_better': False
    },
    'dice': {
        'constructor': TorchDiceLoss,
        'higher_is_better': False
    },
    'SoftTargetCrossEntropy': {
        'constructor': SoftTargetCrossEntropy,
        'higher_is_better': False
    }
}