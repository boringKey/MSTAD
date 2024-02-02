import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssim_l import ssim


class ReverseMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity()
        self.weight = weight

    def Color_distance(self, x, y, ndim=-1):
        (_, channel, _, _) = x.size()
        _img1 = x.permute(0, 2, 3, 1).reshape(-1, channel)
        _img2 = y.permute(0, 2, 3, 1).reshape(-1, channel)
        return 1 - F.cosine_similarity(_img1, _img2, dim=ndim).mean()

    def forward(self,input, reverse_pic):
        loss_0 = self.criterion_mse(input['image'].cuda(), reverse_pic.cuda())
        loss_4 = self.criterion_mse(input["feature_rec"].cuda(), input["feature_align"].cuda())
        return loss_4 + loss_0

class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec1 = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec1, feature_align)


class SplitFeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        loss = 0
        feature_rec = input["features_rec"]
        feature_align = input["features"]
        for i in range(4):
            loss = loss + self.criterion_mse(feature_rec[i], feature_align[i])
        return loss

class ImageMSELoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["image"]
        image_rec = input["image_rec"]
        return self.criterion_mse(image, image_rec)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
