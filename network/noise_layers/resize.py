import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Resize(nn.Module):

    def __init__(self, height_ratio, width_ratio):
        super(Resize, self).__init__()

        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, image_and_cover):

        image, cover_image = image_and_cover

        _, _, h, w = image.shape

        # Random scale within the specified ratio range
        scale_h = random.uniform(
            self.height_ratio[0],
            self.height_ratio[1]
        ) if isinstance(self.height_ratio, (tuple, list)) else self.height_ratio

        scale_w = random.uniform(
            self.width_ratio[0],
            self.width_ratio[1]
        ) if isinstance(self.width_ratio, (tuple, list)) else self.width_ratio

        target_h = max(8, int(round(h * scale_h)))
        target_w = max(8, int(round(w * scale_w)))

        # Random interpolation method
        mode = random.choice([
            "bilinear",
            "bicubic",
            "nearest"
        ])

        if mode in ("bilinear", "bicubic"):
            align_corners = False
        else:
            align_corners = None

        # Resize down/up to simulate distortion
        resized = F.interpolate(
            image,
            size=(target_h, target_w),
            mode=mode,
            align_corners=align_corners
        )

        resized = F.interpolate(
            resized,
            size=(h, w),
            mode=mode,
            align_corners=align_corners
        )

        return resized.clamp(-1.0, 1.0)