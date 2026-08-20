import io
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF


class JpegTest(nn.Module):
    """
    Real PIL-based JPEG compression attack layer.
    Fully resolution-agnostic and runs in-memory via BytesIO (no disk I/O).
    """
    def __init__(self, Q=50, subsample=2):
        super(JpegTest, self).__init__()
        self.Q = int(Q)
        self.subsample = subsample

    def forward(self, image_and_cover):
        if isinstance(image_and_cover, (tuple, list)):
            image, cover_image = image_and_cover
        else:
            image, cover_image = image_and_cover, None

        device = image.device
        dtype = image.dtype
        B, C, H, W = image.shape

        noised_image = torch.empty_like(image)

        # Process each image in batch independently
        for i in range(B):
            # 1. Denormalize from [-1, 1] to [0, 255] uint8 on CPU
            img_np = (
                (image[i].detach().clamp(-1.0, 1.0).permute(1, 2, 0) + 1.0) * 127.5
            ).to(torch.uint8).cpu().numpy()

            im = Image.fromarray(img_np)

            # 2. In-memory JPEG encoding & decoding
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=self.Q, subsampling=self.subsample)
            buffer.seek(0)
            jpeg_im = Image.open(buffer)

            # 3. Convert back to Tensor [0.0, 1.0] -> Normalize back to [-1.0, 1.0]
            # TF.to_tensor gives (C, H, W) in range [0, 1]
            tensor_img = TF.to_tensor(jpeg_im).to(device=device, dtype=dtype)
            tensor_img = (tensor_img - 0.5) / 0.5

            noised_image[i] = tensor_img
            buffer.close()

        if cover_image is not None:
            return noised_image, cover_image
        return noised_image



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class JpegBasic(nn.Module):
    """
    Resolution-independent, fully differentiable/approximated JPEG simulation base module.
    Supports non-square images (H != W) and dynamic batch sizes on GPU/CPU.
    """
    def __init__(self):
        super(JpegBasic, self).__init__()

        # Precompute 8x8 DCT matrix and register as buffer
        coff = torch.zeros((8, 8), dtype=torch.float32)
        coff[0, :] = 1.0 * np.sqrt(1.0 / 8.0)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / 16.0) * np.sqrt(2.0 / 8.0)
        self.register_buffer('coff', coff)

        # Standard JPEG Luminance & Chrominance base tables (8x8)
        lum_tbl = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float32)

        chrom_tbl = torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float32)

        self.register_buffer('lum_tbl', lum_tbl)
        self.register_buffer('chrom_tbl', chrom_tbl)

    def _get_quant_tables(self, scale_factor, H, W, device, dtype):
        """Generates resolution-matched (1, C, H, W) quantization tables."""
        h_blocks = H // 8
        w_blocks = W // 8

        q_lum = (self.lum_tbl.to(device=device, dtype=dtype) * scale_factor).round().clamp(min=1.0)
        q_chrom = (self.chrom_tbl.to(device=device, dtype=dtype) * scale_factor).round().clamp(min=1.0)

        # Tile to match full image dimensions (1, 1, H, W)
        q_lum = q_lum.repeat(h_blocks, w_blocks).unsqueeze(0).unsqueeze(0)
        q_chrom = q_chrom.repeat(h_blocks, w_blocks).unsqueeze(0).unsqueeze(0)

        # Concat along channel dimension: Y -> lum, U/V -> chrom (1, 3, H, W)
        return torch.cat([q_lum, q_chrom, q_chrom], dim=1)

    def std_quantization(self, image_yuv_dct, scale_factor, round_func=torch.round):
        B, C, H, W = image_yuv_dct.shape
        q_tbl = self._get_quant_tables(scale_factor, H, W, image_yuv_dct.device, image_yuv_dct.dtype)
        q_image = image_yuv_dct / q_tbl
        return round_func(q_image)

    def std_reverse_quantization(self, q_image_yuv_dct, scale_factor):
        B, C, H, W = q_image_yuv_dct.shape
        q_tbl = self._get_quant_tables(scale_factor, H, W, q_image_yuv_dct.device, q_image_yuv_dct.dtype)
        return q_image_yuv_dct * q_tbl

    def dct(self, image):
        """Resolution-agnostic 8x8 block 2D DCT."""
        B, C, H, W = image.shape
        h_blocks, w_blocks = H // 8, W // 8

        # Reshape to (B * C * h_blocks * w_blocks, 8, 8)
        x = image.view(B * C, h_blocks, 8, w_blocks, 8).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(-1, 8, 8)

        # 2D DCT: coff @ block @ coff.T
        coff = self.coff.to(device=image.device, dtype=image.dtype)
        x_dct = torch.matmul(coff, torch.matmul(x, coff.t()))

        # Reconstruct back to (B, C, H, W)
        x_dct = x_dct.view(B, C, h_blocks, w_blocks, 8, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        return x_dct.view(B, C, H, W)

    def idct(self, image_dct):
        """Resolution-agnostic 8x8 block 2D Inverse DCT."""
        B, C, H, W = image_dct.shape
        h_blocks, w_blocks = H // 8, W // 8

        x = image_dct.view(B * C, h_blocks, 8, w_blocks, 8).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(-1, 8, 8)

        # 2D IDCT: coff.T @ block @ coff
        coff = self.coff.to(device=image_dct.device, dtype=image_dct.dtype)
        x_idct = torch.matmul(coff.t(), torch.matmul(x, coff))

        x_idct = x_idct.view(B, C, h_blocks, w_blocks, 8, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        return x_idct.view(B, C, H, W)

    def rgb2yuv(self, image_rgb):
        image_yuv = torch.empty_like(image_rgb)
        image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
        return image_yuv

    def yuv2rgb(self, image_yuv):
        image_rgb = torch.empty_like(image_yuv)
        image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] - 0.714103821 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
        return image_rgb

    def subsampling(self, image_yuv, subsample):
        """Applies chroma 4:2:0 subsampling simulation across arbitrary aspect ratios."""
        if subsample == 2:
            y = image_yuv[:, 0:1, :, :]
            uv = image_yuv[:, 1:3, :, :]
            # Subsample 2x2 blocks by nearest/average downsampling then nearest upsampling
            uv_sub = F.interpolate(uv, scale_factor=0.5, mode='nearest')
            uv_up = F.interpolate(uv_sub, size=(image_yuv.shape[2], image_yuv.shape[3]), mode='nearest')
            return torch.cat([y, uv_up], dim=1)
        return image_yuv

    def yuv_dct(self, image, subsample):
        image = (image.clamp(-1.0, 1.0) + 1.0) * 127.5

        # Pad height and width to multiple of 8 independently
        pad_height = (8 - image.shape[2] % 8) % 8
        pad_width = (8 - image.shape[3] % 8) % 8
        if pad_width > 0 or pad_height > 0:
            image = F.pad(image, (0, pad_width, 0, pad_height), mode='reflect')

        image_yuv = self.rgb2yuv(image)
        image_sub = self.subsampling(image_yuv, subsample)
        image_dct = self.dct(image_sub)

        return image_dct, pad_width, pad_height

    def idct_rgb(self, image_quantization, pad_width, pad_height):
        image_idct = self.idct(image_quantization)
        image_ret = self.yuv2rgb(image_idct)

        # Un-pad cleanly
        H, W = image_ret.shape[2], image_ret.shape[3]
        image_rgb = image_ret[:, :, :H - pad_height if pad_height > 0 else H,
                                  :W - pad_width if pad_width > 0 else W]

        return (image_rgb / 127.5) - 1.0


import random
import torch
import torch.nn as nn


class Jpeg(JpegBasic):
    """
    Simulated JPEG compression noise layer.
    Inherits dynamic-resolution DCT/quantization from JpegBasic.
    Supports fixed quality factors (int/float) or random ranges (tuple/list).
    """
    def __init__(self, Q=50, subsample=0):
        super(Jpeg, self).__init__()
        self.Q = Q
        self.subsample = subsample

    def _get_scale_factor(self):
        """Computes JPEG standard scale factor from quality Q."""
        if isinstance(self.Q, (tuple, list)):
            q = random.uniform(self.Q[0], self.Q[1])
        else:
            q = float(self.Q)
        
        q = max(1.0, min(100.0, q))
        return 2.0 - q * 0.02 if q >= 50.0 else 50.0 / q

    def forward(self, image_and_cover):
        # Handle tuple or single tensor input
        if isinstance(image_and_cover, (tuple, list)):
            image, cover_image = image_and_cover
        else:
            image, cover_image = image_and_cover, None

        scale_factor = self._get_scale_factor()

        # [-1, 1] to [0, 255], RGB -> YUV, dynamic padding, 8x8 DCT
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        # Quantization & de-quantization
        image_quantization = self.std_quantization(image_dct, scale_factor)
        image_dequantization = self.std_reverse_quantization(image_quantization, scale_factor)

        # 8x8 IDCT, YUV -> RGB, un-pad, [0, 255] to [-1, 1]
        noised_image = self.idct_rgb(image_dequantization, pad_width, pad_height)
        noised_image = noised_image.clamp(-1.0, 1.0)

        # Preserve pipeline return signature
        if cover_image is not None:
            return noised_image, cover_image
        return noised_image


    import random
import torch
import torch.nn as nn


class JpegSS(JpegBasic):
    """
    Differentiable JPEG simulation using Smooth Step (SS) rounding approximation.
    Resolution-independent across arbitrary H and W dimensions.
    """
    def __init__(self, Q=50, subsample=0):
        super(JpegSS, self).__init__()
        self.Q = Q
        self.subsample = subsample

    def _get_scale_factor(self):
        """Calculates scale factor from a fixed Q or random tuple range."""
        if isinstance(self.Q, (tuple, list)):
            q = random.uniform(self.Q[0], self.Q[1])
        else:
            q = float(self.Q)

        q = max(1.0, min(100.0, q))
        return 2.0 - q * 0.02 if q >= 50.0 else 50.0 / q

    def round_ss(self, x):
        """
        Differentiable continuous approximation to rounding:
        f(x) = x^3 when |x| < 0.5, else x.
        """
        # Periodic residual around nearest integer
        diff = x - torch.round(x)
        smooth_diff = torch.where(torch.abs(diff) < 0.5, diff ** 3, diff)
        return torch.round(x) + smooth_diff - diff

    def forward(self, image_and_cover):
        # Unpack input tuple/list or single tensor
        if isinstance(image_and_cover, (tuple, list)):
            image, cover_image = image_and_cover
        else:
            image, cover_image = image_and_cover, None

        scale_factor = self._get_scale_factor()

        # Dynamic padding, RGB -> YUV, 8x8 DCT
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        # Quantization with differentiable smooth-step rounding
        image_quantization = self.std_quantization(image_dct, scale_factor, round_func=self.round_ss)

        # De-quantization
        image_dequantization = self.std_reverse_quantization(image_quantization, scale_factor)

        # 8x8 IDCT, YUV -> RGB, un-pad, scale back to [-1, 1]
        noised_image = self.idct_rgb(image_dequantization, pad_width, pad_height)
        noised_image = noised_image.clamp(-1.0, 1.0)

        if cover_image is not None:
            return noised_image, cover_image
        return noised_image


    import torch
import torch.nn as nn


class JpegMask(JpegBasic):
    """
    Differentiable JPEG frequency masking layer.
    Zeroes out high-frequency DCT coefficients dynamically across arbitrary resolutions.
    """
    def __init__(self, Q=50, subsample=0):
        super(JpegMask, self).__init__()
        self.Q = Q
        self.subsample = subsample

        # Base 8x8 frequency cutoff masks
        base_mask = torch.zeros(1, 3, 8, 8, dtype=torch.float32)
        base_mask[:, 0:1, :5, :5] = 1.0  # Luminance cutoff (low-to-mid frequencies)
        base_mask[:, 1:3, :3, :3] = 1.0  # Chrominance cutoff
        self.register_buffer('base_mask', base_mask)

    def apply_frequency_mask(self, x):
        """Tiles the 8x8 mask correctly across arbitrary (H, W) 8x8 DCT grid."""
        B, C, H, W = x.shape
        h_blocks = H // 8
        w_blocks = W // 8

        # Correct 2D spatial block tiling: (1, 3, 1, 8, 1, 8) -> (1, 3, h_blocks, 8, w_blocks, 8)
        mask = self.base_mask.view(1, 3, 1, 8, 1, 8).repeat(1, 1, h_blocks, 1, w_blocks, 1)
        mask = mask.permute(0, 1, 2, 3, 4, 5).contiguous().view(1, 3, H, W)

        return x * mask.to(device=x.device, dtype=x.dtype)

    def forward(self, image_and_cover):
        # Handle tuple/list or single tensor input
        if isinstance(image_and_cover, (tuple, list)):
            image, cover_image = image_and_cover
        else:
            image, cover_image = image_and_cover, None

        # [-1, 1] to [0, 255], RGB -> YUV, dynamic padding, 8x8 DCT
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        # Apply low-pass frequency masking
        image_masked = self.apply_frequency_mask(image_dct)

        # 8x8 IDCT, YUV -> RGB, un-pad, scale back to [-1, 1]
        noised_image = self.idct_rgb(image_masked, pad_width, pad_height)
        noised_image = noised_image.clamp(-1.0, 1.0)

        if cover_image is not None:
            return noised_image, cover_image
        return noised_image