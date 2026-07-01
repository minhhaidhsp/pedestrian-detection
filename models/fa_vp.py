"""Frequency-Aware Visual Prompting (FA-VP), paper Eq. (1)-(3).

Eq. (1): I_L = argmin_{I_L} ||I - I_L||_F^2 + gamma * ||grad(I_L)||_F^2
    Closed-form (Tikhonov regularization) via FFT:
        I_L = F^-1[ F(I) / (1 + gamma * |k|^2) ]
    where |k|^2 is the squared spatial frequency grid (2*pi*f)^2 over both H, W.
Eq. (2): I_H = I - I_L
Eq. (3): H_n = Encoder_n(H_{n-1}) + Pfreq_n
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TikhonovFilter(nn.Module):
    """Splits an image into low- and high-frequency components (Eq. 1-2)."""

    def __init__(self, gamma: float = 1.0, learnable_gamma: bool = False):
        super().__init__()
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(gamma)))
        self.learnable_gamma = learnable_gamma

    def _freq_grid(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Squared spatial frequency grid |k|^2 = (2*pi*fy)^2 + (2*pi*fx)^2."""
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(w, device=device, dtype=dtype)
        ky, kx = torch.meshgrid(fy, fx, indexing="ij")
        k_sq = (2 * torch.pi * ky) ** 2 + (2 * torch.pi * kx) ** 2
        return k_sq  # [H, W]

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """img: [B, C, H, W] -> (I_low, I_high), both [B, C, H, W]."""
        b, c, h, w = img.shape
        k_sq = self._freq_grid(h, w, img.device, img.dtype)  # [H, W]

        gamma = self.gamma.clamp(min=0.0) if self.learnable_gamma else self.gamma
        denom = 1.0 + gamma * k_sq  # [H, W]

        img_fft = torch.fft.fft2(img, dim=(-2, -1))  # [B, C, H, W] complex
        low_fft = img_fft / denom.to(img_fft.dtype)
        i_low = torch.fft.ifft2(low_fft, dim=(-2, -1)).real
        i_high = img - i_low
        return i_low, i_high


class FrequencyPromptProjector(nn.Module):
    """Projects I_high into per-encoder-layer frequency prompts Pfreq_n (Eq. 3)."""

    def __init__(self, in_channels: int = 3, prompt_dim: int = 128, num_groups: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, prompt_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1),
        )

    def forward(self, i_high: torch.Tensor, target_shapes: list[tuple[int, int]]) -> list[torch.Tensor]:
        """i_high: [B, C, H, W]. target_shapes: list of (H_n, W_n) per encoder layer.

        Returns list of Pfreq_n, each [B, prompt_dim, H_n, W_n].
        """
        feat = self.proj(i_high)
        prompts = [
            F.interpolate(feat, size=shape, mode="bilinear", align_corners=False)
            for shape in target_shapes
        ]
        return prompts


class FAVPModule(nn.Module):
    """Combines TikhonovFilter + FrequencyPromptProjector into one module."""

    def __init__(
        self,
        gamma: float = 1.0,
        learnable_gamma: bool = False,
        in_channels: int = 3,
        prompt_dim: int = 128,
        num_groups: int = 8,
    ):
        super().__init__()
        self.tikhonov = TikhonovFilter(gamma=gamma, learnable_gamma=learnable_gamma)
        self.projector = FrequencyPromptProjector(
            in_channels=in_channels, prompt_dim=prompt_dim, num_groups=num_groups
        )

    def forward(
        self, img: torch.Tensor, target_shapes: list[tuple[int, int]]
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """img: [B, C, H, W]. Returns (Pfreq_list, I_low, I_high)."""
        i_low, i_high = self.tikhonov(img)
        prompts = self.projector(i_high, target_shapes)
        return prompts, i_low, i_high
