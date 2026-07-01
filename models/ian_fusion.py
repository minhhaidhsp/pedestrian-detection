"""Illumination-Aware Network + middle fusion, paper Eq. (4)-(6).

Eq. (4): w_color = 1 / (1 + alpha * exp(-(iv - 0.5) / beta))
Eq. (5): w_thermal = 1 - w_color
Eq. (6): F_fusion = w_color * F_RGB + w_thermal * F_IR
"""

import torch
import torch.nn as nn

MIN_BETA = 1e-3


class IlluminationAwareNetwork(nn.Module):
    """Predicts a scalar illumination value iv in [0, 1] from the RGB image."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(hidden_channels * 2, 1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: [B, 3, H, W] -> iv: [B] in [0, 1]."""
        feat = self.backbone(rgb).flatten(1)  # [B, hidden*2]
        iv = torch.sigmoid(self.fc(feat)).squeeze(-1)  # [B]
        return iv


class IlluminationGuidedFusion(nn.Module):
    """Fuses RGB/IR feature maps using illumination-conditioned sigmoid gating."""

    def __init__(self, ian_in_channels: int = 3, ian_hidden_channels: int = 32):
        super().__init__()
        self.ian = IlluminationAwareNetwork(ian_in_channels, ian_hidden_channels)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def compute_weights(self, iv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """iv: [B] -> (w_color, w_thermal), each [B]."""
        beta = self.beta.clamp(min=MIN_BETA)
        w_color = 1.0 / (1.0 + self.alpha * torch.exp(-(iv - 0.5) / beta))
        w_thermal = 1.0 - w_color
        return w_color, w_thermal

    def forward(
        self, rgb_img: torch.Tensor, f_rgb: torch.Tensor, f_ir: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """rgb_img: [B,3,H,W] (raw RGB image fed to IAN).
        f_rgb, f_ir: [B, C, H', W'] feature maps to fuse (same shape).
        Returns (f_fusion, debug_dict) with debug_dict = {iv, w_color, w_thermal}.
        """
        iv = self.ian(rgb_img)
        w_color, w_thermal = self.compute_weights(iv)

        w_color_b = w_color.view(-1, 1, 1, 1)
        w_thermal_b = w_thermal.view(-1, 1, 1, 1)
        f_fusion = w_color_b * f_rgb + w_thermal_b * f_ir

        debug_dict = {"iv": iv, "w_color": w_color, "w_thermal": w_thermal}
        return f_fusion, debug_dict
