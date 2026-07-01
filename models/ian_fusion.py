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
        # bias=False on both convs: a conv bias immediately followed by
        # BatchNorm is mathematically redundant (BN's mean-subtraction cancels
        # any constant per-channel shift the bias would add), so with bias=True
        # here that parameter's gradient is always ~0 (confirmed empirically,
        # Giai doan D Buoc 6 test_fa_promptdetr.py) -- permanently-dead weight,
        # not a training bug elsewhere.
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
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


class FusionHead(nn.Module):
    """Applies illumination-gated fusion (Eq. 4-6) given an externally computed iv.

    # DESIGN DECISION: iv (illumination value) is a single scalar describing
    # the whole image's brightness -- it does not depend on feature scale, so
    # it should be computed ONCE (by a shared IlluminationAwareNetwork) and
    # reused across every scale, instead of re-running IAN once per scale
    # (wasteful, and would let different scales silently disagree on the same
    # image's brightness). alpha/beta, in contrast, are kept PER SCALE: how
    # much a given scale should trust color vs. thermal information can
    # reasonably differ between a shallow, high-resolution feature map (P2)
    # and a deep, semantic one (P5), so each scale gets its own trainable
    # alpha/beta pair. See MultiScaleFusion in models/dual_fusion.py.
    """

    def __init__(self, alpha_init: float = 1.0, beta_init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def compute_weights(self, iv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """iv: [B] -> (w_color, w_thermal), each [B]."""
        beta = self.beta.clamp(min=MIN_BETA)
        w_color = 1.0 / (1.0 + self.alpha * torch.exp(-(iv - 0.5) / beta))
        w_thermal = 1.0 - w_color
        return w_color, w_thermal

    def forward(
        self, iv: torch.Tensor, f_rgb: torch.Tensor, f_ir: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """iv: [B] (precomputed). f_rgb, f_ir: [B, C, H, W] (same shape).

        Returns (f_fusion, debug_dict) with debug_dict = {w_color, w_thermal}.
        """
        w_color, w_thermal = self.compute_weights(iv)
        w_color_b = w_color.view(-1, 1, 1, 1)
        w_thermal_b = w_thermal.view(-1, 1, 1, 1)
        f_fusion = w_color_b * f_rgb + w_thermal_b * f_ir
        return f_fusion, {"w_color": w_color, "w_thermal": w_thermal}


class IlluminationGuidedFusion(nn.Module):
    """Fuses RGB/IR feature maps using illumination-conditioned sigmoid gating.

    Standalone single-scale convenience wrapper around IlluminationAwareNetwork
    + FusionHead (computes its own iv from rgb_img on every call). For the
    multi-scale case where iv should be computed once and shared, use
    IlluminationAwareNetwork + FusionHead directly (see MultiScaleFusion in
    models/dual_fusion.py) instead of one IlluminationGuidedFusion per scale.
    """

    def __init__(self, ian_in_channels: int = 3, ian_hidden_channels: int = 32):
        super().__init__()
        self.ian = IlluminationAwareNetwork(ian_in_channels, ian_hidden_channels)
        self.head = FusionHead(alpha_init=1.0, beta_init=0.1)

    def compute_weights(self, iv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """iv: [B] -> (w_color, w_thermal), each [B]."""
        return self.head.compute_weights(iv)

    def forward(
        self, rgb_img: torch.Tensor, f_rgb: torch.Tensor, f_ir: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """rgb_img: [B,3,H,W] (raw RGB image fed to IAN).
        f_rgb, f_ir: [B, C, H', W'] feature maps to fuse (same shape).
        Returns (f_fusion, debug_dict) with debug_dict = {iv, w_color, w_thermal}.
        """
        iv = self.ian(rgb_img)
        f_fusion, debug_dict = self.head(iv, f_rgb, f_ir)
        debug_dict = {"iv": iv, **debug_dict}
        return f_fusion, debug_dict
