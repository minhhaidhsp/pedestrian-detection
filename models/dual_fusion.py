"""Multi-scale illumination-guided fusion between RGB/IR backbone features.

# DESIGN DECISION: fusion is applied at all 4 backbone scales (P2, P3, P4,
# P5), even though the decoder (Giai doan D, Buoc 5) later drops P5. This is
# because the hybrid encoder's CCFM (top-down FPN + bottom-up PAN) benefits
# from P5's deeper semantic context propagating down into P2/P3/P4 before P5
# is discarded -- HybridEncoder.forward is generic over the number of scales
# (only asserts len(feats) == len(in_channels)), so keeping P5 through the
# encoder costs nothing extra in code and preserves that cross-scale signal.
# P5 is cut at the decoder's input instead of here.
"""

import torch
import torch.nn as nn

from models.ian_fusion import FusionHead, IlluminationAwareNetwork

NUM_SCALES = 4  # P2, P3, P4, P5


class MultiScaleFusion(nn.Module):
    """Fuses paired RGB/IR feature maps at P2/P3/P4/P5 via illumination gating.

    Holds one shared IlluminationAwareNetwork (iv is a whole-image property,
    computed once) and 4 independent FusionHead instances -- one per scale,
    each with its own trainable alpha/beta -- per the DESIGN DECISION in
    models/ian_fusion.py (FusionHead docstring).
    """

    def __init__(
        self,
        ian_in_channels: int = 3,
        ian_hidden_channels: int = 32,
        alpha_init: float = 1.0,
        beta_init: float = 0.1,
    ):
        super().__init__()
        self.ian = IlluminationAwareNetwork(ian_in_channels, ian_hidden_channels)
        self.fusion_heads = nn.ModuleList(
            [FusionHead(alpha_init=alpha_init, beta_init=beta_init) for _ in range(NUM_SCALES)]
        )

    def forward(
        self, rgb_img: torch.Tensor, feats_rgb: list[torch.Tensor], feats_ir: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], dict]:
        """rgb_img: [B,3,H,W] (full-resolution RGB image, for IAN).
        feats_rgb, feats_ir: each a list of 4 tensors [P2, P3, P4, P5] (paired shapes).

        Returns (feats_fused, debug_dict) where feats_fused is a list of 4
        fused tensors, and debug_dict = {"iv": iv, "w_color": [...], "w_thermal": [...]}
        (one w_color/w_thermal entry per scale, same order as input).
        """
        assert len(feats_rgb) == NUM_SCALES and len(feats_ir) == NUM_SCALES

        iv = self.ian(rgb_img)  # computed once, shared across all scales

        feats_fused = []
        w_color_list = []
        w_thermal_list = []
        for head, f_rgb, f_ir in zip(self.fusion_heads, feats_rgb, feats_ir):
            f_fusion, debug = head(iv, f_rgb, f_ir)
            feats_fused.append(f_fusion)
            w_color_list.append(debug["w_color"])
            w_thermal_list.append(debug["w_thermal"])

        debug_dict = {"iv": iv, "w_color": w_color_list, "w_thermal": w_thermal_list}
        return feats_fused, debug_dict
