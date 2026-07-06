"""FA-PromptDETR: full assembly of FA-VP + IAN fusion + RT-DETR-based DETR.

Wires together (Giai doan D):
  rgb_img, ir_img
    -> FAVPModule_rgb(rgb_img), FAVPModule_ir(ir_img) -> Pfreq_rgb, Pfreq_ir
    -> DualStreamBackbone(rgb_img, ir_img) -> feats_rgb, feats_ir  [P2,P3,P4,P5]
    -> MultiScaleFusion(rgb_img, feats_rgb, feats_ir) -> feats_fused  [P2,P3,P4,P5]
    -> HybridEncoder(feats_fused, freq_prompts=Pfreq_rgb+Pfreq_ir) -> feats_encoded  [P2,P3,P4,P5]
    -> feats_encoded[:3]  (drop P5, Quyet dinh A)
    -> RTDETRTransformer(feats_encoded[:3]) -> {pred_logits, pred_boxes, enc_pred_logits, enc_pred_boxes}

# KNOWN, ACCEPTED DEAD BRANCH (Buoc 6, confirmed via test_fa_promptdetr.py's
# full-gradient check): `encoder.downsample_convs[2]`/`encoder.pan_blocks[2]`
# (HybridEncoder's LAST bottom-up PAN stage, ~707K params / ~1.3% of total
# model params) never receive gradient. HybridEncoder's CCFM has two stages:
# top-down FPN (P5->P4->P3->P2, which DOES correctly propagate P5's semantic
# context into P2/P3/P4 -- verified NOT dead) and bottom-up PAN (P2->P3->P4->
# P5, whose LAST step re-produces a P5-level output). Since we drop P5 right
# after the encoder (feats_encoded[:3]), that final PAN-stage output is
# computed but never consumed by anything. This is a deliberate, accepted
# trade-off (Buoc 6 decision) to keep HybridEncoder an unmodified, faithful
# copy of upstream RT-DETR's CCFM (already tested in Buoc 4) rather than
# special-casing it to know P5 will be dropped downstream -- NOT a
# gradient-severing bug like the ones found/fixed in Buoc 5. If reporting
# param counts/FLOPs (paper Table 1), note this dead branch is included in
# any measurement of the whole model (it cannot be cleanly excluded without
# modifying HybridEncoder), so a naive count/FLOPs measurement will read
# slightly higher than a hand-optimized version would -- this is expected,
# not a bug.

# DESIGN DECISION: the paper does not specify whether FA-VP (Frequency-Aware
# Visual Prompting) is applied to the RGB image, the IR image, or both. This
# reproduction applies it to BOTH, via two independent (non-weight-sharing)
# FAVPModule instances -- the same "no sharing across modalities" principle
# used for DualStreamBackbone. Rationale: both domains carry modality-specific
# frequency content worth prompting on (RGB: high-frequency texture/detail;
# IR: smooth, low-frequency thermal gradients), and separate instances let
# each learn a prompt suited to its own domain's frequency statistics. This
# MUST be stated explicitly in the paper reproduction's Implementation
# Details, since the paper itself does not define it.

# DESIGN DECISION: Pfreq is injected into the HYBRID ENCODER (right after its
# input_proj, before AIFI/CCFM -- see models/encoder.py), not into the
# backbone. Consequently FAVPModule's target_shapes must be the spatial
# resolution AFTER input_proj, not the backbone's own output resolution --
# these happen to be numerically identical (input_proj is a 1x1 conv,
# stride 1, so H/W are unchanged; only the channel count changes, from the
# backbone's [256,512,1024,2048] to the encoder's hidden_dim=128), but this
# is a coincidence of this architecture, not a rule -- get target_shapes from
# each scale's known stride, not from "whatever shape the backbone happens to
# output before the encoder".

# DESIGN DECISION: FAVPModule produces one Pfreq list per modality; but
# HybridEncoder.forward (Buoc 4) takes a single `freq_prompts` list (assumed
# one prompt source). Rather than extending HybridEncoder's API to accept two
# separate lists (more flexible, but requires re-touching and re-testing
# already-verified Buoc 4 code), Pfreq_rgb and Pfreq_ir are summed per scale
# before injection: Pfreq_total[i] = Pfreq_rgb[i] + Pfreq_ir[i]. This is safe
# because injection itself is a simple addition into proj_feats[i] -- summing
# the two sources first and injecting once is mathematically identical to
# injecting each separately (addition is associative), so nothing is lost by
# doing it this way, and Buoc 4's HybridEncoder needs no changes.
"""

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from losses.detr_loss import HungarianMatcher, SetCriterion


def _finite_stats(tensor: torch.Tensor) -> str:
    """Human-readable min/max (over finite entries only) + NaN/Inf counts."""
    finite_mask = torch.isfinite(tensor)
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    if finite_mask.any():
        finite_vals = tensor[finite_mask]
        return (
            f"min={finite_vals.min().item():.6f} max={finite_vals.max().item():.6f} "
            f"(n_nan={n_nan} n_inf={n_inf} of {tensor.numel()})"
        )
    return f"ALL non-finite (n_nan={n_nan} n_inf={n_inf} of {tensor.numel()})"
from models.backbone import DualStreamBackbone
from models.decoder import RTDETRTransformer
from models.dual_fusion import MultiScaleFusion
from models.encoder import HybridEncoder
from models.fa_vp import FAVPModule

# Fixed by PResNet's 4-stage design (Quyet dinh A: return_idx=[0,1,2,3]) --
# not an independent tunable hyperparameter, so not read from config.
BACKBONE_STRIDES = [4, 8, 16, 32]  # P2, P3, P4, P5
# AIFI runs on the fewest-token scale among the 4 -- with P2 added, that is
# now P5 (last index), not index 2 as in upstream's original 3-scale default
# (see Buoc 4 report). Also architectural, not a sweep-worthy hyperparameter.
AIFI_ENCODER_IDX = [3]


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class FAPromptDETR(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        data_cfg = config["data"]
        model_cfg = config["model"]
        fa_vp_cfg = model_cfg["fa_vp"]
        ian_cfg = model_cfg["ian_fusion"]
        encoder_cfg = model_cfg["encoder"]
        decoder_cfg = model_cfg["decoder"]

        self.num_classes = data_cfg["num_classes"]
        hidden_dim = decoder_cfg["hidden_dim"]

        self.backbone = DualStreamBackbone(
            depth=model_cfg.get("backbone_depth", 50),
            variant=model_cfg.get("backbone_variant", "d"),
            pretrained=model_cfg.get("pretrained_backbone", True),
        )
        backbone_channels = self.backbone.out_channels  # [256,512,1024,2048]

        # Two independent FA-VP instances, one per modality (see module docstring).
        self.favp_rgb = FAVPModule(
            gamma=fa_vp_cfg["gamma"], in_channels=3, prompt_dim=hidden_dim
        )
        self.favp_ir = FAVPModule(
            gamma=fa_vp_cfg["gamma"], in_channels=3, prompt_dim=hidden_dim
        )

        self.fusion = MultiScaleFusion(
            ian_in_channels=3,
            alpha_init=ian_cfg["alpha"],
            beta_init=ian_cfg["beta"],
        )

        self.encoder = HybridEncoder(
            in_channels=backbone_channels,
            feat_strides=BACKBONE_STRIDES,
            hidden_dim=hidden_dim,
            nhead=encoder_cfg["nhead"],
            dim_feedforward=encoder_cfg["dim_feedforward"],
            dropout=encoder_cfg["dropout"],
            enc_act=encoder_cfg["enc_act"],
            use_encoder_idx=AIFI_ENCODER_IDX,
            num_encoder_layers=encoder_cfg["num_encoder_layers"],
            expansion=encoder_cfg["expansion"],
            depth_mult=encoder_cfg["depth_mult"],
            act=encoder_cfg["act"],
        )

        # Decoder consumes only P2/P3/P4 (drop P5, Quyet dinh A).
        self.decoder = RTDETRTransformer(
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            num_queries=decoder_cfg["num_queries"],
            feat_channels=[hidden_dim, hidden_dim, hidden_dim],
            feat_strides=BACKBONE_STRIDES[:3],
            num_levels=3,
            num_decoder_points=decoder_cfg["num_decoder_points"],
            nhead=decoder_cfg["nhead"],
            num_decoder_layers=decoder_cfg["num_decoder_layers"],
            dim_feedforward=decoder_cfg["dim_feedforward"],
            dropout=decoder_cfg["dropout"],
            activation=decoder_cfg["activation"],
        )

        loss_cfg = config["loss"]
        matcher = HungarianMatcher(
            cost_class=loss_cfg["cls_weight"], cost_bbox=loss_cfg["l1_weight"], cost_giou=loss_cfg["giou_weight"]
        )
        weight_dict = {
            "loss_ce": loss_cfg["cls_weight"],
            "loss_bbox": loss_cfg["l1_weight"],
            "loss_giou": loss_cfg["giou_weight"],
        }
        # Same SetCriterion/weight_dict is applied to both the decoder output
        # (Eq. 7-8) and the encoder query-selection proposals (structurally
        # required, see models/decoder.py DESIGN DECISION (2), Buoc 5) --
        # RT-DETR upstream convention, since the paper defines no separate
        # formula for query-selection supervision.
        self.criterion = SetCriterion(matcher, weight_dict=weight_dict)

        # NaN/Inf diagnostics (Giai doan F, CUDA NaN investigation): stats from
        # the last step where all 4 outputs were finite, so that when a
        # NaN/Inf DOES appear we can report the trend (was it climbing toward
        # this, or a sudden jump?) rather than just the crash frame alone.
        self._last_finite_stats: dict[str, str] | None = None

    def _freq_prompt_target_shapes(self, height: int, width: int) -> list[tuple[int, int]]:
        return [(height // s, width // s) for s in BACKBONE_STRIDES]

    def forward(self, rgb_img: torch.Tensor, ir_img: torch.Tensor, targets: list[dict] | None = None):
        height, width = rgb_img.shape[-2:]
        target_shapes = self._freq_prompt_target_shapes(height, width)

        pfreq_rgb, _, _ = self.favp_rgb(rgb_img, target_shapes)
        pfreq_ir, _, _ = self.favp_ir(ir_img, target_shapes)
        pfreq_total = [p_rgb + p_ir for p_rgb, p_ir in zip(pfreq_rgb, pfreq_ir)]

        feats_rgb, feats_ir = self.backbone(rgb_img, ir_img)
        feats_fused, _fusion_debug = self.fusion(rgb_img, feats_rgb, feats_ir)
        feats_encoded = self.encoder(feats_fused, freq_prompts=pfreq_total)

        outputs = self.decoder(feats_encoded[:3])

        if targets is None:
            return outputs

        decoder_out = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}
        encoder_out = {"pred_logits": outputs["enc_pred_logits"], "pred_boxes": outputs["enc_pred_boxes"]}

        # NaN/Inf guard, checked right before matching/loss (Giai doan F): a
        # NaN/Inf cost matrix inside HungarianMatcher's linear_sum_assignment
        # is a *symptom* -- this catches it one step earlier, at its actual
        # source (the model's own output), and reports which images were in
        # the offending batch plus the value trend vs. the last good step.
        to_check = {
            "decoder pred_logits": outputs["pred_logits"],
            "decoder pred_boxes": outputs["pred_boxes"],
            "encoder pred_logits": outputs["enc_pred_logits"],
            "encoder pred_boxes": outputs["enc_pred_boxes"],
        }
        non_finite = {name: t for name, t in to_check.items() if not torch.isfinite(t).all()}
        if non_finite:
            image_ids = [t.get("image_id") for t in targets]
            lines = [f"NaN/Inf detected in model output(s): {list(non_finite.keys())}"]
            lines.append(f"image_ids in this batch: {image_ids}")
            for name, tensor in to_check.items():
                lines.append(f"  current {name}: {_finite_stats(tensor)}")
            if self._last_finite_stats is not None:
                for name in to_check:
                    lines.append(f"  previous good step {name}: {self._last_finite_stats.get(name, 'n/a')}")
            else:
                lines.append("  (no previous good step recorded -- NaN/Inf on the very first batch)")
            raise RuntimeError("\n".join(lines))

        self._last_finite_stats = {name: _finite_stats(t) for name, t in to_check.items()}

        decoder_losses = self.criterion(decoder_out, targets)
        encoder_losses = self.criterion(encoder_out, targets)

        loss_dict = {f"decoder_{k}": v for k, v in decoder_losses.items()}
        loss_dict.update({f"encoder_{k}": v for k, v in encoder_losses.items()})
        loss_dict["loss_total"] = decoder_losses["loss_total"] + encoder_losses["loss_total"]

        return outputs, loss_dict
