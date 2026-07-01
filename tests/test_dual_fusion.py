import torch

from models.dual_fusion import MultiScaleFusion

CHANNELS = [256, 512, 1024, 2048]
RESOLUTIONS = [160, 80, 40, 20]  # matches 640x640 input at strides [4,8,16,32]


def _make_fake_feats(batch=2):
    return [torch.randn(batch, c, r, r) for c, r in zip(CHANNELS, RESOLUTIONS)]


def test_fusion_output_shapes_match_input():
    torch.manual_seed(0)
    rgb_img = torch.randn(2, 3, 640, 640)
    feats_rgb = _make_fake_feats()
    feats_ir = _make_fake_feats()

    fusion = MultiScaleFusion()
    feats_fused, debug = fusion(rgb_img, feats_rgb, feats_ir)

    assert len(feats_fused) == 4
    for fused, rgb_feat in zip(feats_fused, feats_rgb):
        assert fused.shape == rgb_feat.shape

    assert debug["iv"].shape == (2,)
    assert len(debug["w_color"]) == 4
    assert len(debug["w_thermal"]) == 4


def test_fusion_output_has_no_nan_or_inf():
    torch.manual_seed(1)
    rgb_img = torch.randn(2, 3, 640, 640)
    feats_rgb = _make_fake_feats()
    feats_ir = _make_fake_feats()

    fusion = MultiScaleFusion()
    feats_fused, debug = fusion(rgb_img, feats_rgb, feats_ir)

    for fused in feats_fused:
        assert torch.isfinite(fused).all()
    assert torch.isfinite(debug["iv"]).all()


def test_ian_called_exactly_once_for_four_scales():
    torch.manual_seed(2)
    rgb_img = torch.randn(2, 3, 640, 640)
    feats_rgb = _make_fake_feats()
    feats_ir = _make_fake_feats()

    fusion = MultiScaleFusion()
    call_count = {"n": 0}

    def counting_hook(module, inputs, output):
        call_count["n"] += 1

    fusion.ian.register_forward_hook(counting_hook)
    fusion(rgb_img, feats_rgb, feats_ir)

    assert call_count["n"] == 1


class _ConstantIAN(torch.nn.Module):
    """Stub IAN that always reports a fixed iv, regardless of input image."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, rgb_img: torch.Tensor) -> torch.Tensor:
        return torch.full((rgb_img.shape[0],), self.value)


def test_high_iv_yields_higher_mean_w_color_than_low_iv():
    # NOTE: does not feed a real bright/dark image through the actual IAN
    # conv net -- an *untrained* IAN has no reason to associate extreme pixel
    # values with "brighter" in a consistent direction (verified empirically:
    # only ~45% of random seeds showed bright > dark w_color), exactly the
    # same reasoning Giai doan C's test_ian_fusion.py used to justify
    # injecting iv directly instead of relying on the untrained network. This
    # test instead swaps in a stub IAN with a fixed iv, to check that
    # MultiScaleFusion wires iv -> compute_weights -> all 4 scales in the
    # correct direction (Eq. 4), independent of IAN's (as-yet untrained)
    # actual brightness perception.
    feats_rgb = _make_fake_feats(batch=4)
    feats_ir = _make_fake_feats(batch=4)
    dummy_img = torch.zeros(4, 3, 8, 8)

    fusion = MultiScaleFusion()
    fusion.eval()

    fusion.ian = _ConstantIAN(1.0)
    with torch.no_grad():
        _, debug_high = fusion(dummy_img, feats_rgb, feats_ir)

    fusion.ian = _ConstantIAN(0.0)
    with torch.no_grad():
        _, debug_low = fusion(dummy_img, feats_rgb, feats_ir)

    mean_w_color_high = torch.stack(debug_high["w_color"]).mean()
    mean_w_color_low = torch.stack(debug_low["w_color"]).mean()

    assert mean_w_color_high.item() > mean_w_color_low.item()
