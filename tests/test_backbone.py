import torch

from models.backbone import DualStreamBackbone

EXPECTED_CHANNELS = [256, 512, 1024, 2048]
EXPECTED_STRIDES = [4, 8, 16, 32]


def test_dual_backbone_output_scales_channels_and_strides():
    torch.manual_seed(0)
    input_size = 640
    rgb_img = torch.randn(2, 3, input_size, input_size)
    ir_img = torch.randn(2, 3, input_size, input_size)

    backbone = DualStreamBackbone(depth=50, variant="d", pretrained=True)
    backbone.eval()
    with torch.no_grad():
        feats_rgb, feats_ir = backbone(rgb_img, ir_img)

    assert len(feats_rgb) == 4
    assert len(feats_ir) == 4

    expected_resolutions = [input_size // s for s in EXPECTED_STRIDES]
    for feats, name in [(feats_rgb, "rgb"), (feats_ir, "ir")]:
        for i, (feat, channels, res) in enumerate(zip(feats, EXPECTED_CHANNELS, expected_resolutions)):
            assert feat.shape == (2, channels, res, res), (
                f"{name} scale {i}: expected (2,{channels},{res},{res}), got {tuple(feat.shape)}"
            )


def test_dual_backbone_streams_do_not_share_weights():
    backbone = DualStreamBackbone(depth=50, variant="d", pretrained=False)

    rgb_params = dict(backbone.backbone_rgb.named_parameters())
    ir_params = dict(backbone.backbone_ir.named_parameters())

    assert set(rgb_params.keys()) == set(ir_params.keys())
    for name in rgb_params:
        assert rgb_params[name].data_ptr() != ir_params[name].data_ptr(), (
            f"Parameter '{name}' shares memory between backbone_rgb and backbone_ir"
        )

    # Sanity: same random init distribution but different values (independent instances).
    sample_key = next(iter(rgb_params))
    assert not torch.equal(rgb_params[sample_key], ir_params[sample_key])
