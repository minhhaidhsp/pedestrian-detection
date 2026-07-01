import torch

from models.encoder import HybridEncoder
from models.fa_vp import FAVPModule

IN_CHANNELS = [256, 512, 1024, 2048]  # P2, P3, P4, P5 (DualStreamBackbone/MultiScaleFusion output)
FEAT_STRIDES = [4, 8, 16, 32]
HIDDEN_DIM = 128
INPUT_SIZE = 640
RESOLUTIONS = [INPUT_SIZE // s for s in FEAT_STRIDES]  # [160, 80, 40, 20]

# P5 (index 3, stride 32) has the fewest tokens of the 4 scales: 20*20=400,
# vs P4 40*40=1600, P3 80*80=6400, P2 160*160=25600 -- so AIFI must run there.
P5_INDEX = 3
P5_TOKEN_COUNT = RESOLUTIONS[P5_INDEX] ** 2


def _make_backbone_feats(batch=2):
    return [torch.randn(batch, c, r, r) for c, r in zip(IN_CHANNELS, RESOLUTIONS)]


def _make_encoder(**overrides):
    kwargs = dict(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=HIDDEN_DIM,
        nhead=4,
        dim_feedforward=256,
        use_encoder_idx=[P5_INDEX],
        num_encoder_layers=1,
        eval_spatial_size=None,
    )
    kwargs.update(overrides)
    return HybridEncoder(**kwargs)


def test_encoder_without_freq_prompts_output_shapes():
    torch.manual_seed(0)
    feats = _make_backbone_feats()
    encoder = _make_encoder()
    encoder.eval()

    with torch.no_grad():
        outs = encoder(feats, freq_prompts=None)

    assert len(outs) == 4
    for out, res in zip(outs, RESOLUTIONS):
        assert out.shape == (2, HIDDEN_DIM, res, res)


def test_encoder_with_freq_prompts_shapes_unchanged_and_finite():
    torch.manual_seed(1)
    feats = _make_backbone_feats()
    freq_prompts = [torch.randn(2, HIDDEN_DIM, r, r) for r in RESOLUTIONS]

    encoder = _make_encoder()
    encoder.eval()

    with torch.no_grad():
        outs = encoder(feats, freq_prompts=freq_prompts)

    assert len(outs) == 4
    for out, res in zip(outs, RESOLUTIONS):
        assert out.shape == (2, HIDDEN_DIM, res, res)
        assert torch.isfinite(out).all()


def test_freq_prompts_change_the_output():
    torch.manual_seed(2)
    feats = _make_backbone_feats()
    freq_prompts = [torch.randn(2, HIDDEN_DIM, r, r) for r in RESOLUTIONS]

    encoder = _make_encoder()
    encoder.eval()

    with torch.no_grad():
        outs_without = encoder(feats, freq_prompts=None)
        outs_with = encoder(feats, freq_prompts=freq_prompts)

    any_different = any(
        not torch.allclose(a, b, atol=1e-6) for a, b in zip(outs_without, outs_with)
    )
    assert any_different


def test_favp_module_produces_correctly_shaped_prompts_for_all_four_scales():
    torch.manual_seed(3)
    rgb_img = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    target_shapes = [(r, r) for r in RESOLUTIONS]

    favp = FAVPModule(gamma=1.0, in_channels=3, prompt_dim=HIDDEN_DIM, num_groups=8)
    prompts, i_low, i_high = favp(rgb_img, target_shapes)

    assert len(prompts) == 4
    for prompt, res in zip(prompts, RESOLUTIONS):
        assert prompt.shape == (2, HIDDEN_DIM, res, res)


def test_aifi_runs_on_p5_the_fewest_token_scale():
    torch.manual_seed(4)
    feats = _make_backbone_feats()
    encoder = _make_encoder()
    encoder.eval()

    seen_lengths = []

    def hook(module, inputs, output):
        src = inputs[0]  # [B, Len, C]
        seen_lengths.append(src.shape[1])

    encoder.encoder[0].register_forward_hook(hook)

    with torch.no_grad():
        encoder(feats, freq_prompts=None)

    assert len(seen_lengths) == 1
    assert seen_lengths[0] == P5_TOKEN_COUNT
    # sanity: this is indeed the smallest token count among the 4 scales
    all_token_counts = [r * r for r in RESOLUTIONS]
    assert seen_lengths[0] == min(all_token_counts)
