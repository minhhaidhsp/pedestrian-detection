import torch

from models.fa_vp import FAVPModule, FrequencyPromptProjector, TikhonovFilter


def _gradient_energy(img: torch.Tensor) -> torch.Tensor:
    """Sum of squared finite-difference gradients, as a smoothness proxy."""
    dy = img[..., 1:, :] - img[..., :-1, :]
    dx = img[..., :, 1:] - img[..., :, :-1]
    return dy.pow(2).sum() + dx.pow(2).sum()


def test_low_high_reconstruct_original():
    torch.manual_seed(0)
    img = torch.randn(2, 3, 64, 64)
    filt = TikhonovFilter(gamma=1.0)
    i_low, i_high = filt(img)
    assert torch.allclose(i_low + i_high, img, atol=1e-4)


def test_larger_gamma_gives_smoother_low_freq():
    torch.manual_seed(0)
    img = torch.randn(2, 3, 64, 64)

    low_small_gamma, _ = TikhonovFilter(gamma=0.1)(img)
    low_large_gamma, _ = TikhonovFilter(gamma=10.0)(img)

    energy_small = _gradient_energy(low_small_gamma)
    energy_large = _gradient_energy(low_large_gamma)
    assert energy_large < energy_small

    var_small = low_small_gamma.var()
    var_large = low_large_gamma.var()
    assert var_large < var_small


def test_favp_module_output_shapes():
    torch.manual_seed(0)
    img = torch.randn(2, 3, 64, 64)
    target_shapes = [(32, 32), (16, 16), (8, 8)]
    module = FAVPModule(gamma=1.0, in_channels=3, prompt_dim=16, num_groups=4)

    prompts, i_low, i_high = module(img, target_shapes)

    assert len(prompts) == len(target_shapes)
    for prompt, shape in zip(prompts, target_shapes):
        assert prompt.shape == (2, 16, shape[0], shape[1])
    assert i_low.shape == img.shape
    assert i_high.shape == img.shape


def test_frequency_prompt_projector_standalone():
    torch.manual_seed(0)
    i_high = torch.randn(2, 3, 64, 64)
    target_shapes = [(20, 20), (10, 10)]
    projector = FrequencyPromptProjector(in_channels=3, prompt_dim=8, num_groups=4)

    prompts = projector(i_high, target_shapes)

    assert len(prompts) == 2
    assert prompts[0].shape == (2, 8, 20, 20)
    assert prompts[1].shape == (2, 8, 10, 10)


def test_learnable_gamma_is_parameter():
    filt = TikhonovFilter(gamma=1.0, learnable_gamma=True)
    assert isinstance(filt.gamma, torch.nn.Parameter)
