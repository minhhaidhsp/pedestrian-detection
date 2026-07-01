import torch

from models.ian_fusion import IlluminationAwareNetwork, IlluminationGuidedFusion


def test_iv_in_unit_range():
    torch.manual_seed(0)
    rgb = torch.randn(4, 3, 32, 32)
    ian = IlluminationAwareNetwork()
    iv = ian(rgb)
    assert iv.shape == (4,)
    assert torch.all(iv >= 0.0) and torch.all(iv <= 1.0)


def test_weights_sum_to_one():
    torch.manual_seed(0)
    fusion = IlluminationGuidedFusion()
    iv = torch.rand(8)
    w_color, w_thermal = fusion.compute_weights(iv)
    assert torch.allclose(w_color + w_thermal, torch.ones(8), atol=1e-6)


def test_fusion_output_shape():
    torch.manual_seed(0)
    rgb_img = torch.randn(2, 3, 32, 32)
    f_rgb = torch.randn(2, 16, 8, 8)
    f_ir = torch.randn(2, 16, 8, 8)
    fusion = IlluminationGuidedFusion()

    f_fusion, debug = fusion(rgb_img, f_rgb, f_ir)

    assert f_fusion.shape == f_rgb.shape
    assert debug["iv"].shape == (2,)
    assert debug["w_color"].shape == (2,)
    assert debug["w_thermal"].shape == (2,)


def test_bright_scene_favors_color_dark_scene_favors_thermal():
    fusion = IlluminationGuidedFusion()

    iv_bright = torch.tensor([1.0])
    iv_dark = torch.tensor([0.0])

    w_color_bright, w_thermal_bright = fusion.compute_weights(iv_bright)
    w_color_dark, w_thermal_dark = fusion.compute_weights(iv_dark)

    assert w_color_bright.item() > w_thermal_bright.item()
    assert w_color_dark.item() < w_thermal_dark.item()
