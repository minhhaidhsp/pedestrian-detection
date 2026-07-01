import torch

from wise_od.interpolate import _check_compatible_state_dicts, wise_interpolate


def _make_checkpoint(tmp_path, name: str, state_dict: dict):
    path = tmp_path / name
    torch.save(state_dict, path)
    return path


def test_lambda_zero_returns_theta_zs(tmp_path):
    torch.manual_seed(0)
    sd_zs = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    sd_ft = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    zs_path = _make_checkpoint(tmp_path, "zs.pth", sd_zs)
    ft_path = _make_checkpoint(tmp_path, "ft.pth", sd_ft)

    result = wise_interpolate(zs_path, ft_path, lambda_interp=0.0)

    for key in sd_zs:
        assert torch.allclose(result[key], sd_zs[key])


def test_lambda_one_returns_theta_ft(tmp_path):
    torch.manual_seed(1)
    sd_zs = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    sd_ft = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    zs_path = _make_checkpoint(tmp_path, "zs.pth", sd_zs)
    ft_path = _make_checkpoint(tmp_path, "ft.pth", sd_ft)

    result = wise_interpolate(zs_path, ft_path, lambda_interp=1.0)

    for key in sd_ft:
        assert torch.allclose(result[key], sd_ft[key])


def test_lambda_half_is_average(tmp_path):
    torch.manual_seed(2)
    sd_zs = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    sd_ft = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    zs_path = _make_checkpoint(tmp_path, "zs.pth", sd_zs)
    ft_path = _make_checkpoint(tmp_path, "ft.pth", sd_ft)

    result = wise_interpolate(zs_path, ft_path, lambda_interp=0.5)

    for key in sd_zs:
        expected = 0.5 * sd_zs[key] + 0.5 * sd_ft[key]
        assert torch.allclose(result[key], expected)


def test_raises_on_mismatched_keys():
    sd1 = {"w": torch.randn(4, 4)}
    sd2 = {"w": torch.randn(4, 4), "b": torch.randn(4)}

    try:
        _check_compatible_state_dicts(sd1, sd2)
        assert False, "expected ValueError for mismatched keys"
    except ValueError:
        pass
