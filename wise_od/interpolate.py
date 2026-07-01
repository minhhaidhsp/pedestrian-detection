"""Weight-Space Ensembling (WiSE-OD) between zero-shot and fine-tuned weights, Eq. (9).

Eq. (9): theta = (1 - lambda) * theta_ZS + lambda * theta_FT
    lambda_interp = 0 -> theta_ZS (zero-shot / base weights)
    lambda_interp = 1 -> theta_FT (fully fine-tuned weights)
"""

from pathlib import Path

import torch


def _load_state_dict(source):
    """source: a state_dict, or a path to a checkpoint file loadable via torch.load."""
    if isinstance(source, (str, Path)):
        return torch.load(source, map_location="cpu")
    return source


def _check_compatible_state_dicts(sd1: dict, sd2: dict) -> None:
    """Raises ValueError if the two state dicts don't have matching keys/shapes."""
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    if keys1 != keys2:
        only_in_1 = sorted(keys1 - keys2)
        only_in_2 = sorted(keys2 - keys1)
        raise ValueError(
            f"State dicts have mismatched keys. Only in first: {only_in_1}; "
            f"only in second: {only_in_2}"
        )
    for key in keys1:
        if sd1[key].shape != sd2[key].shape:
            raise ValueError(f"Shape mismatch for key '{key}': {sd1[key].shape} vs {sd2[key].shape}")


def wise_interpolate(theta_zs, theta_ft, lambda_interp: float) -> dict:
    """Interpolates between zero-shot (theta_zs) and fine-tuned (theta_ft) weights (Eq. 9).

    theta_zs, theta_ft: state_dict, or a path to a checkpoint file (torch.load).
    lambda_interp: weight in [0, 1]; 0 -> theta_zs, 1 -> theta_ft.
    """
    if not 0.0 <= lambda_interp <= 1.0:
        raise ValueError(f"lambda_interp must be in [0, 1], got {lambda_interp}")

    sd_zs = _load_state_dict(theta_zs)
    sd_ft = _load_state_dict(theta_ft)
    _check_compatible_state_dicts(sd_zs, sd_ft)

    return {
        key: (1.0 - lambda_interp) * sd_zs[key].float() + lambda_interp * sd_ft[key].float()
        for key in sd_zs
    }
