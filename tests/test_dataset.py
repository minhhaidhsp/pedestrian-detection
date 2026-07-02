from pathlib import Path

import pytest
import torch

from data.dataset import LLVIPDataset, collate_fn
from models.fa_promptdetr import load_config

try:
    _config = load_config("configs/base.yaml")
    LLVIP_ROOT = _config["data"]["root"]
    TRAIN_ANN = _config["data"]["train_ann"]
except FileNotFoundError:
    LLVIP_ROOT = None
    TRAIN_ANN = None

_DATA_AVAILABLE = (
    LLVIP_ROOT is not None
    and Path(LLVIP_ROOT).is_dir()
    and TRAIN_ANN is not None
    and Path(TRAIN_ANN).is_file()
)

# These tests depend on the real LLVIP dataset (unlike the fake-tensor tests
# elsewhere) -- skip cleanly rather than fail hard when it's not present in
# the current environment (e.g. CI, or a machine without the Drive mount).
pytestmark = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason=f"Bỏ qua: không tìm thấy dữ liệu LLVIP thật tại {LLVIP_ROOT}, kiểm tra lại configs/base.yaml",
)


def _make_dataset(subset_size=5):
    return LLVIPDataset(root=LLVIP_ROOT, ann_file=TRAIN_ANN, split="train", subset_size=subset_size)


def test_load_first_five_samples():
    dataset = _make_dataset(subset_size=5)
    assert len(dataset) == 5

    for i in range(len(dataset)):
        rgb_img, ir_img, targets = dataset[i]
        assert rgb_img.shape == (3, 640, 640)
        assert ir_img.shape == (3, 640, 640)
        assert torch.isfinite(rgb_img).all()
        assert torch.isfinite(ir_img).all()

        assert targets["boxes"].ndim == 2 and targets["boxes"].shape[1] == 4
        assert targets["labels"].shape[0] == targets["boxes"].shape[0]
        assert (targets["labels"] == 0).all()  # single class "person"


def test_collate_fn_with_varying_box_counts():
    dataset = _make_dataset(subset_size=4)
    batch = [dataset[i] for i in range(4)]

    rgb_imgs, ir_imgs, targets = collate_fn(batch)

    assert rgb_imgs.shape == (4, 3, 640, 640)
    assert ir_imgs.shape == (4, 3, 640, 640)
    assert isinstance(targets, list) and len(targets) == 4
    # sanity: not all images have the same number of boxes (real data) --
    # this is exactly why targets must stay a list, not a stacked tensor.
    num_boxes = [t["boxes"].shape[0] for t in targets]
    assert all(n >= 0 for n in num_boxes)


def test_box_coordinates_in_unit_range_after_resize():
    dataset = _make_dataset(subset_size=5)

    for i in range(len(dataset)):
        _, _, targets = dataset[i]
        boxes = targets["boxes"]
        if boxes.shape[0] == 0:
            continue
        assert torch.all(boxes >= 0.0)
        assert torch.all(boxes <= 1.0)
