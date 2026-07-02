import copy
from pathlib import Path

import pytest
import torch

from data.dataset import LLVIPDataset, collate_fn
from eval.metrics import evaluate
from models.fa_promptdetr import FAPromptDETR, load_config

try:
    _base_config = load_config("configs/base.yaml")
    LLVIP_ROOT = _base_config["data"]["root"]
    TRAIN_ANN = _base_config["data"]["train_ann"]
except FileNotFoundError:
    LLVIP_ROOT = None
    TRAIN_ANN = None

_DATA_AVAILABLE = (
    LLVIP_ROOT is not None
    and Path(LLVIP_ROOT).is_dir()
    and TRAIN_ANN is not None
    and Path(TRAIN_ANN).is_file()
)

# Depends on the real LLVIP dataset -- skip cleanly rather than fail hard
# when it's not present in the current environment.
pytestmark = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason=f"Bỏ qua: không tìm thấy dữ liệu LLVIP thật tại {LLVIP_ROOT}, kiểm tra lại configs/base.yaml",
)


def _make_small_model():
    config = load_config("configs/base.yaml")
    config = copy.deepcopy(config)
    config["model"]["pretrained_backbone"] = False
    config["model"]["decoder"]["num_decoder_layers"] = 2
    config["model"]["decoder"]["dim_feedforward"] = 256
    config["model"]["decoder"]["nhead"] = 4
    config["model"]["encoder"]["dim_feedforward"] = 256
    config["model"]["encoder"]["nhead"] = 4
    return FAPromptDETR(config)


def test_evaluate_runs_without_crashing_on_small_subset():
    torch.manual_seed(0)
    model = _make_small_model()
    model.eval()

    dataset = LLVIPDataset(root=LLVIP_ROOT, ann_file=TRAIN_ANN, split="train", subset_size=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    metrics = evaluate(model, dataloader, ann_file=TRAIN_ANN, device=torch.device("cpu"), score_threshold=0.0)

    for key in ("AP", "AP50", "AP75", "precision", "recall", "f1", "num_detections"):
        assert key in metrics, f"missing metrics key: {key}"
        assert isinstance(metrics[key], (int, float))

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
