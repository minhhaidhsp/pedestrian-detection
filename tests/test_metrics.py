import copy

import torch

from data.dataset import LLVIPDataset, collate_fn
from eval.metrics import evaluate
from models.fa_promptdetr import FAPromptDETR, load_config

LLVIP_ROOT = "H:/My Drive/Dataset/LLVIP"
TRAIN_ANN = "H:/My Drive/Dataset/LLVIP/annotations_coco/train.json"


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
