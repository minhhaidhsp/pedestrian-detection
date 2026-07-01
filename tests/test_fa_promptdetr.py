import copy

import torch

from models.fa_promptdetr import FAPromptDETR, load_config

INPUT_SIZE = 640


def _make_test_config():
    """Base config with a smaller/offline-friendly decoder+encoder for fast,
    network-independent unit tests -- architecture connectivity (what these
    tests check) does not depend on layer count or pretrained weights.
    """
    config = load_config("configs/base.yaml")
    config = copy.deepcopy(config)
    config["model"]["pretrained_backbone"] = False
    config["model"]["decoder"]["num_decoder_layers"] = 2
    config["model"]["decoder"]["dim_feedforward"] = 256
    config["model"]["decoder"]["nhead"] = 4
    config["model"]["encoder"]["dim_feedforward"] = 256
    config["model"]["encoder"]["nhead"] = 4
    return config


def _make_targets(batch_sizes):
    targets = []
    for n in batch_sizes:
        boxes = torch.rand(n, 4) * 0.3 + 0.3  # keep well inside [0,1], sane w/h
        boxes[:, 2:] = boxes[:, 2:].clamp(min=0.05, max=0.3)
        labels = torch.zeros(n, dtype=torch.long)  # single class "person"
        targets.append({"labels": labels, "boxes": boxes})
    return targets


def test_forward_end_to_end_with_targets_batch_one():
    torch.manual_seed(0)
    config = _make_test_config()
    model = FAPromptDETR(config)
    model.train()

    rgb_img = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    ir_img = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    targets = _make_targets([3])

    outputs, loss_dict = model(rgb_img, ir_img, targets)

    for key in ("pred_logits", "pred_boxes", "enc_pred_logits", "enc_pred_boxes"):
        assert torch.isfinite(outputs[key]).all(), f"{key} has NaN/Inf"

    for key, value in loss_dict.items():
        assert torch.isfinite(value).all(), f"loss_dict[{key}] has NaN/Inf"


def test_forward_end_to_end_with_targets_batch_two():
    torch.manual_seed(1)
    config = _make_test_config()
    model = FAPromptDETR(config)
    model.train()

    rgb_img = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    ir_img = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    targets = _make_targets([3, 2])

    outputs, loss_dict = model(rgb_img, ir_img, targets)

    assert outputs["pred_logits"].shape[0] == 2
    assert outputs["pred_boxes"].shape[0] == 2
    for key in ("pred_logits", "pred_boxes", "enc_pred_logits", "enc_pred_boxes"):
        assert torch.isfinite(outputs[key]).all(), f"{key} has NaN/Inf"


def test_loss_total_is_finite_positive():
    torch.manual_seed(2)
    config = _make_test_config()
    model = FAPromptDETR(config)
    model.train()

    rgb_img = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    ir_img = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    targets = _make_targets([3])

    _, loss_dict = model(rgb_img, ir_img, targets)

    assert torch.isfinite(loss_dict["loss_total"])
    assert loss_dict["loss_total"].item() > 0


def test_backward_flows_gradients_to_all_parameters():
    torch.manual_seed(3)
    config = _make_test_config()
    model = FAPromptDETR(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"])

    rgb_img = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    ir_img = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    targets = _make_targets([3, 2])

    # NOTE: RTDETRTransformer deliberately zero-initializes each bbox MLP
    # head's LAST layer (standard DETR/Deformable-DETR box-refinement trick:
    # start refinement as identity). While that layer's weight is exactly
    # zero, the backward pass cannot carry gradient past it into any earlier
    # layer of the same MLP (zero weight -> zero Jacobian), so on the VERY
    # FIRST forward/backward, dec_bbox_head[i].layers[0:-1] and
    # enc_bbox_head.layers[0:-1] show exactly zero gradient. Verified (Buoc 6
    # manual check) that this resolves after exactly one optimizer step (the
    # last layer's weight moves off zero, unblocking the backward path from
    # then on). One warmup step here distinguishes that expected transient
    # from a permanently dead branch.
    optimizer.zero_grad()
    _, warmup_loss_dict = model(rgb_img, ir_img, targets)
    warmup_loss_dict["loss_total"].backward()
    optimizer.step()

    optimizer.zero_grad()
    _, loss_dict = model(rgb_img, ir_img, targets)
    loss_dict["loss_total"].backward()

    # KNOWN, ACCEPTED EXCEPTION (Buoc 6 decision, documented in
    # models/fa_promptdetr.py module docstring): HybridEncoder's LAST
    # bottom-up PAN stage (downsample_convs[2]/pan_blocks[2]) only ever
    # produces a P5-level output, which is discarded (we only pass
    # feats_encoded[:3] to the decoder, dropping P5 per Quyet dinh A). This
    # is accepted wasted compute/params (~1.3% of the model), not a
    # gradient-severing bug -- top-down FPN (which DOES matter for P2/P3/P4)
    # is unaffected.
    expected_dead_prefixes = ("encoder.downsample_convs.2", "encoder.pan_blocks.2")

    missing_grad = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing_grad.append(name)
        elif param.grad.abs().sum().item() == 0:
            missing_grad.append(name + " (all-zero grad)")

    unexpected_missing = [n for n in missing_grad if not n.startswith(expected_dead_prefixes)]
    assert not unexpected_missing, f"parameters with no/zero gradient (dead branch?): {unexpected_missing}"


def test_optimizer_step_changes_parameters():
    torch.manual_seed(4)
    config = _make_test_config()
    model = FAPromptDETR(config)
    model.train()

    lr = config["train"]["lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["train"]["weight_decay"])

    rgb_img = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    ir_img = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    targets = _make_targets([3])

    # NOTE: same cold-start effect as test_backward_flows_gradients_to_all_parameters
    # -- with the bbox MLP heads' last layer zero-initialized, the FIRST
    # step's gradient into earlier MLP layers is exactly zero, so their
    # only possible change on step 1 is AdamW's decoupled weight decay
    # (lr * weight_decay * p, here ~1e-8 * p) -- too small for
    # torch.allclose's default tolerance to register as "changed". One
    # warmup step unblocks real gradient flow into those layers (verified
    # in Buoc 6); measure the *second* step's before/after instead.
    optimizer.zero_grad()
    _, warmup_loss_dict = model(rgb_img, ir_img, targets)
    warmup_loss_dict["loss_total"].backward()
    optimizer.step()

    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    optimizer.zero_grad()
    _, loss_dict = model(rgb_img, ir_img, targets)
    loss_dict["loss_total"].backward()
    optimizer.step()

    expected_dead_prefixes = ("encoder.downsample_convs.2", "encoder.pan_blocks.2")
    unchanged = []
    for name, param in model.named_parameters():
        if name.startswith(expected_dead_prefixes):
            continue  # known dead branch, not expected to change (no gradient)
        if torch.allclose(before[name], param.detach()):
            unchanged.append(name)

    assert not unchanged, f"parameters unchanged after optimizer.step(): {unchanged}"


def test_inference_forward_without_targets():
    torch.manual_seed(5)
    config = _make_test_config()
    model = FAPromptDETR(config)
    model.eval()

    rgb_img = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    ir_img = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        outputs = model(rgb_img, ir_img, targets=None)

    assert set(outputs.keys()) == {"pred_logits", "pred_boxes", "enc_pred_logits", "enc_pred_boxes"}

    num_queries = config["model"]["decoder"]["num_queries"]
    num_classes = config["data"]["num_classes"]
    assert outputs["pred_logits"].shape == (1, num_queries, num_classes)
    assert outputs["pred_boxes"].shape == (1, num_queries, 4)
    assert torch.isfinite(outputs["pred_logits"]).all()
    assert torch.isfinite(outputs["pred_boxes"]).all()
    assert torch.all(outputs["pred_boxes"] >= 0.0) and torch.all(outputs["pred_boxes"] <= 1.0)
