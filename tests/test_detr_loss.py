import torch

from losses.detr_loss import HungarianMatcher, SetCriterion, generalized_box_iou


def _make_targets(num_targets: int, num_classes: int = 1):
    boxes = torch.rand(num_targets, 4) * 0.5 + 0.25  # keep well within [0,1], w/h>0
    boxes[:, 2:] = boxes[:, 2:].clamp(min=0.05, max=0.3)  # sane w,h
    labels = torch.randint(0, num_classes, (num_targets,))
    return {"labels": labels, "boxes": boxes}


def test_matcher_matches_min_pred_target_per_image():
    torch.manual_seed(0)
    num_queries = 7
    batch_targets = [_make_targets(3), _make_targets(10)]  # fewer & more targets than queries

    outputs = {
        "pred_logits": torch.randn(2, num_queries, 1),
        "pred_boxes": torch.rand(2, num_queries, 4),
    }
    matcher = HungarianMatcher()
    indices = matcher(outputs, batch_targets)

    assert len(indices) == 2
    for (src, tgt), targets in zip(indices, batch_targets):
        expected = min(num_queries, len(targets["boxes"]))
        assert len(src) == expected
        assert len(tgt) == expected


def test_generalized_box_iou_identical_boxes_is_one():
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    giou = generalized_box_iou(box, box)
    assert torch.allclose(giou, torch.ones(1, 1), atol=1e-6)


def _reference_giou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float) -> torch.Tensor:
    """Same formula as generalized_box_iou, parametrized by eps directly, to
    compare against a negligibly small epsilon (see test below)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=eps)
    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]
    return iou - (area_c - union) / area_c.clamp(min=eps)


def test_epsilon_bump_does_not_distort_normal_box_giou():
    # Giai doan F: eps was bumped 1e-7 -> 1e-4 for fp16/AMP safety (see
    # generalized_box_iou's DESIGN DECISION comment). For normal, non-degenerate
    # boxes, union/area_c are never anywhere near either epsilon value, so the
    # clamp should be inert here -- confirm generalized_box_iou (using
    # whatever eps it currently has) agrees with a reference computed at a
    # negligible eps (1e-12), i.e. the bump doesn't measurably change results
    # for ordinary boxes.
    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 15.0]])
    boxes2 = torch.tensor([[2.0, 2.0, 12.0, 8.0], [6.0, 4.0, 18.0, 16.0]])

    giou = generalized_box_iou(boxes1, boxes2)
    reference = _reference_giou(boxes1, boxes2, eps=1e-12)

    assert torch.allclose(giou, reference, atol=1e-6)


def test_set_criterion_runs_and_is_finite_positive():
    torch.manual_seed(0)
    num_queries = 5
    targets = [_make_targets(3)]
    outputs = {
        "pred_logits": torch.randn(1, num_queries, 1),
        "pred_boxes": torch.rand(1, num_queries, 4),
    }

    matcher = HungarianMatcher()
    criterion = SetCriterion(matcher, weight_dict={"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0})
    losses = criterion(outputs, targets)

    assert torch.isfinite(losses["loss_total"])
    assert losses["loss_total"].item() > 0


def test_loss_decreases_for_near_correct_predictions():
    torch.manual_seed(0)
    num_queries = 5
    num_targets = 3
    targets = [_make_targets(num_targets)]

    matcher = HungarianMatcher()
    criterion = SetCriterion(matcher, weight_dict={"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0})

    # Random / uninformed prediction.
    random_outputs = {
        "pred_logits": torch.randn(1, num_queries, 1),
        "pred_boxes": torch.rand(1, num_queries, 4),
    }
    random_loss = criterion(random_outputs, targets)["loss_total"]

    # "Good" prediction: first num_targets queries closely match ground truth
    # boxes/labels with confident logits; remaining queries confidently predict
    # "no object" (large negative logit).
    good_boxes = torch.rand(1, num_queries, 4)
    good_boxes[0, :num_targets] = targets[0]["boxes"] + 0.01 * torch.randn(num_targets, 4)
    good_logits = torch.full((1, num_queries, 1), -5.0)
    good_logits[0, :num_targets, 0] = 5.0
    good_outputs = {"pred_logits": good_logits, "pred_boxes": good_boxes}
    good_loss = criterion(good_outputs, targets)["loss_total"]

    assert good_loss.item() < random_loss.item()
