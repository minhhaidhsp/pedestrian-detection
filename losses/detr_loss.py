"""Hungarian matching + focal/L1/GIoU set loss for DETR-style detectors (Eq. 7-8).

Eq. (8): L = 2*L_focal + 5*L_L1 + 2*L_giou

Box convention used throughout this module's public API (matcher/criterion I/O):
(cx, cy, w, h), normalized to [0, 1] -- the standard DETR convention.
`generalized_box_iou` itself operates on (x1, y1, x2, y2) boxes (documented in
its own docstring); use `box_cxcywh_to_xyxy` to convert before calling it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """boxes: [..., 4] in (cx, cy, w, h) -> [..., 4] in (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU between two sets of boxes.

    Input convention: (x1, y1, x2, y2), with x2 >= x1 and y2 >= y1.
    boxes1: [N, 4], boxes2: [M, 4] -> giou: [N, M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-7)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-7)


def sigmoid_focal_loss(
    inputs: torch.Tensor, targets: torch.Tensor, num_boxes: float, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Focal loss on sigmoid logits vs multi-label one-hot targets.

    inputs/targets: [B, Q, num_classes]. Returns a scalar normalized by num_boxes.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and targets (standard DETR-style cost).

    outputs: {"pred_logits": [B, Q, num_classes], "pred_boxes": [B, Q, 4] (cxcywh)}
    targets: list of length B, each {"labels": [N_i], "boxes": [N_i, 4] (cxcywh)}
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [B*Q, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 4]

        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])

        if tgt_ids.numel() == 0:
            return [
                (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
                for _ in range(bs)
            ]

        alpha, gamma = self.focal_alpha, self.focal_gamma
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = cost.view(bs, num_queries, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """Computes DETR set loss: 2*focal(cls) + 5*L1(bbox) + 2*GIoU(bbox), Eq. (8)."""

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_dict: dict | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict or {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _get_src_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes) -> dict:
        pred_logits = outputs["pred_logits"]  # [B, Q, C]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])

        target_onehot = torch.zeros_like(pred_logits)
        if target_classes_o.numel() > 0:
            target_onehot[idx[0], idx[1], target_classes_o] = 1.0

        loss_ce = sigmoid_focal_loss(
            pred_logits, target_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.focal_gamma
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes) -> dict:
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if src_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes

        giou = torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = (1 - giou).sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs: dict, targets: list[dict]) -> dict:
        indices = self.matcher(outputs, targets)
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        loss_total = sum(losses[name] * self.weight_dict[name] for name in losses if name in self.weight_dict)
        losses["loss_total"] = loss_total
        return losses
