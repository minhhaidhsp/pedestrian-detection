"""COCO-style detection metrics for FA-PromptDETR: AP/AP50/AP75 (via
pycocotools COCOeval) plus precision/recall/F1 at a fixed score threshold
and IoU=0.5 (via simple greedy matching -- COCOeval itself only exposes an
interpolated precision-recall curve, not a single-operating-point P/R/F1,
so that part is computed separately here).
"""

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from losses.detr_loss import box_cxcywh_to_xyxy

CATEGORY_ID = 1  # single class "person"


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """boxes1: [N,4], boxes2: [M,4], both (x1,y1,x2,y2). Returns IoU [N,M]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


def _greedy_match_precision_recall_f1(
    detections_by_image: dict, coco_gt: COCO, iou_threshold: float = 0.5
) -> dict:
    """detections_by_image: {image_id: [(score, [x1,y1,x2,y2]), ...]}.

    Iterates only over `detections_by_image`'s keys -- i.e. exactly the
    images the caller actually ran inference on -- NOT coco_gt.getImgIds().
    ann_file may cover the full dataset (e.g. train.json, 12,025 images)
    while the dataloader only evaluated a small subset; scoring against every
    ground-truth image regardless of whether it was evaluated would count
    every box in every un-evaluated image as a false negative and collapse
    recall/AP toward zero.
    """
    tp, fp, fn = 0, 0, 0

    for image_id in detections_by_image.keys():
        dets = sorted(detections_by_image.get(image_id, []), key=lambda d: -d[0])
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        gt_boxes_xywh = [coco_gt.loadAnns([aid])[0]["bbox"] for aid in ann_ids]
        gt_boxes = torch.tensor(
            [[x, y, x + w, y + h] for x, y, w, h in gt_boxes_xywh], dtype=torch.float32
        ) if gt_boxes_xywh else torch.zeros((0, 4))

        matched_gt = torch.zeros(len(gt_boxes_xywh), dtype=torch.bool)

        if not dets:
            fn += len(gt_boxes_xywh)
            continue

        det_boxes = torch.tensor([d[1] for d in dets], dtype=torch.float32)
        if len(gt_boxes) > 0:
            ious = _box_iou(det_boxes, gt_boxes)  # [num_dets, num_gt]
        else:
            ious = torch.zeros((len(dets), 0))

        for i in range(len(dets)):
            if ious.shape[1] == 0:
                fp += 1
                continue
            best_iou, best_j = ious[i].max(dim=0)
            if best_iou.item() >= iou_threshold and not matched_gt[best_j]:
                tp += 1
                matched_gt[best_j] = True
            else:
                fp += 1

        fn += (~matched_gt).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def evaluate(
    model, dataloader, ann_file, device, score_threshold: float = 0.05, max_detections: int = 100
) -> dict:
    """Runs inference over `dataloader` and scores predictions against `ann_file`.

    Returns a dict with AP, AP50, AP75 (pycocotools COCOeval, IoU-averaged /
    IoU=0.5 / IoU=0.75), and precision/recall/f1 (greedy IoU=0.5 matching at
    `score_threshold`).
    """
    model.eval()
    coco_gt = COCO(str(ann_file))

    coco_results = []
    detections_by_image = {}

    for rgb_imgs, ir_imgs, targets in dataloader:
        rgb_imgs = rgb_imgs.to(device)
        ir_imgs = ir_imgs.to(device)

        outputs = model(rgb_imgs, ir_imgs, targets=None)
        scores = outputs["pred_logits"].sigmoid().squeeze(-1)  # [B, Q]
        boxes_xyxy = box_cxcywh_to_xyxy(outputs["pred_boxes"])  # [B, Q, 4] normalized

        for b in range(scores.shape[0]):
            image_id = targets[b]["image_id"]
            orig_w, orig_h = targets[b]["orig_size"]

            img_scores = scores[b]
            img_boxes = boxes_xyxy[b]

            keep = (img_scores >= score_threshold).nonzero(as_tuple=True)[0]
            if keep.numel() > max_detections:
                topk = torch.topk(img_scores[keep], max_detections).indices
                keep = keep[topk]

            image_dets = []
            for idx in keep.tolist():
                score = img_scores[idx].item()
                x1, y1, x2, y2 = img_boxes[idx].tolist()
                x1, x2 = x1 * orig_w, x2 * orig_w
                y1, y2 = y1 * orig_h, y2 * orig_h
                coco_results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": CATEGORY_ID,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
                image_dets.append((score, [x1, y1, x2, y2]))
            detections_by_image[image_id] = image_dets

    pr_f1 = _greedy_match_precision_recall_f1(detections_by_image, coco_gt)

    if not coco_results:
        return {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
            "precision": pr_f1["precision"], "recall": pr_f1["recall"], "f1": pr_f1["f1"],
            "num_detections": 0,
        }

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # Restrict to the images actually evaluated -- see _greedy_match_precision_recall_f1
    # docstring for why (ann_file may cover far more images than the dataloader did).
    coco_eval.params.imgIds = list(detections_by_image.keys())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "precision": pr_f1["precision"],
        "recall": pr_f1["recall"],
        "f1": pr_f1["f1"],
        "num_detections": len(coco_results),
    }
