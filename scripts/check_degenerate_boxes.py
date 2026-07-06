"""Scans a COCO annotation file for degenerate boxes (width<=0 or height<=0),
both in raw pixel coordinates and after normalizing to the model's input size
(matching data/dataset.py's resize+normalize logic) -- a degenerate box can
silently poison generalized_box_iou/matching with NaN/Inf during training,
independent of any AMP/gradient-clipping issue.

Usage:
    python scripts/check_degenerate_boxes.py --ann H:/My Drive/Dataset/LLVIP/annotations_coco/train.json
    python scripts/check_degenerate_boxes.py --config configs/base.yaml --split train
"""

import argparse

from pycocotools.coco import COCO


def check(ann_file: str) -> None:
    coco = COCO(ann_file)
    degenerate = []

    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns([ann_id])[0]
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            img_info = coco.loadImgs([ann["image_id"]])[0]
            degenerate.append(
                {
                    "ann_id": ann_id,
                    "image_id": ann["image_id"],
                    "file_name": img_info["file_name"],
                    "bbox_xywh": [x, y, w, h],
                }
            )

    print(f"Scanned {len(coco.getAnnIds())} annotations across {len(coco.getImgIds())} images.")
    print(f"Degenerate boxes (w<=0 or h<=0) found: {len(degenerate)}")
    for d in degenerate[:50]:
        print(f"  image_id={d['image_id']} file={d['file_name']} bbox_xywh={d['bbox_xywh']}")
    if len(degenerate) > 50:
        print(f"  ... and {len(degenerate) - 50} more (truncated)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", type=str, default=None, help="Path to a COCO JSON directly")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    if args.ann:
        ann_file = args.ann
    else:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from models.fa_promptdetr import load_config

        config = load_config(args.config)
        ann_file = config["data"]["train_ann"] if args.split == "train" else config["data"]["val_ann"]

    print(f"Checking: {ann_file}")
    check(ann_file)


if __name__ == "__main__":
    main()
