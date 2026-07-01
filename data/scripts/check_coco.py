"""Sanity-check a COCO-format annotation JSON by loading it with pycocotools.

Usage:
    python data/scripts/check_coco.py --ann "H:/My Drive/Dataset/LLVIP/annotations_coco/train.json"
"""

import argparse
from pathlib import Path

from pycocotools.coco import COCO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", type=Path, required=True, help="Path to COCO JSON file")
    args = parser.parse_args()
    ann_path = args.ann.expanduser().resolve()

    coco = COCO(str(ann_path))
    print(f"Loaded: {ann_path}")
    print(f"  Images:      {len(coco.getImgIds())}")
    print(f"  Annotations: {len(coco.getAnnIds())}")
    print(f"  Categories:  {coco.loadCats(coco.getCatIds())}")


if __name__ == "__main__":
    main()
