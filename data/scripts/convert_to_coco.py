"""Convert LLVIP Pascal VOC annotations to COCO-format JSON.

Reads Annotations/*.xml (VOC format, single class "person") and the visible
image list for each split, and emits one COCO JSON per split with a single
category (person, category_id=1). Image dimensions come from each XML's
<size> tag, matching the paired visible/infrared images (already registered
to the same resolution).

Usage:
    python data/scripts/convert_to_coco.py \
        --root "H:/My Drive/Dataset/LLVIP" \
        --out-dir "H:/My Drive/Dataset/LLVIP/annotations_coco"
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

CATEGORIES = [{"id": 1, "name": "person", "supercategory": "none"}]


def convert_split(root: Path, split: str) -> dict:
    visible_dir = root / "visible" / split
    ann_dir = root / "Annotations"

    images = []
    annotations = []
    ann_id = 1

    image_files = sorted(visible_dir.iterdir(), key=lambda p: p.name)
    for image_id, img_path in enumerate(image_files, start=1):
        xml_path = ann_dir / f"{img_path.stem}.xml"
        tree = ET.parse(xml_path)
        root_el = tree.getroot()

        size_el = root_el.find("size")
        width = int(size_el.findtext("width"))
        height = int(size_el.findtext("height"))

        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        for obj in root_el.findall("object"):
            name = obj.findtext("name")
            if name != "person":
                continue
            bnd = obj.find("bndbox")
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
            w = xmax - xmin
            h = ymax - ymin

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to LLVIP dataset root (containing visible/, infrared/, Annotations/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write train.json / test.json into (created if missing)",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, out_name in (("train", "train.json"), ("test", "test.json")):
        print(f"Converting split '{split}'...")
        coco_dict = convert_split(root, split)
        out_path = out_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f)
        print(
            f"  Wrote {out_path} "
            f"({len(coco_dict['images'])} images, "
            f"{len(coco_dict['annotations'])} annotations)"
        )


if __name__ == "__main__":
    main()
