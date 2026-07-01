"""LLVIP dataset loader: paired RGB/IR images + COCO-format person boxes.

Reads images from `root/visible/{split}/` and `root/infrared/{split}/`, and
annotations from a COCO JSON (produced by data/scripts/convert_to_coco.py in
Giai doan B). Verified (Giai doan E) against the real LLVIP files: visible and
infrared images share identical (width, height) per pair -- already spatially
registered, no warp/align needed, only a synchronized resize. Also verified
the infrared JPGs are already stored as 3-channel RGB with R==G==B (i.e.
already grayscale-replicated at the source) -- `.convert("RGB")` below is
still applied explicitly for robustness/defensiveness (Quyet dinh C, Buoc 2),
not because the real files currently need it.
"""

from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LLVIPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        ann_file: str | Path,
        split: str,
        input_size: tuple[int, int] = (640, 640),
        subset_size: int | None = None,
    ):
        """root: LLVIP dataset root (contains visible/, infrared/).
        ann_file: path to COCO JSON (train.json or test.json).
        split: "train" or "test" -- selects visible/{split}/, infrared/{split}/.
        subset_size: if given, only use the first `subset_size` images (by
        COCO image id order) -- for fast local debugging, not a dataset-level
        default.
        """
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.visible_dir = self.root / "visible" / split
        self.infrared_dir = self.root / "infrared" / split

        self.coco = COCO(str(ann_file))
        img_ids = sorted(self.coco.getImgIds())
        if subset_size is not None:
            img_ids = img_ids[:subset_size]
        self.img_ids = img_ids

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        orig_w, orig_h = img_info["width"], img_info["height"]
        target_h, target_w = self.input_size

        rgb_img = self._load_image(self.visible_dir / file_name)
        ir_img = self._load_image(self.infrared_dir / file_name)

        rgb_img = rgb_img.resize((target_w, target_h), Image.BILINEAR)
        ir_img = ir_img.resize((target_w, target_h), Image.BILINEAR)

        rgb_tensor = TF.normalize(TF.to_tensor(rgb_img), IMAGENET_MEAN, IMAGENET_STD)
        ir_tensor = TF.normalize(TF.to_tensor(ir_img), IMAGENET_MEAN, IMAGENET_STD)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1 = x * scale_x, y * scale_y
            w_s, h_s = w * scale_x, h * scale_y
            cx = (x1 + w_s / 2) / target_w
            cy = (y1 + h_s / 2) / target_h
            boxes.append([cx, cy, w_s / target_w, h_s / target_h])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros(len(boxes), dtype=torch.long)  # single class "person"

        targets = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": img_id,
            "orig_size": (orig_w, orig_h),  # for converting predictions back for COCOeval
        }
        return rgb_tensor, ir_tensor, targets


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, dict]]):
    """Stacks images (fixed resize -> same shape); targets stay a list of
    dicts, since each image has a different number of boxes and cannot be
    stacked into a single tensor.
    """
    rgb_imgs, ir_imgs, targets = zip(*batch)
    return torch.stack(rgb_imgs, dim=0), torch.stack(ir_imgs, dim=0), list(targets)
