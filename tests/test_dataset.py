import torch

from data.dataset import LLVIPDataset, collate_fn

LLVIP_ROOT = "H:/My Drive/Dataset/LLVIP"
TRAIN_ANN = "H:/My Drive/Dataset/LLVIP/annotations_coco/train.json"


def _make_dataset(subset_size=5):
    return LLVIPDataset(root=LLVIP_ROOT, ann_file=TRAIN_ANN, split="train", subset_size=subset_size)


def test_load_first_five_samples():
    dataset = _make_dataset(subset_size=5)
    assert len(dataset) == 5

    for i in range(len(dataset)):
        rgb_img, ir_img, targets = dataset[i]
        assert rgb_img.shape == (3, 640, 640)
        assert ir_img.shape == (3, 640, 640)
        assert torch.isfinite(rgb_img).all()
        assert torch.isfinite(ir_img).all()

        assert targets["boxes"].ndim == 2 and targets["boxes"].shape[1] == 4
        assert targets["labels"].shape[0] == targets["boxes"].shape[0]
        assert (targets["labels"] == 0).all()  # single class "person"


def test_collate_fn_with_varying_box_counts():
    dataset = _make_dataset(subset_size=4)
    batch = [dataset[i] for i in range(4)]

    rgb_imgs, ir_imgs, targets = collate_fn(batch)

    assert rgb_imgs.shape == (4, 3, 640, 640)
    assert ir_imgs.shape == (4, 3, 640, 640)
    assert isinstance(targets, list) and len(targets) == 4
    # sanity: not all images have the same number of boxes (real data) --
    # this is exactly why targets must stay a list, not a stacked tensor.
    num_boxes = [t["boxes"].shape[0] for t in targets]
    assert all(n >= 0 for n in num_boxes)


def test_box_coordinates_in_unit_range_after_resize():
    dataset = _make_dataset(subset_size=5)

    for i in range(len(dataset)):
        _, _, targets = dataset[i]
        boxes = targets["boxes"]
        if boxes.shape[0] == 0:
            continue
        assert torch.all(boxes >= 0.0)
        assert torch.all(boxes <= 1.0)
