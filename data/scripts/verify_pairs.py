"""Verify LLVIP dataset integrity: visible/infrared/annotation triplets.

Checks, for each split (train/test), that every visible image has a matching
infrared image and a matching Pascal VOC annotation XML with the same stem.
Also compares totals against the numbers reported in the paper (train=12025,
test=3463 pairs).

Usage:
    python data/scripts/verify_pairs.py --root "H:/My Drive/Dataset/LLVIP"
"""

import argparse
from pathlib import Path

PAPER_COUNTS = {"train": 12025, "test": 3463}
MAX_LISTED_ERRORS = 20


def stems(dir_path: Path) -> dict[str, Path]:
    """Map file stem -> path for all files directly inside dir_path."""
    return {p.stem: p for p in dir_path.iterdir() if p.is_file()}


def verify_split(root: Path, split: str) -> dict:
    visible_dir = root / "visible" / split
    infrared_dir = root / "infrared" / split
    ann_dir = root / "Annotations"

    for d in (visible_dir, infrared_dir, ann_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {d}")

    visible = stems(visible_dir)
    infrared = stems(infrared_dir)
    annotations = stems(ann_dir)

    missing_infrared = []
    missing_annotation = []
    valid_pairs = 0

    for stem, vpath in visible.items():
        has_infrared = stem in infrared
        has_annotation = stem in annotations
        if not has_infrared:
            missing_infrared.append(vpath.name)
        if not has_annotation:
            missing_annotation.append(vpath.name)
        if has_infrared and has_annotation:
            valid_pairs += 1

    return {
        "split": split,
        "num_visible": len(visible),
        "num_infrared": len(infrared),
        "valid_pairs": valid_pairs,
        "missing_infrared": missing_infrared,
        "missing_annotation": missing_annotation,
    }


def print_report(result: dict) -> None:
    split = result["split"]
    expected = PAPER_COUNTS.get(split)
    print(f"\n=== Split: {split} ===")
    print(f"  Visible images found:     {result['num_visible']}")
    print(f"  Infrared images found:    {result['num_infrared']}")
    print(f"  Valid pairs (vis+ir+ann): {result['valid_pairs']}")
    print(f"  Missing infrared:         {len(result['missing_infrared'])}")
    print(f"  Missing annotation:       {len(result['missing_annotation'])}")

    if result["missing_infrared"]:
        print(f"  First {MAX_LISTED_ERRORS} missing-infrared files:")
        for name in result["missing_infrared"][:MAX_LISTED_ERRORS]:
            print(f"    - {name}")

    if result["missing_annotation"]:
        print(f"  First {MAX_LISTED_ERRORS} missing-annotation files:")
        for name in result["missing_annotation"][:MAX_LISTED_ERRORS]:
            print(f"    - {name}")

    if expected is not None:
        diff = result["valid_pairs"] - expected
        if diff == 0:
            print(f"  Paper comparison: MATCH (expected {expected})")
        else:
            sign = "+" if diff > 0 else ""
            print(
                f"  Paper comparison: MISMATCH -- expected {expected}, "
                f"got {result['valid_pairs']} ({sign}{diff})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to LLVIP dataset root (containing visible/, infrared/, Annotations/)",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    print(f"Verifying LLVIP dataset at: {root}")

    results = [verify_split(root, split) for split in ("train", "test")]
    for result in results:
        print_report(result)

    print("\n=== Summary ===")
    for result in results:
        expected = PAPER_COUNTS.get(result["split"])
        status = "OK" if result["valid_pairs"] == expected else "MISMATCH"
        print(
            f"  {result['split']}: {result['valid_pairs']} valid pairs "
            f"(paper: {expected}) -> {status}"
        )


if __name__ == "__main__":
    main()
