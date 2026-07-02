"""VRAM breakdown diagnostic for FA-PromptDETR (CUDA OOM investigation).

NOT part of the training pipeline -- a one-off diagnostic to measure VRAM
usage at each pipeline stage (FA-VP, backbone, fusion, encoder, decoder,
loss, backward), to confirm where a CUDA OOM crash's memory is actually
going before trying fixes. Requires an actual CUDA GPU -- run this on the
same machine/Colab instance where the OOM happened, not on a machine with a
small/no GPU.

Usage:
    python scripts/diagnose_vram.py --config configs/base.yaml --batch-size 1
    python scripts/diagnose_vram.py --config configs/base.yaml --batch-size 8
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `data`/`models` imports

from data.dataset import LLVIPDataset, collate_fn  # noqa: E402
from models.fa_promptdetr import FAPromptDETR, load_config  # noqa: E402


def report(tag: str):
    allocated = torch.cuda.memory_allocated() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{tag}] allocated={allocated:.3f} GiB  max_allocated={max_allocated:.3f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--subset-size", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "This diagnostic requires a CUDA GPU -- run it on the machine/Colab "
            "instance where the OOM happened, not a CPU-only or low-VRAM machine."
        )

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    config = load_config(args.config)
    report("before model build")

    model = FAPromptDETR(config).to(device)
    model.train()
    report("after model build (weights on GPU)")

    data_cfg = config["data"]
    dataset = LLVIPDataset(
        root=data_cfg["root"],
        ann_file=data_cfg["train_ann"],
        split="train",
        input_size=tuple(data_cfg["input_size"]),
        subset_size=args.subset_size or args.batch_size,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    rgb_imgs, ir_imgs, targets = next(iter(dataloader))
    rgb_imgs, ir_imgs = rgb_imgs.to(device), ir_imgs.to(device)
    targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
    report(f"after loading 1 batch (batch_size={args.batch_size}) onto GPU")

    height, width = rgb_imgs.shape[-2:]
    target_shapes = model._freq_prompt_target_shapes(height, width)

    pfreq_rgb, _, _ = model.favp_rgb(rgb_imgs, target_shapes)
    pfreq_ir, _, _ = model.favp_ir(ir_imgs, target_shapes)
    pfreq_total = [p_rgb + p_ir for p_rgb, p_ir in zip(pfreq_rgb, pfreq_ir)]
    report("after FA-VP (both streams)")

    feats_rgb, feats_ir = model.backbone(rgb_imgs, ir_imgs)
    report("after backbone (dual ResNet50)")

    feats_fused, _ = model.fusion(rgb_imgs, feats_rgb, feats_ir)
    report("after fusion")

    feats_encoded = model.encoder(feats_fused, freq_prompts=pfreq_total)
    report("after encoder (HybridEncoder, AIFI+CCFM)")

    outputs = model.decoder(feats_encoded[:3])
    report("after decoder (RTDETRTransformer, incl. P2-scale deformable attention)")

    decoder_out = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}
    encoder_out = {"pred_logits": outputs["enc_pred_logits"], "pred_boxes": outputs["enc_pred_boxes"]}
    decoder_losses = model.criterion(decoder_out, targets)
    encoder_losses = model.criterion(encoder_out, targets)
    loss_total = decoder_losses["loss_total"] + encoder_losses["loss_total"]
    report("after loss computation")

    loss_total.backward()
    report("after backward")

    print(f"\nPeak VRAM for this run: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GiB")


if __name__ == "__main__":
    main()
