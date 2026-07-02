"""Training entrypoint for FA-PromptDETR (Giai doan E).

Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/base.yaml --subset-size 32 --max-epochs 3 --device cpu --batch-size 2
    python train.py --config configs/base.yaml --resume runs/fa_promptdetr_base/checkpoint_epoch3.pth ...
"""

import argparse
import contextlib
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import LLVIPDataset, collate_fn
from models.fa_promptdetr import FAPromptDETR, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint_epochN.pth to resume from")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override config train.epochs")
    parser.add_argument("--subset-size", type=int, default=None, help="Limit training set to the first N images")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config train.batch_size")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Force device (default: auto-detect)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for clear local tracebacks)")
    parser.add_argument("--log-interval", type=int, default=10, help="Log training loss every N steps")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps before optimizer.step() (effective batch = batch_size * N)",
    )
    return parser.parse_args()


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger("fa_promptdetr.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to console and {log_path}")
    return logger


def build_dataloader(config: dict, split: str, batch_size: int, subset_size: int | None, num_workers: int):
    data_cfg = config["data"]
    ann_file = data_cfg["train_ann"] if split == "train" else data_cfg["val_ann"]
    dataset = LLVIPDataset(
        root=data_cfg["root"],
        ann_file=ann_file,
        split=split,
        input_size=tuple(data_cfg["input_size"]),
        subset_size=subset_size,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
        },
        path,
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    train_cfg = config["train"]
    max_epochs = args.max_epochs if args.max_epochs is not None else train_cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else train_cfg["batch_size"]

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(config["experiment"]["output_dir"])
    logger = setup_logger(Path("logs"))

    # AMP is CUDA-only: torch.cuda.amp.GradScaler has no CPU equivalent, and
    # CPU training was already verified fp32-only (Giai doan E). enabled=False
    # makes autocast/GradScaler complete no-ops, so the rest of the loop below
    # doesn't need a separate CPU/GPU code path.
    accum_steps = max(args.grad_accum_steps, 1)
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    effective_batch = batch_size * accum_steps
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"max_epochs={max_epochs} batch_size={batch_size} subset_size={args.subset_size}")
    logger.info(f"amp={use_amp} grad_accum_steps={accum_steps} effective_batch_size={effective_batch}")

    model = FAPromptDETR(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    start_epoch = 1
    if args.resume is not None:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Allow --max-epochs to extend the cosine schedule beyond the run
        # that produced this checkpoint (state_dict restores the old T_max).
        scheduler.T_max = max_epochs
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        else:
            logger.info("Checkpoint has no scaler_state_dict (pre-AMP checkpoint) -- starting scaler fresh")
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed at epoch {checkpoint['epoch']}, continuing from epoch {start_epoch}")

    dataloader = build_dataloader(
        config, split="train", batch_size=batch_size, subset_size=args.subset_size, num_workers=args.num_workers
    )
    logger.info(f"Training set size: {len(dataloader.dataset)} images, {len(dataloader)} batches/epoch")

    grad_clip_norm = train_cfg.get("grad_clip_norm")

    model.train()
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        num_batches = 0

        optimizer.zero_grad()
        for step, (rgb_imgs, ir_imgs, targets) in enumerate(dataloader, start=1):
            rgb_imgs = rgb_imgs.to(device)
            ir_imgs = ir_imgs.to(device)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

            with autocast_ctx:
                _, loss_dict = model(rgb_imgs, ir_imgs, targets)
                loss_total = loss_dict["loss_total"]
                loss = loss_total / accum_steps

            scaler.scale(loss).backward()

            is_accum_boundary = (step % accum_steps == 0) or (step == len(dataloader))
            if is_accum_boundary:
                if grad_clip_norm:
                    # Unscale before clipping -- gradients are still scaled by
                    # the GradScaler's factor at this point (no-op if AMP is off).
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss_sum += loss_total.item()
            num_batches += 1

            if step % args.log_interval == 0 or step == len(dataloader):
                logger.info(f"epoch {epoch} step {step}/{len(dataloader)} loss_total={loss_total.item():.4f}")

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss_sum / max(num_batches, 1)
        logger.info(f"epoch {epoch} done in {epoch_time:.1f}s, avg_loss_total={avg_loss:.4f}")

        checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pth"
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, config)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
