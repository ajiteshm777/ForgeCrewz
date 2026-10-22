"""Train Mask R-CNN on PhenoBench leaves.

Overfit sanity test:
    python src/train.py --data /content/PhenoBench --out /content/drive/MyDrive/ckpt --overfit 10 --epochs 30
Full run:
    python src/train.py --data /content/PhenoBench --out /content/drive/MyDrive/ckpt --epochs 25 --bs 2
Resume is automatic if <out>/last.pth exists.
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import PhenoBenchLeaves, collate_fn
from model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--overfit", type=int, default=0, help="train on N images only")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PhenoBenchLeaves(args.data, "train", train=(args.overfit == 0), limit=args.overfit or None)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    collate_fn=collate_fn, drop_last=len(ds) > args.bs)

    model = build_model().to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    last = os.path.join(args.out, "last.pth")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    warm = min(200, len(dl))  # warmup steps, never longer than one epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for it, (imgs, tgts) in enumerate(dl):
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

            # warmup for the first epoch avoids early divergence
            if epoch == 0 and it < warm:
                for g in opt.param_groups:
                    g["lr"] = args.lr * (it + 1) / warm

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                losses = model(imgs, tgts)
                loss = sum(losses.values())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            if it % 20 == 0:
                parts = " ".join(f"{k.replace('loss_', '')}={v.item():.3f}" for k, v in losses.items())
                print(f"ep{epoch} it{it}/{len(dl)} loss={loss.item():.3f} | {parts}")

        sched.step()
        print(f"== epoch {epoch} avg loss {running / len(dl):.4f} ({time.time() - t0:.0f}s)")

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": epoch}
        torch.save(ck, last)  # always resumable
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            torch.save({"model": model.state_dict()}, os.path.join(args.out, f"epoch{epoch + 1:02d}.pth"))


if __name__ == "__main__":
    main()
