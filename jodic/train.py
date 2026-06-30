from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets.synthetic_speckle import SyntheticSpeckleDataset
from .models.jodic import JODICModel
from .models.losses import JODICLoss


def parse_args():
    p = argparse.ArgumentParser(description="Train JO-DIC on synthetic speckle data")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--max-disp", type=int, default=48)
    p.add_argument("--feature-channels", type=int, default=32)
    p.add_argument("--masa-heads", type=int, default=4)
    p.add_argument("--flow-iters", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save", type=str, default="checkpoints/jodic_synthetic.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SyntheticSpeckleDataset(length=128, height=args.height, width=args.width, max_disp=args.max_disp)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = JODICModel(
        in_channels=1,
        feature_channels=args.feature_channels,
        max_disp=args.max_disp,
        masa_heads=args.masa_heads,
        flow_iters=args.flow_iters,
    ).to(device)

    criterion = JODICLoss(lambda_disp=1.0, lambda_flow=1.0, lambda_consistency=0.2)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            left0 = batch["left0"].to(device)
            right0 = batch["right0"].to(device)
            left1 = batch["left1"].to(device)
            right1 = batch["right1"].to(device)

            targets = {k: v.to(device) for k, v in batch.items() if k not in ["left0", "right0", "left1", "right1"]}

            out = model(left0, right0, left1, right1)
            loss, losses = criterion(out, targets)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            pbar.set_postfix(total=f"{losses['total'].item():.4f}", geo=f"{losses['consistency'].item():.4f}")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, save_path)
    print(f"saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
