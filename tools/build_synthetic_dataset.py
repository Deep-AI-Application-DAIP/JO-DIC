from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from jodic.datasets.synthetic_speckle import SyntheticSpeckleDataset


def save_img(tensor, path: Path):
    arr = tensor.squeeze().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="synthetic_speckle_demo")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--max-disp", type=int, default=48)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = SyntheticSpeckleDataset(length=args.n, height=args.height, width=args.width, max_disp=args.max_disp)
    for i in tqdm(range(args.n)):
        item = ds[i]
        sample_dir = out / f"sample_{i:04d}"
        sample_dir.mkdir(exist_ok=True)
        for key in ["left0", "right0", "left1", "right1"]:
            save_img(item[key], sample_dir / f"{key}.png")
        np.save(sample_dir / "disp0.npy", item["disp0"].numpy())
        np.save(sample_dir / "disp1.npy", item["disp1"].numpy())
        np.save(sample_dir / "flow.npy", item["flow"].numpy())

    print(f"saved {args.n} samples to {out}")


if __name__ == "__main__":
    main()
