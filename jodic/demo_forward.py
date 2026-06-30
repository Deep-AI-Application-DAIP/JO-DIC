from __future__ import annotations

import torch
from .models.jodic import JODICModel


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = JODICModel(in_channels=1, feature_channels=32, max_disp=32, masa_heads=4, flow_iters=2).to(device)
    model.eval()

    b, h, w = 1, 96, 128
    left0 = torch.rand(b, 1, h, w, device=device)
    right0 = torch.rand(b, 1, h, w, device=device)
    left1 = torch.rand(b, 1, h, w, device=device)
    right1 = torch.rand(b, 1, h, w, device=device)

    intrinsics = {"fx": 1200.0, "fy": 1200.0, "cx": w / 2.0, "cy": h / 2.0, "baseline": 80.0}

    with torch.no_grad():
        out = model(left0, right0, left1, right1, intrinsics=intrinsics)

    print("JO-DIC forward test")
    for k in ["disp0", "disp1", "flow", "disp3d"]:
        print(f"{k:>10s}: {tuple(out[k].shape)}")
    print("strain:")
    for k, v in out["strain"].items():
        print(f"  {k:>7s}: {tuple(v.shape)}")
    print("Done.")


if __name__ == "__main__":
    main()
