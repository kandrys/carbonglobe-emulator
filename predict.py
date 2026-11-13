
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from config import Config
from main import norm, AGES ,denorm_y, StepModel, HEIGHT_MEAN, AGB_MEAN


# d = np.load("data/test_ssp126.npz")
# print(d.files)
# print(d["test_x"].shape, d["test_y0"].shape)
# test_y0 = d["test_y0"]
# print(test_y0[:5, :, 0]) 




def load_stats(data: Path):
    stats = np.load(data / "data_stats" / "data_stats.npz")
    x_mean = stats["x_mean"].astype(np.float32)
    x_std  = stats["x_std"].astype(np.float32)
    y_mean = stats["y_mean"].astype(np.float32)
    y_std  = stats["y_std"].astype(np.float32)

    return x_mean, x_std, y_mean, y_std
# Public baseline (Lucas Bruzzone): test_y0 is an affine of physical units.
# Offset ≈ transformed physical zero (mode). Height scale is exact from
# anchors 0 m → −0.9009 and 35 m → 29.7706; other scales are calibrated.
Y0_OFFSET = np.array(
    [-0.900947, -0.401412, -0.532395, 0.117938, 0.259026, 0.127815, 0.084656],
    dtype=np.float64,
)
Y0_SCALE = np.array(
    [0.876341, 0.876, 0.8413, 0.9540, 0.8440, 0.8438, 0.8588],
    dtype=np.float64,
)


def decode_test_y0(test_y0: np.ndarray) -> tuple[np.ndarray, bool]:
    y0 = test_y0.astype(np.float64, copy=True)
    mins = y0[..., :2].reshape(-1, 2).min(0)
    if np.allclose(mins, Y0_OFFSET[:2], atol=0.003):
        y0 = (y0 - Y0_OFFSET) / Y0_SCALE
        return y0.astype(np.float32), True
    return test_y0.astype(np.float32, copy=True), False


def load_test_data(data: Path, scenario: str):
    with np.load(data / f"test_{scenario}.npz") as d :
        test_x = d["test_x"]
        test_y0 = d["test_y0"]
    return test_x, test_y0
@torch.no_grad()
def predict_test(model, test_x, test_y0, x_mean, x_std, y_mean, y_std,
                 device, batch_sites=64) -> np.ndarray:
    model.eval()
    n_sites = test_x.shape[0]
    out = []
    for s in range(0, n_sites, batch_sites):
        x = torch.tensor(norm(test_x[s : s + batch_sites], x_mean, x_std), dtype=torch.float32)
        y0 = torch.tensor(norm(test_y0[s : s + batch_sites], y_mean, y_std), dtype=torch.float32)
        B = x.shape[0]
        # profile = all 15 seed ages at year 0 for each site, shared by every age row
        profile = y0.reshape(B, -1).repeat_interleave(15, dim=0)
        y = y0.reshape(-1, 7)
        x = torch.repeat_interleave(x, repeats=15, dim=0)
        ages = torch.tensor(AGES, dtype=torch.float32).repeat(B) / 500
        x = x.to(device)
        y = y.to(device)
        ages = ages.to(device)
        profile = profile.to(device)
        for i in range(40):
            y = model(y, x[:, i], ages + i / 500, profile)
        y = y.cpu().numpy()
        y = denorm_y(y, y_mean, y_std).reshape(B, 15, 7)
        out.append(y)
    return np.concatenate(out)






def parse_args():
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, default=cfg.ckpt_path)
    parser.add_argument("--n_hidden", type=int, default=cfg.n_hidden,
                        help="must match the value used in training")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--batch_sites", type=int, default=64)
    parser.add_argument("--clip_y0", action="store_true",
                        help="clip negative height, agb, soil, lai, rh in test_y0 to 0")
    parser.add_argument("--no_decode_y0", action="store_true",
                        help="keep raw test_y0; by default apply the public affine decode")
    parser.add_argument("--data_dir", type=Path, default=cfg.data_dir)
    return parser.parse_args()


def main():

    args = parse_args()
    print(f"model {args.ckpt_path}  n_hidden {args.n_hidden}  decode_y0 {not args.no_decode_y0}  "
          f"clip_y0 {args.clip_y0}  out {args.out}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x_mean, x_std, y_mean, y_std = load_stats(args.data_dir)
    model = StepModel(n_in=7 + 12 * 136 + 1 + 15 * 7, n_hidden=args.n_hidden).to(device)
    model.set_physical_bounds(y_mean, y_std)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    rows = []
    for scenario in ["ssp126","ssp585"]:
        test_x, test_y0 = load_test_data(args.data_dir,scenario)
        if not args.no_decode_y0:
            test_y0, decoded = decode_test_y0(test_y0)
            print(f"{scenario}: y0 {'decoded' if decoded else 'left as stored'}")
        if args.clip_y0 or not args.no_decode_y0:
            nonneg = [0, 1, 2, 3, 6]
            test_y0[..., nonneg] = np.maximum(test_y0[..., nonneg], 0)
        pred = predict_test(model,test_x,test_y0,x_mean,x_std,y_mean,y_std,device,args.batch_sites)
        for site in range(pred.shape[0]):
            for a, age in enumerate(AGES):
                rows.append([f"{scenario}_{site}_{age}",
                             float(pred[site, a, 0]) / HEIGHT_MEAN,
                             float(pred[site, a, 1]) / AGB_MEAN])
        print(f"{scenario}: {pred.shape[0]} sites")

    out_path = args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "height", "agb"])
        writer.writerows(rows)
    print(f"saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

