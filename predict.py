
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

        x = torch.tensor(norm(test_x[s : s + batch_sites],x_mean,x_std),dtype=torch.float32)
        y = torch.tensor(norm(test_y0[s : s+ batch_sites],y_mean,y_std),dtype=torch.float32)
        x = x.to(device)
        y = y.to(device).reshape(-1,7)
        x = torch.repeat_interleave(x,repeats = 15,dim=0)
        ages = torch.tensor(AGES,dtype=torch.float32)
        ages = ages.repeat(x.shape[0]// 15) / 500
        ages = ages.to(device)
        for i in range(40):
            y = model(y,x[:,i],ages + i / 500 )
        y = y.cpu().numpy()
        y = denorm_y(y,y_mean,y_std).reshape(x.shape[0]// 15,15,7)
        out.append(y)
    return np.concatenate(out)   # (n_sites, 15, 7)






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
    parser.add_argument("--data_dir", type=Path, default=cfg.data_dir)
    return parser.parse_args()


def main():

    args = parse_args()
    print(f"model {args.ckpt_path}  n_hidden {args.n_hidden}  clip_y0 {args.clip_y0}  out {args.out}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = StepModel(n_in=7 + 12 * 136 + 1, n_hidden=args.n_hidden).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    x_mean, x_std, y_mean, y_std = load_stats(args.data_dir)

    rows = []
    for scenario in ["ssp126","ssp585"]:
        test_x, test_y0 = load_test_data(args.data_dir,scenario)
        if args.clip_y0:
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

