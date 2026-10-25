

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List
from config import Config

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
HEIGHT_MEAN = 10.90645
AGB_MEAN = 3.39567

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def split_sites(n_sites :int , val_frac: float, seed: int) ->List[int,int]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_sites)
    a = rng.permutation(idx)
    n_val = int(n_sites * val_frac)
    val_idx = a[:n_val]
    train_idx = a[n_val:]
    return train_idx, val_idx


def load_data(data:Path):
    AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    TARGETS = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
    X = np.load(data / "data_global" / "glob_X_fea.npy", mmap_mode="r")
    Y = {age: np.load(data / "data_global" / f"glob_Y{age:03d}.npy", mmap_mode="r") for age in AGES}
    stats = np.load(data / "data_stats" / "data_stats.npz")
    x_mean = stats["x_mean"].astype(np.float32)
    x_std  = stats["x_std"].astype(np.float32)
    y_mean = stats["y_mean"].astype(np.float32)
    y_std  = stats["y_std"].astype(np.float32)
    return X,x_mean,x_std,Y,y_mean,y_std
def norm(x: np.ndarray,x_mean:float,x_std:float, EPS :float = 1e-6)->float:
    return (x - x_mean) / (x_std + EPS)
def denorm_y(y_norm: np.ndarray,y_mean:float,y_std:float, EPS :float = 1e-6)->float:
    return y_norm * (y_std + EPS) + y_mean
class CarbonDataset(Dataset):
    def __init__(self, X: np.ndarray, Y:Dict[int,np.ndarray], site_idx:np.ndarray, k: int,
                 x_mean: np.ndarray,x_std:float,y_mean:np.ndarray,y_std:float):
        if k >40 or k<1:
            raise ValueError(f"bad k value {k }")
        self.k = k
        x_ram = X[site_idx]
        y_list = [Y[age][site_idx, :, 0, :] for age in AGES] 

        self.x = norm(x_ram,x_mean,x_std)
        y_n = norm(y_list,y_mean,y_std)
        self.y = np.stack(y_n,axis = 1)
    def __len__(self):
        return self.x.shape[0] * 15 * (40-self.k+1)
    def __getitem__(self, idx):
        i, r =divmod(idx,len(AGES) * (40 - self.k + 1))
        a,t = divmod(r,40-self.k+1)
        y_start = self.y[i,a,t] 
        x_seg = self.x[i,t:t+self.k]
        y_targets = self.y[i,a,t+1:t+self.k+1]
        age_start = (AGES[a]+t) / 500
        return torch.tensor(x_seg,dtype= torch.float32), torch.tensor(y_start,dtype= torch.float32), torch.tensor(y_targets,dtype = torch.float32), torch.tensor(age_start,dtype=torch.float32) 

class StepModel(nn.Module):
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.head = nn.Sequential(
                nn.LayerNorm(n_in),
                nn.Linear(n_in,n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden,n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden,n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden,7)
                )
    def forward(self, y_t, x_t, age_t):
        x = x_t.flatten(start_dim = 1)
        age_t1 = age_t.unsqueeze(1)
        data  = torch.cat([x,age_t1,y_t],dim=1)
        return y_t + self.head(data)
def train_one_epoch(
        model:nn.Module,
        loader:DataLoader,
        optimizer:torch.optim.Optimizer,
        criterion: nn.Module,
        cfg:Config,
        epoch: int,
        device:torch.device)-> float:
    model.train()
    accum = max(1,cfg.accum_steps)
    optimizer.zero_grad(set_to_none=True)
    total,n = 0.0,0
    for step , (x_seg, y_start, y_targets, age) in enumerate(loader):
        x_seg = x_seg.to(device)
        y_start = y_start.to(device)
        y_targets = y_targets.to(device)
        age = age.to(device)
        loss = 0 
        y = y_start
        for j in range(x_seg.shape[1]):
            age_t = age + j / 500
            y = model(y,x_seg[:,j],age_t)
            loss = loss + criterion(y,y_targets[:,j])
        loss = loss / x_seg.shape[1] / accum
        loss.backward()
        if (step+1) % accum==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total += float(loss.detach().item())* accum
        n+=1
    if n %accum !=0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    lrs = ", ".join(f"{g['lr']:.2e}"for g in optimizer.param_groups)
    print(f"LR at end of epoch: {lrs}")
    return total / max(n,1)
@torch.no_grad()
def validate(
        model:nn.Module,
        loader:DataLoader,
        y_mean:np.ndarray,
        y_std:np.ndarray,
        device:torch.device)-> float:
    model.eval()
    preds, trues = [], []
    for x_seg, y_start, y_targets, age in loader:
        x_seg = x_seg.to(device)
        age = age.to(device)
        y = y_start.to(device)
        for j in range(x_seg.shape[1]):
            y = model(y,x_seg[:,j],age + j / 500)
        preds.append(y.cpu().numpy())
        trues.append(y_targets[:,-1].numpy())
    pred = denorm_y(np.concatenate(preds),y_mean,y_std)
    true = denorm_y(np.concatenate(trues),y_mean,y_std)
    mse_h = np.mean((pred[:,0] / HEIGHT_MEAN - true[:,0] / HEIGHT_MEAN) ** 2)
    mse_a = np.mean((pred[:,1] / AGB_MEAN - true[:,1] / AGB_MEAN) ** 2)
    return float(0.5 * (mse_h + mse_a))
def main():
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X, x_mean, x_std, Y, y_mean, y_std = load_data(cfg.data_dir)
    train_idx, val_idx = split_sites(X.shape[0], cfg.val_frac, cfg.seed)
    val_ds = CarbonDataset(X, Y, site_idx=val_idx[:cfg.n_val_sites], k=40,
                           x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    train_ds = CarbonDataset(X, Y, site_idx=train_idx[:cfg.n_train_sites], k=cfg.k,
                             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = StepModel(n_in=7 + 12 * 136 + 1, n_hidden=cfg.n_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    best = float("inf")
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, epoch, device)
        score = validate(model, val_loader, y_mean, y_std, device)
        print(f"epoch {epoch + 1}/{cfg.epochs}  loss {loss:.4f}  val {score:.4f}")
        if score < best:
            best = score
            torch.save(model.state_dict(), cfg.ckpt_path)
    print(f"best validation score: {best:.4f}")


if __name__ == "__main__":
    main()
