from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("data")
    ckpt_path: Path = Path("best_model.pt")

    seed: int = 67
    val_frac: float = 0.1
    n_train_sites: int = 3000
    n_val_sites: int = 500

    k: int = 1
    batch_size: int = 512
    accum_steps: int = 1
    n_hidden: int = 256
    lr: float = 1e-3
    epochs: int = 5
