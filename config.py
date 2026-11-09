from dataclasses import dataclass
from pathlib import Path
from typing import List,Tuple
from dataclasses import field

@dataclass
class Config:
    data_dir: Path = Path("data")
    ckpt_path: Path = Path("best_model.pt")
    k_schedule: List[Tuple[int, int]] = field(default_factory=lambda: [(1, 3), (5, 3), (10, 3), (40, 5)])
    seed: int = 67
    val_frac: float = 0.1
    n_train_sites: int = 3000
    n_val_sites: int = 500
    loss_weights: Tuple[float] = (0.9, 2.4, 0.3, 0.3, 0.3, 0.3, 0.3) 
    batch_size: int = 512
    accum_steps: int = 1
    n_hidden: int = 256
    lr: float = 1e-3
    noise_sigma: float = 0.01
    chunk_sites: int = 8000
