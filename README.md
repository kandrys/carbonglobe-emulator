# CarbonGlobe emulator

An autoregressive machine-learning emulator of the Ecosystem Demography (ED) model, built for the
[IEEE BigData Cup 2026: AI Emulation Challenge](https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge)
on the [CarbonGlobe](https://huggingface.co/datasets/zhwang1/CarbonGlobe) dataset (NeurIPS 2025).

Given the initial state of a forest and 40 years of monthly environmental forcing, the model rolls the
forest forward one year at a time and predicts its state after 40 years.

![One site over 40 years: drivers and the 7 targets](figures/jedno_miejsce_40_lat.png)

## Task

- **Inputs:** 136 drivers (meteorology, CO₂, soil) per month, 12 months per year, 40 years.
- **State:** 7 annual ecosystem variables: `height`, `agb`, `soil`, `lai`, `gpp`, `npp`, `rh`.
- **Initial conditions:** 15 seed forest ages: 1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500.
- **Scoring:** mean squared error of year-40 `height` and `agb`, each divided by its global mean
  (`height / 10.90645`, `agb / 3.39567`). Lower is better.

## Approach

The model learns one annual step:

```
y[t+1] = y[t] + f(y[t], x[t], age[t])
```

- `y[t]`: normalized 7-variable state in year `t`
- `x[t]`: normalized forcing for year `t` (12 months × 136 drivers, flattened)
- `age[t]`: forest age in year `t`, divided by 500
- `f`: an MLP (LayerNorm, 3 hidden ReLU layers, 7 outputs) predicting the state increment

Training samples are segments of `k` years. The model is unrolled `k` steps on its own predictions and the
loss is the MSE averaged over all steps. With `k = 1` this is plain teacher forcing; larger `k` trains the
model to correct its own accumulated errors (drift).

Validation always runs the full 40-year rollout from the initial state, exactly like the Kaggle test, and
reports the competition metric.

## Data

Download CarbonGlobe from [Hugging Face](https://huggingface.co/datasets/zhwang1/CarbonGlobe) or
[Kaggle](https://www.kaggle.com/datasets/zhihaow/carbonglobe) (~29 GB) into `data/`:

```
data/
├── data_global/
│   ├── glob_X_fea.npy      (54152, 40, 12, 136)  forcing: site × year × month × driver
│   ├── glob_Y001.npy       (54152, 41, 12, 7)    states for seed age 1: site × year × month × target
│   └── ...                 one file per seed age, up to glob_Y500.npy
└── data_stats/
    └── data_stats.npz      x_mean, x_std, y_mean, y_std
```

Arrays are opened with `mmap_mode="r"`, so only the selected sites are read into memory.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

The script splits sites into train and validation, trains the model, prints the validation score after
every epoch, and saves the best model to `best_model.pt`.

All settings (paths, number of sites, `k`, batch size, learning rate, epochs) live in `config.py`.

## Project structure

| File | Contents |
|---|---|
| `main.py` | data loading, site split, dataset, model, training, validation |
| `config.py` | all hyperparameters and paths |
| `plot_one_site.py` | plot of drivers and targets for a single site |
| `figures/` | generated plots |

## Citation

```bibtex
@inproceedings{wang2025carbonglobe,
  title     = {CarbonGlobe: A Global-Scale, Multi-Decade Dataset and Benchmark for Carbon Forecasting in Forest Ecosystems},
  author    = {Wang, Zhihao and Ma, Lei and Hurtt, George and Jia, Xiaowei and Li, Yanhua and Li, Ruohan and Li, Zhili and Xu, Shuo and Xie, Yiqun},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2025}
}
```
