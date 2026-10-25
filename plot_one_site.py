"""One CarbonGlobe site: selected drivers and all 7 targets over 40 years.

Reads a single site from the full arrays via memmap (nothing else is loaded into RAM):
  data/data_global/glob_X_fea.npy       (54152, 40, 12, 136)  monthly forcing
  data/data_global/glob_Y{age:03d}.npy  (54152, 41, 12, 7)    annual states, 12 months are copies

Default: site 43609 (3.25°S, 60.25°W, Amazon), seed forest age 1.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "data_global"
FIG_DIR = ROOT / "figures"

TARGETS = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
TARGET_NAMES = {
    "height": "canopy height",
    "agb": "aboveground biomass",
    "soil": "soil carbon",
    "lai": "leaf area index",
    "gpp": "gross primary production",
    "npp": "net primary production",
    "rh": "heterotrophic respiration",
}

# Inferred from value ranges, not documented: 0 ≈ air temperature (K), 1 ≈ precipitation, 8 = one of 24 CO2 channels
TEMP, PRECIP, CO2 = 0, 1, 8


def load(site: int, age: int):
    X = np.load(DATA / "glob_X_fea.npy", mmap_mode="r")
    Y = np.load(DATA / f"glob_Y{age:03d}.npy", mmap_mode="r")
    x = np.asarray(X[site], dtype=np.float32)
    y = np.asarray(Y[site, :, 0, :], dtype=np.float32)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=int, default=43609)
    parser.add_argument("--age", type=int, default=1)
    args = parser.parse_args()

    x, y = load(args.site, args.age)
    driver_years = np.arange(1981, 2021)
    target_years = np.arange(1981, 2022)

    temp_c = x[:, :, TEMP].mean(axis=1) - 273.15
    precip = x[:, :, PRECIP].sum(axis=1)
    co2 = x[:, :, CO2].mean(axis=1)

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=False)
    fig.suptitle(
        f"One site, 40 years  ·  site {args.site}  ·  seed forest age {args.age}",
        fontsize=13,
    )

    drivers = [
        (axes[0, 0], driver_years, temp_c, "Air temperature", "°C, annual mean", "#1d4e89"),
        (axes[0, 1], driver_years, precip, "Precipitation", "annual sum", "#2a9d8f"),
        (axes[0, 2], driver_years, co2, "CO₂ (one of 24 channels)", "ppm, annual mean", "#b08900"),
    ]
    for ax, years, series, title, ylab, color in drivers:
        ax.plot(years, series, color=color, lw=1.6)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlim(1981, 2021)
        ax.grid(True, alpha=0.3)

    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.0,
        0.55,
        "Drivers are monthly.\n"
        "Here: year = mean or sum\n"
        "of 12 months.\n\n"
        "Targets are annual\n"
        "(the 12 months are copies\n"
        "of the same value).\n\n"
        "Years: 1981–2021\n"
        "per the CarbonGlobe paper.\n"
        "Forcing has 40 years,\n"
        "forest state has 41 points\n"
        "(initial + 40 steps).",
        fontsize=8,
        va="center",
        family="sans-serif",
        transform=axes[0, 3].transAxes,
    )

    stock_colors = {"height": "#1b4332", "agb": "#2d6a4f", "soil": "#9c6644", "lai": "#52b788"}
    flux_colors = {"gpp": "#9b2226", "npp": "#bb3e03", "rh": "#ca6702"}

    for i, name in enumerate(["height", "agb", "soil", "lai"]):
        ax = axes[1, i]
        ax.plot(target_years, y[:, TARGETS.index(name)], color=stock_colors[name], lw=1.6)
        ax.set_title(f"{name}  ·  {TARGET_NAMES[name]}", loc="left", fontsize=10)
        ax.set_xlim(1981, 2021)
        ax.grid(True, alpha=0.3)

    for i, name in enumerate(["gpp", "npp", "rh"]):
        ax = axes[2, i]
        ax.plot(target_years, y[:, TARGETS.index(name)], color=flux_colors[name], lw=1.6)
        ax.set_title(f"{name}  ·  {TARGET_NAMES[name]}", loc="left", fontsize=10)
        ax.set_xlim(1981, 2021)
        ax.set_xlabel("year")
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_ylabel("stock (ED units)")
    axes[2, 0].set_ylabel("flux (ED units)")
    axes[2, 3].axis("off")
    axes[2, 3].text(
        0.0,
        0.55,
        "What to look for\n\n"
        "height, agb, soil:\n"
        "almost smooth growth.\n"
        "The forest accumulates.\n\n"
        "gpp, npp, rh:\n"
        "also rising (young forest),\n"
        "but jumpy year to year.\n\n"
        "lai sits in between:\n"
        "a trend plus wiggles.",
        fontsize=8,
        va="center",
        transform=axes[2, 3].transAxes,
    )

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"site_{args.site}_age_{args.age:03d}.png"
    fig.savefig(fig_path, dpi=140)
    print(fig_path)


if __name__ == "__main__":
    main()
