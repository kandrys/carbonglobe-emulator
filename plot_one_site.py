"""Jedno miejsce z CarbonGlobe: drivery i 7 targetów przez ~40 lat.

Dane to wycinki ściągnięte zakresami HTTP (nie cały zbiór):
  data/x_amazon.bin  kształt (40, 12, 136)  wymuszenie miesięczne
  data/y_amazon.bin  kształt (41, 12, 7)    stany roczne, 12 miesięcy jest kopią

Miejsce: site 43609, 3.25°S, 60.25°W, wiek startowy lasu = 1 (plik glob_Y001).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures" / "jedno_miejsce_40_lat.png"

TARGETS = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
TARGET_PL = {
    "height": "wysokość",
    "agb": "biomasa nadziemna",
    "soil": "węgiel w glebie",
    "lai": "indeks liści",
    "gpp": "fotosynteza",
    "npp": "produkcja netto",
    "rh": "oddychanie gleby",
}

# Indeksy, które da się rozpoznać po wartościach i opisie zbioru.
# 0: temperatura powietrza ~283 K globalnie
# 1: opad, w Amazonii suma roczna ~2300 mm
# 8: jeden z 24 kanałów CO2, trend ~1.6 ppm/rok
TEMP, PRECIP, CO2 = 0, 1, 8


def load():
    x = np.frombuffer((DATA / "x_amazon.bin").read_bytes(), dtype="<f4").reshape(40, 12, 136)
    y = np.frombuffer((DATA / "y_amazon.bin").read_bytes(), dtype="<f4").reshape(41, 12, 7)
    y = y[:, 0, :]  # miesiące w Y są identyczne — zostaje wartość roczna
    return x, y


def main():
    x, y = load()
    driver_years = np.arange(1981, 2021)
    target_years = np.arange(1981, 2022)

    temp_c = x[:, :, TEMP].mean(axis=1) - 273.15
    precip_mm = x[:, :, PRECIP].sum(axis=1)
    co2 = x[:, :, CO2].mean(axis=1)

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=False)
    fig.suptitle(
        "Jedno miejsce, 40 lat  ·  Amazonia 3.25°S, 60.25°W  ·  las zasiany w wieku 1 roku",
        fontsize=13,
    )

    drivers = [
        (axes[0, 0], driver_years, temp_c, "Temperatura powietrza", "°C, średnia roczna", "#1d4e89"),
        (axes[0, 1], driver_years, precip_mm, "Opad", "mm / rok", "#2a9d8f"),
        (axes[0, 2], driver_years, co2, "CO₂ (jeden kanał z 24)", "ppm, średnia roczna", "#b08900"),
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
        "Drivery są miesięczne.\n"
        "Tu: rok = średnia albo suma\n"
        "z 12 miesięcy.\n\n"
        "Targety w pliku są roczne\n"
        "(12 miesięcy to kopia\n"
        "tej samej wartości).\n\n"
        "Oś lat: 1981–2021\n"
        "według opisu CarbonGlobe.\n"
        "Wymuszenie ma 40 lat,\n"
        "stan lasu ma 41 punktów\n"
        "(start + 40 kroków).",
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
        ax.set_title(f"{name}  ·  {TARGET_PL[name]}", loc="left", fontsize=10)
        ax.set_xlim(1981, 2021)
        ax.grid(True, alpha=0.3)

    for i, name in enumerate(["gpp", "npp", "rh"]):
        ax = axes[2, i]
        ax.plot(target_years, y[:, TARGETS.index(name)], color=flux_colors[name], lw=1.6)
        ax.set_title(f"{name}  ·  {TARGET_PL[name]}", loc="left", fontsize=10)
        ax.set_xlim(1981, 2021)
        ax.set_xlabel("rok")
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_ylabel("stan (jednostki ED)")
    axes[2, 0].set_ylabel("strumień (jednostki ED)")
    axes[2, 3].axis("off")
    axes[2, 3].text(
        0.0,
        0.55,
        "Na co patrzeć\n\n"
        "height, agb, soil:\n"
        "prawie gładki wzrost.\n"
        "Las się kumuluje.\n\n"
        "gpp, npp, rh:\n"
        "też rosną (młody las),\n"
        "ale rok do roku skaczą.\n\n"
        "lai jest pośrodku:\n"
        "jest trend i są ząbki.",
        fontsize=8,
        va="center",
        transform=axes[2, 3].transAxes,
    )

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    print(FIG)


if __name__ == "__main__":
    main()
