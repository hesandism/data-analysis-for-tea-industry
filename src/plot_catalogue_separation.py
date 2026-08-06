"""Mean price per catalogue across every sale — shows the four catalogues occupy
separate price bands with no overlap.

Output: data/results/catalogue_mean_price_separation.{png,pdf}
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "final_clean_dataset_long.csv"
OUT = ROOT / "data" / "results"

# Categorical slots 1-4 (blue, orange, aqua, yellow), assigned in fixed order.
SERIES = [
    ("high_grown", "High Grown", "#2a78d6"),
    ("low_grown", "Low Grown", "#eb6834"),
    ("off_grade", "Off-Grade", "#1baf7a"),
    ("dust", "Dust", "#eda100"),
]

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2dd"

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
    )
}


def parse_sale_date(raw: str):
    """'24TH/25TH October 2023' -> Timestamp('2023-10-24')."""
    if not isinstance(raw, str):
        return pd.NaT
    m = re.search(r"(\d{1,2})\s*(?:ST|ND|RD|TH)?", raw, flags=re.I)
    mon = re.search(r"([A-Za-z]+)\s+(\d{4})", raw)
    if not (m and mon and mon.group(1).lower() in MONTHS):
        return pd.NaT
    return pd.Timestamp(int(mon.group(2)), MONTHS[mon.group(1).lower()], int(m.group(1)))


def break_long_gaps(s: pd.Series, max_gap_days: int = 28) -> pd.Series:
    """Insert a NaN wherever consecutive sales are far apart, so the line breaks
    across genuinely missing weeks (2025 sales 1-14) instead of drawing a straight
    interpolation through them. Single skipped sales stay connected."""
    breaks = s.index[s.index.to_series().diff() > pd.Timedelta(days=max_gap_days)]
    if len(breaks) == 0:
        return s
    filler = pd.Series(float("nan"), index=breaks - pd.Timedelta(days=7))
    return pd.concat([s, filler]).sort_index()


def main() -> None:
    df = pd.read_csv(DATA)
    df["sale_date"] = df["sale_date_raw"].map(parse_sale_date)

    start_date = pd.Timestamp(2025, 4, 1)
    end_date = pd.Timestamp(2026, 3, 31)

    wide = (
        df.dropna(subset=["sale_date", "price_mid_lkr"])
        .loc[lambda frame: frame["sale_date"].between(start_date, end_date)]
        .pivot_table(index="sale_date", columns="catalogue", values="price_mid_lkr", aggfunc="mean")
        .sort_index()
    )


    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for key, label, color in SERIES:
        s = break_long_gaps(wide[key].dropna())
        ax.plot(s.index, s.values, lw=2, color=color, label=label, solid_capstyle="round")
        # direct label at the right edge (identity is never colour-alone)
        ax.annotate(
            label,
            xy=(s.index[-1], s.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=INK,
        )

    ax.set_title(
        "The prices of the tea catalogues",
        fontsize=14,
        color=INK,
        pad=14,
        loc="left",
    )
    ax.set_ylabel("Mean price (LKR/kg)", fontsize=10, color=INK_MUTED)
    ax.set_xlabel("Sale date", fontsize=10, color=INK_MUTED)

    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)

    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=4, labelcolor=INK_MUTED)
    ax.margins(x=0.02)
    ax.set_xlim(right=wide.index.max() + pd.Timedelta(days=95))

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"catalogue_mean_price_separation.{ext}", dpi=200, facecolor=fig.get_facecolor())
    print(wide.describe().round(0).to_string())
    print(f"\nsaved -> {OUT / 'catalogue_mean_price_separation.png'}")


if __name__ == "__main__":
    main()
