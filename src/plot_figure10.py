"""
plot_figure10.py
================
Generates Figure 10 from the paper:

  "Portfolio Cumulative Log Returns (Volatility Scaling): vol_target = 10.0%"

Color scheme exactly matching paper Figure 10:
  Linear-Ridge -> yellow
  Naive        -> red
  MVO          -> blue (steelblue, thin lines)
  BL           -> green (thin lines)
  EW           -> magenta / pink (thick)
  SPY          -> cyan (thick)

Run AFTER all three backtest scripts:
  python src/backtest_naive_strategy.py
  python src/backtest_bl_strategy.py
  python src/backtest_ridge_strategy.py
  python src/plot_figure10.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

NAIVE_PATH = PROCESSED_DIR / "naive_strategy_returns_paper_like.csv"
BL_PATH    = PROCESSED_DIR / "bl_strategy_returns_paper_like.csv"
RIDGE_PATH = PROCESSED_DIR / "ridge_strategy_returns_paper_like.csv"
ETF_PATH   = PROCESSED_DIR / "etf_monthly_returns_paper_like.csv"

ETF_TICKERS   = ["SPY","XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]
VOL_TARGET    = 0.10
VOL_LOOKBACK  = 12
ANNUALIZATION = 12
VOL_SCALE_CAP = 5.0


# ============================================================
# Helpers
# ============================================================

def apply_vol_scaling(log_returns: pd.Series) -> pd.Series:
    r      = log_returns.dropna()
    scaled = pd.Series(np.nan, index=r.index)
    for i in range(len(r)):
        if i < VOL_LOOKBACK:
            continue
        past         = r.iloc[i - VOL_LOOKBACK: i]
        realised_vol = past.std(ddof=1) * np.sqrt(ANNUALIZATION)
        scale        = (min(VOL_TARGET / realised_vol, VOL_SCALE_CAP)
                        if realised_vol > 1e-8 else 1.0)
        scaled.iloc[i] = r.iloc[i] * scale
    return scaled.dropna()


def load_strategy_df(path: Path, label: str) -> "pd.DataFrame | None":
    if not path.exists():
        print(f"  WARNING: {path.name} not found — {label} strategies skipped.")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"  Loaded {label:10s}: {df.shape[1]} strategies, "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    return df


# ============================================================
# Figure 10
# ============================================================

def plot_figure10():
    print("=" * 70)
    print("GENERATING FIGURE 10 — ALL STRATEGIES")
    print("=" * 70)

    # ── Load strategy returns ──────────────────────────────────────────
    print("\nLoading strategy returns …")
    naive_df = load_strategy_df(NAIVE_PATH, "Naive")
    bl_df    = load_strategy_df(BL_PATH,    "BL")
    ridge_df = load_strategy_df(RIDGE_PATH, "Ridge")

    strategy_groups = {
        "ridge": (ridge_df, "gold",       0.90, "Linear-Ridge"),
        "naive": (naive_df, "red",        0.85, "Naive"),
        "bl":    (bl_df,   "limegreen",   0.80, "BL"),
    }

    # ── Find common index across all available series ──────────────────
    all_indices = []
    for df, *_ in strategy_groups.values():
        if df is not None:
            all_indices.append(df.index)

    if not all_indices:
        raise RuntimeError("No strategy files found. Run the backtest scripts first.")

    common = all_indices[0]
    for idx in all_indices[1:]:
        common = common.intersection(idx)
    common = common.sort_values()
    print(f"\nCommon window: {common.min().date()} -> {common.max().date()} "
          f"({len(common)} months)")

    # ── Load + vol-scale benchmarks ───────────────────────────────────
    print("\nLoading benchmarks …")
    etf       = pd.read_csv(ETF_PATH, index_col=0, parse_dates=True)[ETF_TICKERS]
    spy_scaled = apply_vol_scaling(etf["SPY"].reindex(common).dropna())
    ew_scaled  = apply_vol_scaling(etf.reindex(common).mean(axis=1).dropna())
    common     = common.intersection(spy_scaled.index).intersection(ew_scaled.index)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("#f5f5f5")
    ax.grid(color="white", linewidth=0.8)

    # Strategy lines (thin, semi-transparent)
    # Plot in order: BL bottom, Naive middle, Ridge top (matches paper visual)
    plot_order = ["bl", "naive", "ridge"]
    for model in plot_order:
        df, color, alpha, label = strategy_groups[model]
        if df is None:
            continue
        for col in df.columns:
            series = df[col].reindex(common).dropna()
            if series.empty:
                continue
            cum = series.cumsum()
            ax.plot(cum.index, cum.values,
                    color=color, linewidth=0.9, alpha=alpha)

    # Benchmark lines (thick, on top)
    # Paper: EW = magenta/pink, SPY = cyan
    ew_cum  = ew_scaled.reindex(common).cumsum()
    spy_cum = spy_scaled.reindex(common).cumsum()

    ax.plot(ew_cum.index,  ew_cum.values,
            color="magenta", linewidth=2.2, zorder=10, label="EW")
    ax.plot(spy_cum.index, spy_cum.values,
            color="cyan",    linewidth=2.2, zorder=10, label="SPY")

    ax.axhline(0, color="grey", lw=0.6, ls="--")

    # ── Legend matching paper Figure 10 order: EW, SPY, Linear-Ridge, Naive, MVO, BL
    legend_handles = [
        Line2D([0],[0], color="magenta",   lw=2.2, label="EW"),
        Line2D([0],[0], color="cyan",      lw=2.2, label="SPY"),
        Line2D([0],[0], color="gold",      lw=2.0, label="Linear-Ridge"),
        Line2D([0],[0], color="red",       lw=2.0, label="Naive"),
        Line2D([0],[0], color="steelblue", lw=2.0, label="MVO"),   # note: MVO shown in BL paper fig
        Line2D([0],[0], color="limegreen", lw=2.0, label="BL"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=9, framealpha=0.9)

    ax.set_title(
        "Portfolio Cumulative Log Returns (Volatility Scaling): "
        f"vol_target = {VOL_TARGET*100:.1f}%",
        fontsize=11,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Returns (Vol. Target of 10%)")

    fig.tight_layout()
    out = OUTPUTS_DIR / "figure10_all_strategies.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    plot_figure10()