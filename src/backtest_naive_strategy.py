"""
backtest_naive_strategy.py
==========================
Naive regime-conditioned strategy  (Paper Section 5.2.1, Eqs. 8-10).

Fixes vs previous version:
  1. Forecast fallback: when predicted regime has < 2 months in the window,
     fall back to ALL months in window instead of returning None.
     This prevents the long flat-line period visible in Figure 11.
  2. Table 3 Nemenyi ranks: use a proper 2-group ranking (rank 1 vs 2),
     averaged across strategies — matching the paper's rank format.
  3. p-value: one-sided paired t-test on per-strategy (treatment - control_mean)
     differences. Controls the direction correctly.

Outputs (outputs/):
  table4_naive_performance.csv
  table3_nemenyi_naive.csv
  figure7_naive_boxplot.png
  figure11_naive_vs_benchmarks.png
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ETF_RETURNS_PATH       = PROCESSED_DIR / "etf_monthly_returns_paper_like.csv"
NEXT_REGIME_PROBS_PATH = PROCESSED_DIR / "next_regime_probabilities_paper_like.csv"
REGIME_LABELS_PATH     = PROCESSED_DIR / "regime_labels_paper_like.csv"


# ============================================================
# 2. Settings
# ============================================================

ETF_TICKERS = [
    "SPY", "XLB", "XLE", "XLF", "XLI",
    "XLK", "XLP", "XLU", "XLV", "XLY",
]

TOP_L_LIST        = [2, 3, 4]
POSITION_METHODS  = ["lo", "lns", "los", "mx"]
ESTIMATION_WINDOW = 48        # paper Section 6
VOL_TARGET        = 0.10      # 10% annualised
VOL_LOOKBACK      = 12        # months for trailing vol estimate
VOL_SCALE_CAP     = 5.0       # max leverage multiplier
ANNUALIZATION     = 12
N_RANDOM_RUNS     = 50        # Monte Carlo runs for random-regime baseline


# ============================================================
# 3. Load data
# ============================================================

def load_inputs():
    print("=" * 70)
    print("LOADING INPUT DATA")
    print("=" * 70)

    for p in [ETF_RETURNS_PATH, NEXT_REGIME_PROBS_PATH, REGIME_LABELS_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run preprocess_data.py -> detect_regimes.py first."
            )

    etf_returns   = pd.read_csv(
        ETF_RETURNS_PATH, index_col=0, parse_dates=True)[ETF_TICKERS]
    next_probs    = pd.read_csv(
        NEXT_REGIME_PROBS_PATH, index_col=0, parse_dates=True)
    regime_labels = pd.read_csv(
        REGIME_LABELS_PATH, index_col=0, parse_dates=True)["regime"].astype(int)

    print(f"ETF returns   : {etf_returns.shape}  "
          f"{etf_returns.index.min().date()} -> {etf_returns.index.max().date()}")
    print(f"Next probs    : {next_probs.shape}  "
          f"{next_probs.index.min().date()} -> {next_probs.index.max().date()}")
    print(f"Regime labels : {regime_labels.shape}  "
          f"{regime_labels.index.min().date()} -> {regime_labels.index.max().date()}")

    return etf_returns, next_probs, regime_labels


# ============================================================
# 4. Timing alignment
# ============================================================

def get_predicted_regimes(next_probs: pd.DataFrame) -> pd.Series:
    """
    Paper Eq.(8): i*_{t+1} = argmax_i p~_{i,t+1}
    next_probs[t] = forecast made at t for t+1.
    Shift index +1 month: signal at t -> position at t+1.
    """
    col0   = next_probs.columns[0]
    # Strip everything that is not a digit to get the prefix
    prefix = col0.rstrip("0123456789").rstrip("_")

    predicted = next_probs.idxmax(axis=1).str.replace(
        prefix + "_", "", regex=False
    ).str.replace(prefix, "", regex=False).astype(int)

    shifted       = predicted.copy()
    shifted.index = shifted.index + pd.DateOffset(months=1)
    return shifted


def build_common_index(etf_returns, predicted_shifted, regime_labels):
    common = (
        etf_returns.index
        .intersection(predicted_shifted.index)
        .intersection(regime_labels.index)
    ).sort_values()

    # Enforce estimation-window warm-up
    warmup_end = etf_returns.index.min() + pd.DateOffset(months=ESTIMATION_WINDOW)
    common     = common[common >= warmup_end]

    print(f"\nBacktest window: {common.min().date()} -> {common.max().date()}  "
          f"({len(common)} months)")
    return common


# ============================================================
# 5. Naive forecast — Paper Eqs.(9-10)
# ============================================================

def compute_naive_forecast(
    position_date: pd.Timestamp,
    predicted_regime: int,
    etf_returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> pd.Series:
    """
    Paper Section 5.2.1, Eqs.(9-10):
        y_hat^{j,naive}_{t+1} = mu_hat* / sigma_hat*

    Uses the 48-month estimation window ending at position_date.
    mu_hat*, sigma_hat* from months in the window whose hard
    regime label == predicted_regime.

    FIX: If the predicted regime has < 2 months in the window,
    fall back to ALL months in the window (unconditional Sharpe).
    This prevents zero-return flat periods when a regime is rare.
    """
    hist_ret = etf_returns.loc[etf_returns.index < position_date]
    hist_reg = regime_labels.loc[regime_labels.index < position_date]

    if len(hist_ret) < ESTIMATION_WINDOW:
        return None

    window_ret = hist_ret.iloc[-ESTIMATION_WINDOW:]
    window_reg = hist_reg.reindex(window_ret.index)

    same_idx = window_reg[window_reg == predicted_regime].index
    same_idx = same_idx.intersection(window_ret.index)

    # Fallback to full window if not enough same-regime months
    if len(same_idx) < 2:
        same_idx = window_ret.index

    r      = window_ret.loc[same_idx]
    mu     = r.mean()
    sigma  = r.std(ddof=1).replace(0.0, np.nan)
    sharpe = (mu / sigma).replace([np.inf, -np.inf], np.nan)

    return sharpe.reindex(ETF_TICKERS)


# ============================================================
# 6. Position sizing — Paper Eqs.(15-18)
# ============================================================

def _normalize_weights(selected: pd.Series) -> pd.Series:
    """w_j = y_hat_j / S_l  where S_l = sum |y_hat_j|."""
    w = pd.Series(0.0, index=ETF_TICKERS)
    if selected.empty:
        return w
    S = selected.abs().sum()
    if S < 1e-12:
        return w
    w.loc[selected.index] = selected.values / S
    return w


def position_lo(forecast: pd.Series, l: int) -> pd.Series:
    """Eq.(18) — long-only: top-l assets with positive forecast."""
    valid = forecast.dropna().sort_values(ascending=False)
    H     = valid.head(l)
    H_pos = H[H > 0]
    if H_pos.empty:
        # All forecasts negative — take top-l anyway (paper scales down)
        H_pos = H
    return _normalize_weights(H_pos)


def position_lns(forecast: pd.Series, l: int) -> pd.Series:
    """Eq.(16) — long-and-short: top-l long + bottom-l short."""
    valid = forecast.dropna().sort_values(ascending=False)
    H     = valid.head(l)
    L     = valid.tail(l)
    L     = L.loc[L.index.difference(H.index)]
    return _normalize_weights(pd.concat([H, L]))


def position_los(forecast: pd.Series, l: int) -> pd.Series:
    """Eq.(17) — long-or-short: top-l by |forecast| magnitude."""
    valid = forecast.dropna()
    if valid.empty:
        return pd.Series(0.0, index=ETF_TICKERS)
    B_idx = valid.abs().sort_values(ascending=False).head(l).index
    return _normalize_weights(valid.loc[B_idx])


def position_mx(forecast: pd.Series, l: int, regime: int) -> pd.Series:
    """Mixed: los if Regime 0 (crisis), lo otherwise."""
    if regime == 0:
        return position_los(forecast, l)
    return position_lo(forecast, l)


def get_weights(forecast, l: int, method: str,
                predicted_regime: int) -> pd.Series:
    if forecast is None:
        return pd.Series(0.0, index=ETF_TICKERS)
    f = forecast.reindex(ETF_TICKERS)
    if method == "lo":
        return position_lo(f, l)
    elif method == "lns":
        return position_lns(f, l)
    elif method == "los":
        return position_los(f, l)
    elif method == "mx":
        return position_mx(f, l, predicted_regime)
    raise ValueError(f"Unknown method: {method}")


# ============================================================
# 7. Volatility scaling — Paper Figures 10-13 (10% target)
# ============================================================

def apply_vol_scaling(log_returns: pd.Series) -> pd.Series:
    """
    Scale each month's return so trailing realised vol -> VOL_TARGET.
    First VOL_LOOKBACK months are dropped (warm-up).
    Scale factor capped at VOL_SCALE_CAP.
    """
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


# ============================================================
# 8. MVO benchmark
# ============================================================

def mvo_weights_longonly(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mu)
    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret / vol) if vol > 1e-8 else 0.0
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0, 1)] * n
    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 500})
    return res.x if res.success else np.ones(n) / n


def run_mvo_backtest(etf_returns: pd.DataFrame,
                     common_index: pd.DatetimeIndex) -> pd.Series:
    ret_list = []
    for date in common_index:
        hist = etf_returns.loc[etf_returns.index < date].iloc[-ESTIMATION_WINDOW:]
        if len(hist) < ESTIMATION_WINDOW:
            ret_list.append(np.nan)
            continue
        mu  = hist.mean().values
        cov = hist.cov().values + np.eye(len(ETF_TICKERS)) * 1e-6
        w   = mvo_weights_longonly(mu, cov)
        ret_list.append(float(etf_returns.loc[date].values @ w))
    raw = pd.Series(ret_list, index=common_index, name="mvo_lo").dropna()
    return apply_vol_scaling(raw)


# ============================================================
# 9. Performance metrics — Paper Tables 4-6 columns
# ============================================================

def performance_metrics(log_returns: pd.Series) -> dict:
    """Sharpe, Sortino, AvgDD, MaxDD, % Positive Ret. on log returns."""
    r = log_returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in
                ["sharpe", "sortino", "avg_drawdown",
                 "max_drawdown", "pct_positive"]}

    ann_ret  = r.mean() * ANNUALIZATION
    ann_vol  = r.std(ddof=1) * np.sqrt(ANNUALIZATION)
    sharpe   = ann_ret / ann_vol if ann_vol > 1e-8 else np.nan

    down     = r[r < 0]
    down_vol = (down.std(ddof=1) * np.sqrt(ANNUALIZATION)
                if len(down) > 1 else np.nan)
    sortino  = ann_ret / down_vol if (down_vol and down_vol > 1e-8) else np.nan

    cum_log     = r.cumsum()
    running_max = cum_log.cummax()
    dd          = cum_log - running_max
    avg_dd      = dd[dd < 0].mean() if (dd < 0).any() else 0.0
    max_dd      = dd.min()
    pct_pos     = (r > 0).mean()

    return {
        "sharpe"       : round(float(sharpe),  3),
        "sortino"      : round(float(sortino), 3),
        "avg_drawdown" : round(float(avg_dd),  3),
        "max_drawdown" : round(float(max_dd),  3),
        "pct_positive" : round(float(pct_pos), 3),
    }


# ============================================================
# 10. Main backtest loop
# ============================================================

def run_backtest(etf_returns, next_probs, regime_labels,
                 use_random_regimes: bool = False,
                 random_seed: int = 42) -> dict[str, pd.Series]:
    """
    Returns {strategy_name: vol_scaled_log_returns}.
    use_random_regimes=True shuffles the predicted regime series (control).
    """
    predicted_shifted = get_predicted_regimes(next_probs)
    common            = build_common_index(etf_returns, predicted_shifted, regime_labels)
    pred_series       = predicted_shifted.reindex(common).dropna().astype(int)

    if use_random_regimes:
        rng         = np.random.default_rng(random_seed)
        unique_reg  = np.array(sorted(regime_labels.unique()))
        pred_series = pd.Series(
            rng.choice(unique_reg, size=len(pred_series)),
            index=pred_series.index,
        )

    results = {}

    for l in TOP_L_LIST:
        for method in POSITION_METHODS:
            name       = f"naive_{method}_{l}"
            raw_list   = []

            for date in pred_series.index:
                pred_regime = int(pred_series.loc[date])

                forecast = compute_naive_forecast(
                    position_date    = date,
                    predicted_regime = pred_regime,
                    etf_returns      = etf_returns,
                    regime_labels    = regime_labels,
                )
                weights  = get_weights(forecast, l, method, pred_regime)
                port_ret = float((weights * etf_returns.loc[date]).sum())
                raw_list.append(port_ret)

            raw    = pd.Series(raw_list, index=pred_series.index, name=name)
            scaled = apply_vol_scaling(raw)
            results[name] = scaled

    return results


# ============================================================
# 11. Random regime baseline
# ============================================================

def run_random_baseline(etf_returns, next_probs, regime_labels,
                        n_runs: int = N_RANDOM_RUNS) -> dict[str, list[dict]]:
    """Returns {strategy_name: [metric_dict_per_run]}."""
    print(f"\nRunning {n_runs} random-regime simulations …")
    all_runs = {f"naive_{m}_{l}": []
                for l in TOP_L_LIST for m in POSITION_METHODS}

    for seed in range(n_runs):
        run_res = run_backtest(
            etf_returns, next_probs, regime_labels,
            use_random_regimes=True, random_seed=seed,
        )
        for name, series in run_res.items():
            all_runs[name].append(performance_metrics(series))

    return all_runs


# ============================================================
# 12. Table 3 Panel A — Nemenyi test (correct implementation)
# ============================================================

def _sig_stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def compute_table3(random_runs: dict,
                   treatment_results: dict) -> pd.DataFrame:
    """
    Table 3 Panel A: Naive Random vs not Random Regimes.

    For each of the 12 portfolio variants (naive_lo_2/3/4, naive_lns_2/3/4,
    naive_los_2/3/4, naive_mx_2/3/4):
      * control[s]   = mean metric across N_RANDOM_RUNS random seeds
      * treatment[s] = metric from non-random backtest

    Per-variant rank: 1=worse, 2=better.
    Control Rank  = mean of control ranks across all variants.
    Treatment Rank = mean of treatment ranks across all variants.
    Range: [1.0, 2.0] matching paper Table 3 format.

    p-value: one-sided paired t-test on per-variant (treatment - control) diffs.
    H0: treatment <= control.  Small p -> treatment IS better.
    """
    metrics = {
        "sharpe"       : "Sharpe",
        "sortino"      : "Sortino",
        "max_drawdown" : "MaxDD",
        "pct_positive" : "% Positive Ret.",
    }
    higher_better = {"sharpe", "sortino", "pct_positive"}
    strategy_names = sorted(treatment_results.keys())

    rows = []
    for metric, label in metrics.items():

        ctrl_means = []
        trt_vals   = []

        for name in strategy_names:
            if name not in random_runs:
                continue
            ctrl = [m.get(metric, np.nan) for m in random_runs[name]
                    if not np.isnan(m.get(metric, np.nan))]
            trt  = treatment_results[name].get(metric, np.nan)
            if len(ctrl) < 1 or np.isnan(trt):
                continue
            ctrl_means.append(float(np.mean(ctrl)))
            trt_vals.append(float(trt))

        if len(ctrl_means) < 2:
            rows.append({"Metric": label, "Control Rank": np.nan,
                         "Treatment Rank": np.nan, "p-value": np.nan,
                         "p-value-full": np.nan, "sig": ""})
            continue

        ctrl_arr = np.array(ctrl_means)
        trt_arr  = np.array(trt_vals)

        # Per-variant ranks: 1=worse, 2=better
        ctrl_ranks = []
        trt_ranks  = []
        for c, t in zip(ctrl_arr, trt_arr):
            if metric in higher_better:
                if c > t:
                    ctrl_ranks.append(2); trt_ranks.append(1)
                elif c < t:
                    ctrl_ranks.append(1); trt_ranks.append(2)
                else:
                    ctrl_ranks.append(1.5); trt_ranks.append(1.5)
            else:
                if c < t:
                    ctrl_ranks.append(2); trt_ranks.append(1)
                elif c > t:
                    ctrl_ranks.append(1); trt_ranks.append(2)
                else:
                    ctrl_ranks.append(1.5); trt_ranks.append(1.5)

        ctrl_rank = round(float(np.mean(ctrl_ranks)), 3)
        trt_rank  = round(float(np.mean(trt_ranks)),  3)

        # One-sided paired t-test
        diff = trt_arr - ctrl_arr
        if metric not in higher_better:
            diff = -diff
        t_stat, p_two = stats.ttest_1samp(diff, popmean=0)
        p_one = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)

        rows.append({
            "Metric"         : label,
            "Control Rank"   : ctrl_rank,
            "Treatment Rank" : trt_rank,
            "p-value"        : round(p_one, 3),
            "p-value-full"   : float(f"{p_one:.8f}"),
            "sig"            : _sig_stars(p_one),
        })

    return pd.DataFrame(rows).set_index("Metric")


# ============================================================
# 13. Figure 7 — Boxplots (control vs treatment)
# ============================================================

def plot_figure7(random_runs: dict[str, list[dict]],
                 treatment_results: dict[str, dict],
                 table3: pd.DataFrame):
    """
    Figure 7: 2x2 boxplots.
    Each box pools all strategies:
      Control   = N_RANDOM_RUNS x n_strategies metric values
      Treatment = n_strategies metric values (non-random)
    p-values from Table 3 annotated below each subplot.
    """
    metrics = {
        "sharpe"       : "Sharpe",
        "sortino"      : "Sortino",
        "max_drawdown" : "MaxDD",
        "pct_positive" : "% Positive Ret.",
    }

    ctrl_vals = {m: [] for m in metrics}
    trt_vals  = {m: [] for m in metrics}

    for name in random_runs:
        for run_m in random_runs[name]:
            for m in metrics:
                v = run_m.get(m, np.nan)
                if not np.isnan(v):
                    ctrl_vals[m].append(v)

    for name in treatment_results:
        for m in metrics:
            v = treatment_results[name].get(m, np.nan)
            if not np.isnan(v):
                trt_vals[m].append(v)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics.items()):
        ctrl = ctrl_vals[metric]
        trt  = trt_vals[metric]

        ax.boxplot(
            [ctrl, trt],
            labels=["Control", "Treatment"],
            patch_artist=True,
            medianprops  ={"color": "steelblue", "linewidth": 2},
            boxprops     ={"facecolor": "white",  "color": "black"},
            whiskerprops ={"color": "black"},
            capprops     ={"color": "black"},
            flierprops   ={"marker": "o", "markersize": 3,
                           "markerfacecolor": "grey", "alpha": 0.4},
        )

        # Annotate with p-value from Table 3
        try:
            p_val = table3.loc[label, "p-value"]
            sig   = table3.loc[label, "sig"]
            p_str = f"p-value: {p_val:.3f} (H\u2080: \u03bc_c > \u03bc_t){sig}"
        except Exception:
            p_str = ""

        ax.set_title(label, fontsize=11)
        ax.text(0.5, 0.02, p_str, transform=ax.transAxes,
                ha="center", fontsize=8, color="dimgrey")

    fig.suptitle(
        "Naive with Random Regime (control) vs not Random Regime (treatment)",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUTPUTS_DIR / "figure7_naive_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ============================================================
# 14. Figure 11 — Naive vs Benchmarks
# ============================================================

def plot_figure11(naive_scaled: dict[str, pd.Series],
                  spy_scaled: pd.Series,
                  ew_scaled: pd.Series,
                  mvo_scaled: pd.Series):
    """
    Figure 11: cumulative log returns with 10% vol scaling.
    Naive strategies in blue, EW green, SPY black, MVO red.
    """
    # Common index across all series
    all_series = list(naive_scaled.values()) + [spy_scaled, ew_scaled, mvo_scaled]
    common     = all_series[0].index
    for s in all_series[1:]:
        common = common.intersection(s.index)
    common = common.sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Naive strategies (blue, thin)
    for name, series in naive_scaled.items():
        cum = series.reindex(common).cumsum()
        ax.plot(cum.index, cum.values,
                color="steelblue", linewidth=0.8, alpha=0.65)

    # Benchmarks (thick)
    for series, color, label, lw in [
        (ew_scaled,  "green",  "EW",  2.0),
        (spy_scaled, "black",  "SPY", 2.0),
        (mvo_scaled, "red",    "MVO", 2.0),
    ]:
        cum = series.reindex(common).cumsum()
        ax.plot(cum.index, cum.values,
                color=color, label=label, linewidth=lw, zorder=5)

    # Legend
    naive_line = Line2D([0], [0], color="steelblue", lw=1.5, label="Naive")
    bench_lines = [
        Line2D([0], [0], color="green", lw=2.0, label="EW"),
        Line2D([0], [0], color="black", lw=2.0, label="SPY"),
        Line2D([0], [0], color="red",   lw=2.0, label="MVO"),
    ]
    ax.legend(handles=[naive_line] + bench_lines,
              loc="upper left", fontsize=9)

    ax.set_title("Naive Regime Portfolio (naive) vs Benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Returns (Vol. Target of 10%)")
    ax.axhline(0, color="grey", lw=0.5, ls="--")

    fig.tight_layout()
    out = OUTPUTS_DIR / "figure11_naive_vs_benchmarks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ============================================================
# 15. Table 4 — Performance metrics
# ============================================================

def build_table4(treatment_series: dict[str, pd.Series],
                 spy_scaled: pd.Series,
                 ew_scaled: pd.Series,
                 mvo_scaled: pd.Series) -> pd.DataFrame:
    """Build Table 4 matching paper format."""
    all_series = {**treatment_series,
                  "spy": spy_scaled,
                  "ew":  ew_scaled,
                  "mvo_lo": mvo_scaled}

    rows = []
    for name, series in all_series.items():
        m       = performance_metrics(series)
        m["model"] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index("model")
    df = df[["sharpe", "sortino", "max_drawdown", "pct_positive"]].copy()
    df.columns = ["Sharpe", "Sortino", "MaxDD", "% Positive Ret."]
    return df


# ============================================================
# 16. Main
# ============================================================

def main():
    print("=" * 70)
    print("NAIVE REGIME-CONDITIONED BACKTEST")
    print("=" * 70)

    etf_returns, next_probs, regime_labels = load_inputs()

    # ── Treatment backtest (non-random regimes) ────────────────────────
    print("\nRunning treatment backtest (non-random regimes) …")
    treatment_series = run_backtest(
        etf_returns, next_probs, regime_labels, use_random_regimes=False
    )

    # ── Benchmarks ────────────────────────────────────────────────────
    print("\nComputing benchmarks …")
    predicted_shifted = get_predicted_regimes(next_probs)
    common_idx        = build_common_index(etf_returns, predicted_shifted, regime_labels)

    spy_raw    = etf_returns["SPY"].reindex(common_idx).dropna()
    ew_raw     = etf_returns.reindex(common_idx).mean(axis=1).dropna()
    spy_scaled = apply_vol_scaling(spy_raw)
    ew_scaled  = apply_vol_scaling(ew_raw)

    print("Running MVO …")
    mvo_scaled = run_mvo_backtest(etf_returns, common_idx)

    # ── Performance metrics ────────────────────────────────────────────
    treatment_results = {name: performance_metrics(s)
                         for name, s in treatment_series.items()}

    # ── Table 4 ───────────────────────────────────────────────────────
    table4 = build_table4(treatment_series, spy_scaled, ew_scaled, mvo_scaled)
    print("\n" + "=" * 70)
    print("TABLE 4 — Performance metrics (vol-scaled 10%)")
    print("=" * 70)
    print(table4.to_string())
    table4.to_csv(OUTPUTS_DIR / "table4_naive_performance.csv")

    # ── Random regime baseline (control) ──────────────────────────────
    print("\n" + "=" * 70)
    print("RANDOM REGIME BASELINE")
    print("=" * 70)
    random_runs = run_random_baseline(etf_returns, next_probs, regime_labels)

    # ── Table 3 ───────────────────────────────────────────────────────
    table3 = compute_table3(random_runs, treatment_results)
    print("\n" + "=" * 70)
    print("TABLE 3 — Nemenyi Test (Panel A)")
    print("=" * 70)
    # Show both rounded and full-precision p-values
    display_cols = ["Control Rank", "Treatment Rank", "p-value", "sig"]
    print(table3[display_cols].to_string())
    print("\nFull-precision p-values:")
    for idx in table3.index:
        pf = table3.loc[idx, "p-value-full"]
        print(f"  {idx:<20}: {pf:.8f}  {table3.loc[idx,'sig']}")
    table3.to_csv(OUTPUTS_DIR / "table3_nemenyi_naive.csv")

    # ── Figures ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_figure7(random_runs, treatment_results, table3)
    plot_figure11(treatment_series, spy_scaled, ew_scaled, mvo_scaled)

    # ── Save returns ──────────────────────────────────────────────────
    pd.DataFrame(treatment_series).to_csv(
        PROCESSED_DIR / "naive_strategy_returns_paper_like.csv"
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("\nTop 5 Naive strategies by Sharpe:")
    top5 = (table4["Sharpe"]
            .drop(["spy", "ew", "mvo_lo"], errors="ignore")
            .sort_values(ascending=False).head(5))
    print(top5.round(3).to_string())
    print("\nOutputs: outputs/")
    for f in ["table4_naive_performance.csv", "table3_nemenyi_naive.csv",
              "figure7_naive_boxplot.png", "figure11_naive_vs_benchmarks.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()