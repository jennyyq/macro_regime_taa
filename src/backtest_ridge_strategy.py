"""
backtest_ridge_strategy.py
==========================
Linear Ridge Regression regime-conditioned strategy
(Paper Section 5.2.3, Eqs. 12-14).

Key formulas:
  Eq.(12): y_hat^{i,j,ridge}_{t+1} = beta_hat^{i,j} @ X^i_{1:t}
           Per-regime ridge model predicting ETF j return from macro PCA features.
  Eq.(13): beta_hat^{i,j} = argmin ||y^j - X^i beta||^2 + lambda ||beta||^2
  Eq.(14): y_hat^{j,ridge}_{t+1} = sum_i p_tilde_{i,t+1} * y_hat^{i,j,ridge}_{t+1}
           Probability-weighted aggregation across all regimes.

X = FRED-MD macro PCA features  (NOT ETF returns — key difference from Naive/BL)
y = ETF log returns

Control  = Ridge with random regimes
Treatment = Ridge with non-random regimes

Produces:
  Table 6  : Ridge + MVO performance metrics
  Table 3  : Nemenyi test Panel C (Ridge Random vs not Random)
  Figure 9 : boxplots Ridge (control=random) vs Ridge (treatment=non-random)
  Figure 13: cumulative log returns Ridge vs Benchmarks
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
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ETF_RETURNS_PATH       = PROCESSED_DIR / "etf_monthly_returns_paper_like.csv"
MACRO_PCA_PATH         = PROCESSED_DIR / "macro_pca_paper_like.csv"
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
ESTIMATION_WINDOW = 48        # paper Section 6: 48-month window
VOL_TARGET        = 0.10      # 10% annualised
VOL_LOOKBACK      = 12
VOL_SCALE_CAP     = 5.0
ANNUALIZATION     = 12
N_RANDOM_RUNS     = 50
RIDGE_ALPHA       = 1.0       # lambda in Eq.(13); paper does not specify exact value
N_TOTAL_REGIMES   = 6         # Regimes 0-5


# ============================================================
# 3. Load data
# ============================================================

def load_inputs():
    print("=" * 70)
    print("LOADING INPUT DATA")
    print("=" * 70)

    for p in [ETF_RETURNS_PATH, MACRO_PCA_PATH,
              NEXT_REGIME_PROBS_PATH, REGIME_LABELS_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\nRun preprocess_data.py -> detect_regimes.py first."
            )

    etf_returns   = pd.read_csv(
        ETF_RETURNS_PATH, index_col=0, parse_dates=True)[ETF_TICKERS]
    macro_pca     = pd.read_csv(
        MACRO_PCA_PATH, index_col=0, parse_dates=True)
    next_probs    = pd.read_csv(
        NEXT_REGIME_PROBS_PATH, index_col=0, parse_dates=True)
    regime_labels = pd.read_csv(
        REGIME_LABELS_PATH, index_col=0, parse_dates=True)["regime"].astype(int)

    print(f"ETF returns   : {etf_returns.shape}  "
          f"{etf_returns.index.min().date()} -> {etf_returns.index.max().date()}")
    print(f"Macro PCA     : {macro_pca.shape}  "
          f"{macro_pca.index.min().date()} -> {macro_pca.index.max().date()}")
    print(f"Next probs    : {next_probs.shape}  "
          f"{next_probs.index.min().date()} -> {next_probs.index.max().date()}")
    print(f"Regime labels : {regime_labels.shape}  "
          f"{regime_labels.index.min().date()} -> {regime_labels.index.max().date()}")

    return etf_returns, macro_pca, next_probs, regime_labels


# ============================================================
# 4. Timing alignment
# ============================================================

def get_next_probs_aligned(next_probs: pd.DataFrame) -> pd.DataFrame:
    """
    next_probs[t] = p_tilde_{t+1} (regime forecast for t+1, made at t).
    Shift index +1 month so that at position date t+1, we use the
    forecast made at t for t+1.

    Paper Eq.(14): y_hat^{j,ridge}_{t+1} = sum_i p_tilde_{i,t+1} * y_hat^{i,j}_{t+1}
    The p_tilde_{i,t+1} used at position_date t+1 is next_probs at time t.
    """
    shifted       = next_probs.copy()
    shifted.index = shifted.index + pd.DateOffset(months=1)
    return shifted


def get_predicted_regime_shifted(next_probs: pd.DataFrame) -> pd.Series:
    """Argmax of next_probs, shifted +1 month for position timing."""
    col0   = next_probs.columns[0]
    prefix = col0.rstrip("0123456789").rstrip("_")
    predicted = (
        next_probs.idxmax(axis=1)
        .str.replace(prefix + "_", "", regex=False)
        .str.replace(prefix, "", regex=False)
        .astype(int)
    )
    shifted       = predicted.copy()
    shifted.index = shifted.index + pd.DateOffset(months=1)
    return shifted


def build_common_index(etf_returns, macro_pca, probs_shifted, regime_labels):
    common = (
        etf_returns.index
        .intersection(macro_pca.index)
        .intersection(probs_shifted.index)
        .intersection(regime_labels.index)
    ).sort_values()
    warmup_end = etf_returns.index.min() + pd.DateOffset(months=ESTIMATION_WINDOW)
    common     = common[common >= warmup_end]
    print(f"\nBacktest window: {common.min().date()} -> {common.max().date()}  "
          f"({len(common)} months)")
    return common


# ============================================================
# 5. Ridge forecast — Paper Eqs.(12-14)
# ============================================================

def compute_ridge_forecast(
    position_date: pd.Timestamp,
    prob_vector: np.ndarray,        # p_tilde_{t+1}, shape (N_TOTAL_REGIMES,)
    etf_returns: pd.DataFrame,
    macro_pca: pd.DataFrame,
    regime_labels: pd.Series,
    alpha: float = RIDGE_ALPHA,
) -> "pd.Series | None":
    """
    Paper Eqs.(12-14):

    For each regime i in {0, ..., 5}:
      X^i = macro PCA features from months in window where regime == i
      y^j_i = ETF j log returns from those same months (one-step-ahead aligned)

      Eq.(13): fit Ridge(alpha) on (X^i_{t-1}, y^j_i_t) pairs
               (X at t-1 predicts return at t — one-step-ahead)
      Eq.(12): y_hat^{i,j}_{t+1} = beta^{i,j} @ x_current
               where x_current = macro PCA at position_date

    Eq.(14): y_hat^{j,ridge}_{t+1} = sum_i p_tilde_{i,t+1} * y_hat^{i,j}_{t+1}

    Notes:
      - X = macro PCA features  (NOT ETF returns — key per paper Section 5.2.3)
      - One-step-ahead: X at month t predicts return at month t+1
      - If regime i has < 3 months in window, skip (not enough for ridge)
      - Final forecast is probability-weighted across all regimes
    """
    # Historical data up to (but not including) position_date
    hist_ret = etf_returns.loc[etf_returns.index < position_date]
    hist_mac = macro_pca.loc[macro_pca.index < position_date]
    hist_reg = regime_labels.loc[regime_labels.index < position_date]

    if len(hist_ret) < ESTIMATION_WINDOW:
        return None

    # 48-month estimation window
    window_ret = hist_ret.iloc[-ESTIMATION_WINDOW:]
    window_mac = hist_mac.reindex(window_ret.index)
    window_reg = hist_reg.reindex(window_ret.index)

    # Current macro state (x at position_date) — used for prediction
    if position_date not in macro_pca.index:
        return None
    x_current = macro_pca.loc[position_date].values.reshape(1, -1)

    n_etf = len(ETF_TICKERS)
    regime_forecasts = np.full((N_TOTAL_REGIMES, n_etf), np.nan)

    for i in range(N_TOTAL_REGIMES):
        # Months in window assigned to regime i
        regime_mask = window_reg == i
        regime_dates = window_reg[regime_mask].index

        if len(regime_dates) < 3:
            # Not enough data for this regime — skip
            continue

        # One-step-ahead alignment:
        # X at month t (regime_dates[:-1]) predicts return at month t+1 (regime_dates[1:])
        # We need consecutive pairs where both X_t and y_{t+1} are available
        X_pairs = []
        y_pairs = []

        for d in regime_dates:
            # Get the next month's return
            next_month_idx = window_ret.index.get_loc(d)
            if next_month_idx + 1 >= len(window_ret):
                continue   # no next month in window
            next_date = window_ret.index[next_month_idx + 1]

            x_t  = window_mac.loc[d].values      # macro features at t
            y_t1 = window_ret.loc[next_date].values  # returns at t+1

            if np.any(np.isnan(x_t)) or np.any(np.isnan(y_t1)):
                continue

            X_pairs.append(x_t)
            y_pairs.append(y_t1)

        if len(X_pairs) < 3:
            continue

        X_train = np.array(X_pairs)   # (n_pairs, n_features)
        y_train = np.array(y_pairs)   # (n_pairs, n_etf)

        # Eq.(13): fit ridge regression for each ETF
        try:
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_train, y_train)
            # Eq.(12): predict with current macro state
            pred = ridge.predict(x_current)[0]   # shape (n_etf,)
            regime_forecasts[i] = pred
        except Exception:
            continue

    # Eq.(14): probability-weighted aggregation
    # p_tilde_{i,t+1} weights each regime's forecast
    weighted_forecast = np.zeros(n_etf)
    total_weight      = 0.0

    for i in range(N_TOTAL_REGIMES):
        if not np.any(np.isnan(regime_forecasts[i])):
            w = float(prob_vector[i])
            weighted_forecast += w * regime_forecasts[i]
            total_weight      += w

    if total_weight < 1e-8:
        return None

    # Renormalise if some regimes were skipped
    weighted_forecast /= total_weight

    return pd.Series(weighted_forecast, index=ETF_TICKERS)


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
    """Eq.(18) — long-only: top-l by forecast value, positive only."""
    valid = forecast.dropna().sort_values(ascending=False)
    H     = valid.head(l)
    H_pos = H[H > 0]
    if H_pos.empty:
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


def position_mx(forecast: pd.Series, l: int, predicted_regime: int) -> pd.Series:
    """Mixed: los if Regime 0 (crisis), lo otherwise."""
    if predicted_regime == 0:
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
# 7. Volatility scaling
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


# ============================================================
# 8. MVO benchmark
# ============================================================

def mvo_weights_longonly(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mu)
    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret / vol) if vol > 1e-8 else 0.0
    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-9, "maxiter": 500})
    return res.x if res.success else np.ones(n) / n


def run_mvo_backtest(etf_returns: pd.DataFrame,
                     common_index: pd.DatetimeIndex) -> pd.Series:
    raw = []
    for date in common_index:
        hist = etf_returns.loc[etf_returns.index < date].iloc[-ESTIMATION_WINDOW:]
        if len(hist) < ESTIMATION_WINDOW:
            raw.append(np.nan); continue
        mu  = hist.mean().values
        cov = hist.cov().values + np.eye(len(ETF_TICKERS)) * 1e-6
        w   = mvo_weights_longonly(mu, cov)
        raw.append(float(etf_returns.loc[date].values @ w))
    return apply_vol_scaling(pd.Series(raw, index=common_index).dropna())


# ============================================================
# 9. Performance metrics
# ============================================================

def performance_metrics(log_returns: pd.Series) -> dict:
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
# 10. Main Ridge backtest loop
# ============================================================

def run_ridge_backtest(
    etf_returns, macro_pca, next_probs, regime_labels,
    use_random_regimes: bool = False,
    random_seed: int = 42,
) -> dict[str, pd.Series]:
    """Returns {strategy_name: vol_scaled_log_returns}."""

    probs_shifted   = get_next_probs_aligned(next_probs)
    pred_shifted    = get_predicted_regime_shifted(next_probs)
    common          = build_common_index(
        etf_returns, macro_pca, probs_shifted, regime_labels
    )

    pred_series  = pred_shifted.reindex(common).dropna().astype(int)
    probs_series = probs_shifted.reindex(common).dropna()

    if use_random_regimes:
        rng        = np.random.default_rng(random_seed)
        unique_reg = np.array(sorted(regime_labels.unique()))
        # Randomise predicted regime (for argmax sizing)
        pred_series = pd.Series(
            rng.choice(unique_reg, size=len(pred_series)),
            index=pred_series.index,
        )
        # Randomise probability vectors uniformly
        rand_probs = rng.dirichlet(
            np.ones(N_TOTAL_REGIMES), size=len(probs_series)
        )
        probs_series = pd.DataFrame(
            rand_probs,
            index=probs_series.index,
            columns=probs_series.columns,
        )

    results = {}

    for l in TOP_L_LIST:
        for method in POSITION_METHODS:
            name     = f"ridge_{method}_{l}"
            raw_list = []
            print(f"  Backtesting {name} …")

            for date in pred_series.index:
                pred_regime = int(pred_series.loc[date])

                # p_tilde_{t+1}: probability vector for this position date
                prob_vec = probs_series.loc[date].values.astype(float)
                # Renormalise (Eq.6)
                prob_sum = prob_vec.sum()
                if prob_sum > 1e-8:
                    prob_vec = prob_vec / prob_sum
                else:
                    prob_vec = np.ones(N_TOTAL_REGIMES) / N_TOTAL_REGIMES

                # Ridge forecast: Eqs.(12-14)
                forecast = compute_ridge_forecast(
                    position_date = date,
                    prob_vector   = prob_vec,
                    etf_returns   = etf_returns,
                    macro_pca     = macro_pca,
                    regime_labels = regime_labels,
                    alpha         = RIDGE_ALPHA,
                )

                weights  = get_weights(forecast, l, method, pred_regime)
                port_ret = float((weights * etf_returns.loc[date]).sum())
                raw_list.append(port_ret)

            raw    = pd.Series(raw_list, index=pred_series.index, name=name)
            scaled = apply_vol_scaling(raw)
            results[name] = scaled
            m = performance_metrics(scaled)
            print(f"    Sharpe={m['sharpe']:.3f}  Sortino={m['sortino']:.3f}  "
                  f"MaxDD={m['max_drawdown']:.3f}  %Pos={m['pct_positive']:.3f}")

    return results


# ============================================================
# 11. Random Ridge baseline
# ============================================================

def run_random_baseline(etf_returns, macro_pca, next_probs, regime_labels,
                        n_runs: int = N_RANDOM_RUNS) -> dict[str, list[dict]]:
    print(f"\nRunning {n_runs} random-regime Ridge simulations …")
    all_runs = {f"ridge_{m}_{l}": []
                for l in TOP_L_LIST for m in POSITION_METHODS}
    for seed in range(n_runs):
        run_res = run_ridge_backtest(
            etf_returns, macro_pca, next_probs, regime_labels,
            use_random_regimes=True, random_seed=seed,
        )
        for name, series in run_res.items():
            all_runs[name].append(performance_metrics(series))
        if (seed + 1) % 10 == 0:
            print(f"  Completed {seed + 1}/{n_runs} runs")
    return all_runs


# ============================================================
# 12. Table 3 Panel C — Nemenyi test
# ============================================================

def _sig_stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def compute_table3_panel_c(random_runs: dict,
                            treatment_results: dict) -> pd.DataFrame:
    """
    Table 3 Panel C: Ridge Random vs not Random Regimes.

    Per-variant rank: 1=worse, 2=better.
    Control Rank  = mean rank of control (random regime) across variants.
    Treatment Rank = mean rank of treatment (non-random) across variants.
    p-value: one-sided paired t-test on (treatment - control_mean) per variant.
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
        ctrl_means, trt_vals = [], []

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
        ctrl_ranks, trt_ranks = [], []
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
# 13. Figure 9 — Boxplots (Ridge random vs non-random)
# ============================================================

def plot_figure9(random_runs: dict, treatment_results: dict,
                 table3c: pd.DataFrame):
    """
    Figure 9: 2x2 boxplots.
    Control  = Ridge with random regimes.
    Treatment = Ridge with non-random regimes.
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
            boxprops     ={"facecolor": "white", "color": "black"},
            whiskerprops ={"color": "black"},
            capprops     ={"color": "black"},
            flierprops   ={"marker": "o", "markersize": 3,
                           "markerfacecolor": "grey", "alpha": 0.4},
        )

        try:
            p_val = table3c.loc[label, "p-value"]
            sig   = table3c.loc[label, "sig"]
            p_str = f"p-value: {p_val:.3f} (H\u2080: \u03bc_c > \u03bc_t){sig}"
        except Exception:
            p_str = ""

        ax.set_title(label, fontsize=11)
        ax.text(0.5, 0.02, p_str, transform=ax.transAxes,
                ha="center", fontsize=8, color="dimgrey")

    fig.suptitle(
        "Ridge with Random Regime (control) vs not Random Regime (treatment)",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUTPUTS_DIR / "figure9_ridge_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ============================================================
# 14. Figure 13 — Ridge vs Benchmarks
# ============================================================

def plot_figure13(ridge_scaled: dict[str, pd.Series],
                  mvo_scaled: pd.Series,
                  spy_scaled: pd.Series,
                  ew_scaled: pd.Series):
    """
    Figure 13: Ridge strategies (blue/cyan) vs EW, SPY (black/dark),
    MVO (red). Cumulative log returns, 10% vol scaling.
    """
    all_series = (list(ridge_scaled.values()) +
                  [mvo_scaled, spy_scaled, ew_scaled])
    common = all_series[0].index
    for s in all_series[1:]:
        common = common.intersection(s.index)
    common = common.sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Ridge strategies (cyan/blue — paper says blue)
    for name, series in ridge_scaled.items():
        cum = series.reindex(common).cumsum()
        ax.plot(cum.index, cum.values,
                color="steelblue", linewidth=0.8, alpha=0.65)

    # MVO (red)
    cum_mvo = mvo_scaled.reindex(common).cumsum()
    ax.plot(cum_mvo.index, cum_mvo.values,
            color="red", label="MVO", linewidth=1.8, zorder=4)

    # Benchmarks (thick dark lines)
    for series, color, label, lw in [
        (ew_scaled,  "green", "EW",  2.0),
        (spy_scaled, "black", "SPY", 2.0),
    ]:
        cum = series.reindex(common).cumsum()
        ax.plot(cum.index, cum.values,
                color=color, label=label, linewidth=lw, zorder=5)

    # Legend
    ridge_line = Line2D([0], [0], color="steelblue", lw=1.5, label="Ridge")
    mvo_line   = Line2D([0], [0], color="red",       lw=1.8, label="MVO")
    ew_line    = Line2D([0], [0], color="green",     lw=2.0, label="EW")
    spy_line   = Line2D([0], [0], color="black",     lw=2.0, label="SPY")
    ax.legend(handles=[ew_line, spy_line, ridge_line, mvo_line],
              loc="upper left", fontsize=9)

    ax.set_title("Ridge Regime Portfolio (ridge) vs Benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Returns (Vol. Target of 10%)")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    fig.tight_layout()
    out = OUTPUTS_DIR / "figure13_ridge_vs_benchmarks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ============================================================
# 15. Table 6 — Performance metrics
# ============================================================

def build_table6(ridge_series: dict[str, pd.Series],
                 mvo_scaled: pd.Series,
                 spy_scaled: pd.Series,
                 ew_scaled: pd.Series) -> pd.DataFrame:
    """Build Table 6 in paper format (Ridge + MVO + benchmarks)."""
    all_series = {**ridge_series,
                  "mvo_lo": mvo_scaled,
                  "spy":    spy_scaled,
                  "ew":     ew_scaled}
    rows = []
    for name, series in all_series.items():
        m = performance_metrics(series)
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
    print("LINEAR RIDGE REGRESSION REGIME-CONDITIONED BACKTEST")
    print("=" * 70)

    etf_returns, macro_pca, next_probs, regime_labels = load_inputs()

    # ── Treatment: Ridge with non-random regimes ───────────────────────
    print("\nRunning Ridge treatment backtest (non-random regimes) …")
    ridge_treatment_series = run_ridge_backtest(
        etf_returns, macro_pca, next_probs, regime_labels,
        use_random_regimes=False,
    )

    # ── Common index for benchmarks ───────────────────────────────────
    probs_shifted = get_next_probs_aligned(next_probs)
    common_idx    = build_common_index(
        etf_returns, macro_pca, probs_shifted, regime_labels
    )

    # ── MVO and market benchmarks ─────────────────────────────────────
    print("\nRunning MVO benchmark …")
    mvo_scaled = run_mvo_backtest(etf_returns, common_idx)
    spy_scaled = apply_vol_scaling(
        etf_returns["SPY"].reindex(common_idx).dropna()
    )
    ew_scaled  = apply_vol_scaling(
        etf_returns.reindex(common_idx).mean(axis=1).dropna()
    )

    # ── Performance metrics ────────────────────────────────────────────
    treatment_results = {n: performance_metrics(s)
                         for n, s in ridge_treatment_series.items()}

    # ── Table 6 ───────────────────────────────────────────────────────
    table6 = build_table6(ridge_treatment_series, mvo_scaled,
                          spy_scaled, ew_scaled)
    print("\n" + "=" * 70)
    print("TABLE 6 — Performance metrics (Ridge + MVO, vol-scaled 10%)")
    print("=" * 70)
    print(table6.to_string())
    table6.to_csv(OUTPUTS_DIR / "table6_ridge_performance.csv")
    print(f"\nSaved: table6_ridge_performance.csv")

    # ── Random Ridge baseline ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RANDOM REGIME BASELINE (Ridge control)")
    print("=" * 70)
    ridge_random_runs = run_random_baseline(
        etf_returns, macro_pca, next_probs, regime_labels
    )

    # ── Table 3 Panel C ───────────────────────────────────────────────
    table3c = compute_table3_panel_c(ridge_random_runs, treatment_results)
    print("\n" + "=" * 70)
    print("TABLE 3 Panel C — Nemenyi Test (Ridge Random vs not Random)")
    print("=" * 70)
    display_cols = ["Control Rank", "Treatment Rank", "p-value", "sig"]
    print(table3c[display_cols].to_string())
    print("\nFull-precision p-values:")
    for idx in table3c.index:
        pf  = table3c.loc[idx, "p-value-full"]
        sig = table3c.loc[idx, "sig"]
        print(f"  {idx:<20}: {pf:.8f}  {sig}")
    table3c.to_csv(OUTPUTS_DIR / "table3c_nemenyi_ridge.csv")
    print(f"\nSaved: table3c_nemenyi_ridge.csv")

    # ── Figures ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_figure9(ridge_random_runs, treatment_results, table3c)
    plot_figure13(ridge_treatment_series, mvo_scaled, spy_scaled, ew_scaled)

    # ── Save returns ──────────────────────────────────────────────────
    pd.DataFrame(ridge_treatment_series).to_csv(
        PROCESSED_DIR / "ridge_strategy_returns_paper_like.csv"
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("\nTop 5 Ridge strategies by Sharpe:")
    top5 = (table6["Sharpe"]
            .filter(like="ridge_")
            .sort_values(ascending=False).head(5))
    print(top5.round(3).to_string())
    print("\nOutputs saved to outputs/:")
    for f in ["table6_ridge_performance.csv",
              "table3c_nemenyi_ridge.csv",
              "figure9_ridge_boxplot.png",
              "figure13_ridge_vs_benchmarks.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()