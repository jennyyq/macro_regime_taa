"""
backtest_bl_strategy.py
=======================
Black-Litterman regime-conditioned strategy  (Paper Section 5.2.2, 5.3.2).

Paper Eq.(11): view on ETF j = regime-conditional sample mean return
    q*_{j,t+1} = E[r^j_{t+1} | Regime = i*_{t+1}] = mu_hat*_{j,1:t}

Paper Eq.(19): BL posterior weights
    w_{t+1} = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1}
              [(τΣ)^{-1} μ̂_{1:t} + P^T Ω^{-1} q*_{t+1}]

Control  = MVO (mean-variance optimisation, no regime views)
Treatment = BL with regime-conditional views

Produces:
  Table 5  : BL + MVO performance metrics
  Table 3  : Nemenyi test Panel B (BL Random vs not Random)
  Figure 8 : boxplots MVO (control) vs BL (treatment)
  Figure 12: cumulative log returns BL vs Benchmarks
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
POSITION_METHODS  = ["lo", "lns"]     # paper Table 5 shows bl_lo and bl_lns
ESTIMATION_WINDOW = 48
VOL_TARGET        = 0.10
VOL_LOOKBACK      = 12
VOL_SCALE_CAP     = 5.0
ANNUALIZATION     = 12
N_RANDOM_RUNS     = 50

TAU               = 0.05   # BL scaling parameter tau (uncertainty in prior)
# Omega = tau * Sigma (proportional to prior, standard BL specification)


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
                f"Missing: {p}\nRun preprocess_data.py -> detect_regimes.py first."
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
    Paper Eq.(8): i*_{t+1} = argmax p~_{i,t+1}
    Shift +1 month so signal at t -> position at t+1.
    """
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


def build_common_index(etf_returns, predicted_shifted, regime_labels):
    common = (
        etf_returns.index
        .intersection(predicted_shifted.index)
        .intersection(regime_labels.index)
    ).sort_values()
    warmup_end = etf_returns.index.min() + pd.DateOffset(months=ESTIMATION_WINDOW)
    common     = common[common >= warmup_end]
    print(f"\nBacktest window: {common.min().date()} -> {common.max().date()}  "
          f"({len(common)} months)")
    return common


# ============================================================
# 5. BL views — Paper Eq.(11)
# ============================================================

def compute_bl_views(
    position_date: pd.Timestamp,
    predicted_regime: int,
    etf_returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> "pd.Series | None":
    """
    Paper Eq.(11):
        q*_{j,t+1} = mu_hat*_{j,1:t}
        = sample mean return of ETF j on months in the 48-month window
          whose hard regime label == predicted_regime.

    This is the "view" in the BL framework — the regime-conditional
    expected return replaces the market equilibrium return as the view.

    Falls back to full-window mean if < 2 same-regime months.
    """
    hist_ret = etf_returns.loc[etf_returns.index < position_date]
    hist_reg = regime_labels.loc[regime_labels.index < position_date]

    if len(hist_ret) < ESTIMATION_WINDOW:
        return None

    window_ret = hist_ret.iloc[-ESTIMATION_WINDOW:]
    window_reg = hist_reg.reindex(window_ret.index)

    same_idx = window_reg[window_reg == predicted_regime].index
    same_idx = same_idx.intersection(window_ret.index)

    if len(same_idx) < 2:
        same_idx = window_ret.index   # fallback to full window

    views = window_ret.loc[same_idx].mean()
    return views.reindex(ETF_TICKERS)


# ============================================================
# 6. BL posterior expected returns — Paper Eq.(19)
# ============================================================

def compute_bl_posterior_mu(
    mu_prior: np.ndarray,
    cov_prior: np.ndarray,
    views_q: np.ndarray,
    tau: float = TAU,
) -> np.ndarray:
    """
    Paper Eq.(19):
        mu_BL = [(tauSigma)^{-1} + P^T Omega^{-1} P]^{-1}
                [(tauSigma)^{-1} mu_hat + P^T Omega^{-1} q*]

    where:
        mu_hat  = sample mean (prior belief, Sec 5.2.2)
        Sigma   = sample covariance (prior belief, Sec 5.2.2)
        tau     = scaling parameter for prior uncertainty (TAU = 0.05)
        P       = identity matrix (one view per asset, Sec 5.3.2)
        q*      = regime-conditional expected returns (Eq.11, the "views")
        Omega   = tau * P * Sigma * P^T  (standard BL proportional uncertainty)
                  This ties view uncertainty to prior uncertainty, which is
                  the most common and principled BL Omega specification.

    Returns: mu_BL — posterior expected return vector (n,)
    The caller converts mu_BL into portfolio weights via position sizing.
    """
    n   = len(mu_prior)
    P   = np.eye(n)   # one view per asset

    # Standard BL: Omega proportional to prior covariance
    # Omega = tau * P * Sigma * P^T
    Omega = tau * P @ cov_prior @ P.T
    # Add small diagonal for numerical stability
    Omega += np.eye(n) * 1e-8

    tauSig = tau * cov_prior
    tauSig += np.eye(n) * 1e-8   # numerical stability

    try:
        tauSig_inv = np.linalg.solve(tauSig, np.eye(n))
        Omega_inv  = np.linalg.solve(Omega,  np.eye(n))

        M     = tauSig_inv + P.T @ Omega_inv @ P
        M_inv = np.linalg.solve(M, np.eye(n))
        rhs   = tauSig_inv @ mu_prior + P.T @ Omega_inv @ views_q
        mu_bl = M_inv @ rhs

    except np.linalg.LinAlgError:
        # Fallback: weighted average of prior and view
        mu_bl = 0.5 * mu_prior + 0.5 * views_q

    return mu_bl


# ============================================================
# 7. BL position sizing — Paper Sec 5.3.2, Eq.(19)
# ============================================================

def _normalize_weights(selected: pd.Series) -> pd.Series:
    """Proportional weights: w_j = |selected_j| / sum|selected|, with sign."""
    w = pd.Series(0.0, index=ETF_TICKERS)
    if selected.empty:
        return w
    S = selected.abs().sum()
    if S < 1e-12:
        return w
    w.loc[selected.index] = selected.values / S
    return w


def position_lo_bl(mu_bl: np.ndarray, l: int) -> pd.Series:
    """
    BL long-only (lo): top-l ETFs by posterior expected return,
    weighted proportionally to their posterior mean.
    Paper Sec 5.3.2: long-only constraint applied to BL posterior mu.
    """
    mu_s = pd.Series(mu_bl, index=ETF_TICKERS).sort_values(ascending=False)
    top  = mu_s.head(l)
    # Long-only: only positive expected returns
    top_pos = top[top > 0]
    if top_pos.empty:
        top_pos = top   # all negative: take top-l anyway (scale down per paper)
    return _normalize_weights(top_pos)


def position_lns_bl(mu_bl: np.ndarray, l: int) -> pd.Series:
    """
    BL long-and-short (lns): top-l long + bottom-l short by posterior mean.
    """
    mu_s = pd.Series(mu_bl, index=ETF_TICKERS).sort_values(ascending=False)
    H    = mu_s.head(l)
    L    = mu_s.tail(l)
    L    = L.loc[L.index.difference(H.index)]
    return _normalize_weights(pd.concat([H, L]))


def get_bl_weights(mu_bl: np.ndarray, l: int, method: str) -> pd.Series:
    """Dispatch to correct BL position sizing method."""
    if method == "lo":
        return position_lo_bl(mu_bl, l)
    elif method == "lns":
        return position_lns_bl(mu_bl, l)
    raise ValueError(f"Unknown BL method: {method}")


# ============================================================
# 8. MVO benchmark (same as in naive backtest)
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
                     common_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """
    MVO control strategies: mvo_lo_2/3/4 and mvo_lns_2/3/4.
    Paper Table 5 shows both long-only and long-short MVO variants.
    """
    print("Running MVO backtest …")
    raw_lo = []

    for date in common_index:
        hist = etf_returns.loc[etf_returns.index < date].iloc[-ESTIMATION_WINDOW:]
        if len(hist) < ESTIMATION_WINDOW:
            raw_lo.append(np.nan)
            continue
        mu  = hist.mean().values
        cov = hist.cov().values + np.eye(len(ETF_TICKERS)) * 1e-6
        w   = mvo_weights_longonly(mu, cov)
        raw_lo.append(float(etf_returns.loc[date].values @ w))

    raw_series = pd.Series(raw_lo, index=common_index).dropna()

    # For MVO variants with l, we use different subsets of ETFs
    # Paper shows mvo_lo_l and mvo_lns_l — here we generate them
    results = {}
    for l in TOP_L_LIST:
        # mvo_lo_l: long-only top-l by posterior mean
        ret_lo, ret_lns = [], []
        for date in common_index:
            hist = etf_returns.loc[etf_returns.index < date].iloc[-ESTIMATION_WINDOW:]
            if len(hist) < ESTIMATION_WINDOW:
                ret_lo.append(np.nan)
                ret_lns.append(np.nan)
                continue
            mu  = hist.mean().values
            cov = hist.cov().values + np.eye(len(ETF_TICKERS)) * 1e-6

            # lo: top-l by mean, long-only
            mu_s = pd.Series(mu, index=ETF_TICKERS).sort_values(ascending=False)
            top  = mu_s.head(l)
            top_pos = top[top > 0]
            if top_pos.empty:
                top_pos = top
            w_lo = pd.Series(0.0, index=ETF_TICKERS)
            S = top_pos.abs().sum()
            if S > 1e-12:
                w_lo.loc[top_pos.index] = top_pos.values / S
            ret_lo.append(float(etf_returns.loc[date].values @ w_lo.values))

            # lns: top-l long + bottom-l short by mean
            H = mu_s.head(l)
            L = mu_s.tail(l)
            L = L.loc[L.index.difference(H.index)]
            sel = pd.concat([H, L])
            w_lns = pd.Series(0.0, index=ETF_TICKERS)
            S2 = sel.abs().sum()
            if S2 > 1e-12:
                w_lns.loc[sel.index] = sel.values / S2
            ret_lns.append(float(etf_returns.loc[date].values @ w_lns.values))

        results[f"mvo_lo_{l}"]  = apply_vol_scaling(
            pd.Series(ret_lo,  index=common_index).dropna())
        results[f"mvo_lns_{l}"] = apply_vol_scaling(
            pd.Series(ret_lns, index=common_index).dropna())

    return results


# ============================================================
# 9. Volatility scaling
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
# 10. Performance metrics
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
# 11. Main BL backtest loop
# ============================================================

def run_bl_backtest(etf_returns, next_probs, regime_labels,
                    use_random_regimes: bool = False,
                    random_seed: int = 42) -> dict[str, pd.Series]:
    """Returns {strategy_name: vol_scaled_log_returns}."""
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
            name     = f"bl_{method}_{l}"
            raw_list = []

            for date in pred_series.index:
                pred_regime = int(pred_series.loc[date])

                # Historical data
                hist_ret = etf_returns.loc[etf_returns.index < date]
                if len(hist_ret) < ESTIMATION_WINDOW:
                    raw_list.append(np.nan)
                    continue
                window_ret = hist_ret.iloc[-ESTIMATION_WINDOW:]

                # Prior: sample mean and covariance
                mu_prior  = window_ret.mean().values
                cov_prior = window_ret.cov().values + np.eye(len(ETF_TICKERS)) * 1e-6

                # Views: Eq.(11) regime-conditional expected returns
                views = compute_bl_views(
                    date, pred_regime, etf_returns, regime_labels
                )
                if views is None:
                    raw_list.append(np.nan)
                    continue

                # BL posterior expected returns: Eq.(19)
                mu_bl = compute_bl_posterior_mu(
                    mu_prior, cov_prior, views.values, TAU
                )

                # Position sizing
                weights  = get_bl_weights(mu_bl, l, method)
                port_ret = float((weights * etf_returns.loc[date]).sum())
                raw_list.append(port_ret)

            raw    = pd.Series(raw_list, index=pred_series.index, name=name)
            scaled = apply_vol_scaling(raw)
            results[name] = scaled

    return results


# ============================================================
# 12. Random BL baseline
# ============================================================

def run_random_baseline(etf_returns, next_probs, regime_labels,
                        n_runs: int = N_RANDOM_RUNS) -> dict[str, list[dict]]:
    print(f"\nRunning {n_runs} random-regime BL simulations …")
    all_runs = {f"bl_{m}_{l}": []
                for l in TOP_L_LIST for m in POSITION_METHODS}
    for seed in range(n_runs):
        run_res = run_bl_backtest(
            etf_returns, next_probs, regime_labels,
            use_random_regimes=True, random_seed=seed,
        )
        for name, series in run_res.items():
            all_runs[name].append(performance_metrics(series))
    return all_runs


# ============================================================
# 13. Table 3 Panel B — Nemenyi test (correct implementation)
# ============================================================

def _sig_stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def compute_table3_panel_b(random_runs: dict,
                            treatment_results: dict) -> pd.DataFrame:
    """
    Table 3 Panel B: BL Random vs not Random Regimes.

    Paper method (re-reading Table 3 carefully):
      - For each of the 6 portfolio variants (bl_lo_2, bl_lo_3, bl_lo_4,
        bl_lns_2, bl_lns_3, bl_lns_4):
          * control[s]   = mean metric across N_RANDOM_RUNS random seeds
          * treatment[s] = metric from non-random BL backtest
      - For each variant s, rank {control[s], treatment[s]} as 1 or 2.
        In the paper: rank 1 = WORSE, rank 2 = BETTER (ascending rank).
        So for Sharpe (higher=better):
          if control[s] > treatment[s]: control rank=2, treatment rank=1
          else:                          control rank=1, treatment rank=2
      - Control Rank = mean of control ranks across all variants
      - Treatment Rank = mean of treatment ranks across all variants
      - p-value: one-sided t-test on the 6 per-variant (treatment - control) diffs
        H0: treatment <= control (treatment NOT better)

    This produces ranks in [1.0, 2.0] range matching paper exactly.
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

        ctrl_means = []   # one per strategy variant
        trt_vals   = []   # one per strategy variant

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
        n        = len(ctrl_arr)

        # ── Per-variant ranks: 1=worse, 2=better ──────────────────────
        ctrl_ranks = []
        trt_ranks  = []

        for c, t in zip(ctrl_arr, trt_arr):
            if metric in higher_better:
                # Higher is better
                if c > t:
                    ctrl_ranks.append(2); trt_ranks.append(1)
                elif c < t:
                    ctrl_ranks.append(1); trt_ranks.append(2)
                else:
                    ctrl_ranks.append(1.5); trt_ranks.append(1.5)
            else:
                # Lower is better (MaxDD)
                if c < t:
                    ctrl_ranks.append(2); trt_ranks.append(1)
                elif c > t:
                    ctrl_ranks.append(1); trt_ranks.append(2)
                else:
                    ctrl_ranks.append(1.5); trt_ranks.append(1.5)

        ctrl_rank = round(float(np.mean(ctrl_ranks)), 3)
        trt_rank  = round(float(np.mean(trt_ranks)),  3)

        # ── p-value: one-sided paired t-test ──────────────────────────
        # diff = treatment - control (positive if treatment better)
        diff = trt_arr - ctrl_arr
        if metric not in higher_better:
            diff = -diff   # MaxDD: lower is better, flip

        t_stat, p_two = stats.ttest_1samp(diff, popmean=0)
        # H0: mean(diff) <= 0 (treatment NOT better)
        # Reject when t_stat >> 0 (treatment IS better)
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
# 14. Figure 8 — MVO (control) vs BL (treatment) boxplots
# ============================================================

def plot_figure8(mvo_results: dict[str, dict],
                 bl_treatment_results: dict[str, dict],
                 bl_random_runs: dict[str, list[dict]]):
    """
    Figure 8: 2x2 boxplots.
    Control  = MVO strategies (mvo_lo_2/3/4, mvo_lns_2/3/4).
    Treatment = BL non-random strategies.
    p-values from t-test comparing the two groups.
    """
    metrics = {
        "sharpe"       : "Sharpe",
        "sortino"      : "Sortino",
        "max_drawdown" : "MaxDD",
        "pct_positive" : "% Positive Ret.",
    }

    ctrl_vals = {m: [] for m in metrics}
    trt_vals  = {m: [] for m in metrics}

    for m_dict in mvo_results.values():
        for metric in metrics:
            v = m_dict.get(metric, np.nan)
            if not np.isnan(v):
                ctrl_vals[metric].append(v)

    for m_dict in bl_treatment_results.values():
        for metric in metrics:
            v = m_dict.get(metric, np.nan)
            if not np.isnan(v):
                trt_vals[metric].append(v)

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

        # p-value: one-sided t-test H0: ctrl mean >= trt mean
        if len(ctrl) > 1 and len(trt) > 1:
            t_stat, p_two = stats.ttest_ind(ctrl, trt)
            higher_better = metric in ("sharpe", "sortino", "pct_positive")
            if higher_better:
                # trt better means trt mean > ctrl mean -> t_stat < 0 (ctrl lower)
                p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
            else:
                p_one = p_two / 2 if t_stat < 0 else 1.0 - p_two / 2
            p_str = f"p-value: {p_one:.3f} (H\u2080: \u03bc_c > \u03bc_t)"
        else:
            p_str = ""

        ax.set_title(label, fontsize=11)
        ax.text(0.5, 0.02, p_str, transform=ax.transAxes,
                ha="center", fontsize=8, color="dimgrey")

    fig.suptitle(
        "MVO (control) vs Black-Litterman with Regimes (treatment)",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUTPUTS_DIR / "figure8_bl_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ============================================================
# 15. Figure 12 — BL vs Benchmarks
# ============================================================

def plot_figure12(bl_scaled: dict[str, pd.Series],
                  mvo_scaled_dict: dict[str, pd.Series],
                  spy_scaled: pd.Series,
                  ew_scaled: pd.Series):
    """
    Figure 12: BL strategies (pink) vs EW (black), SPY (black), MVO (red).
    Cumulative log returns, 10% vol scaling.
    """
    all_series = (list(bl_scaled.values()) +
                  list(mvo_scaled_dict.values()) +
                  [spy_scaled, ew_scaled])
    common = all_series[0].index
    for s in all_series[1:]:
        common = common.intersection(s.index)
    common = common.sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))

    # BL strategies (pink)
    for name, series in bl_scaled.items():
        cum = series.reindex(common).cumsum()
        ax.plot(cum.index, cum.values,
                color="hotpink", linewidth=0.8, alpha=0.7)

    # MVO strategies (red)
    for name, series in mvo_scaled_dict.items():
        cum = series.reindex(common).cumsum()
        ax.plot(cum.index, cum.values,
                color="red", linewidth=0.8, alpha=0.5)

    # Benchmarks (thick)
    for series, color, label, lw in [
        (ew_scaled,  "black", "EW",  2.0),
        (spy_scaled, "grey",  "SPY", 2.0),
    ]:
        cum = series.reindex(common).cumsum()
        ax.plot(cum.index, cum.values,
                color=color, label=label, linewidth=lw, zorder=5)

    # Legend
    bl_line  = Line2D([0], [0], color="hotpink", lw=1.5, label="BL")
    mvo_line = Line2D([0], [0], color="red",     lw=1.5, label="MVO")
    ew_line  = Line2D([0], [0], color="black",   lw=2.0, label="EW")
    spy_line = Line2D([0], [0], color="grey",    lw=2.0, label="SPY")
    ax.legend(handles=[ew_line, spy_line, bl_line, mvo_line],
              loc="upper left", fontsize=9)

    ax.set_title("Black-Litterman Portfolio (bl) vs Benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Returns (Vol. Target of 10%)")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    fig.tight_layout()
    out = OUTPUTS_DIR / "figure12_bl_vs_benchmarks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ============================================================
# 16. Table 5 — Performance metrics
# ============================================================

def build_table5(bl_series: dict[str, pd.Series],
                 mvo_series: dict[str, pd.Series],
                 spy_scaled: pd.Series,
                 ew_scaled: pd.Series) -> pd.DataFrame:
    """Build Table 5 in paper format (BL + MVO + benchmarks)."""
    all_series = {**bl_series, **mvo_series,
                  "spy": spy_scaled, "ew": ew_scaled}
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
# 17. Main
# ============================================================

def main():
    print("=" * 70)
    print("BLACK-LITTERMAN REGIME-CONDITIONED BACKTEST")
    print("=" * 70)

    etf_returns, next_probs, regime_labels = load_inputs()

    # ── BL treatment backtest ──────────────────────────────────────────
    print("\nRunning BL treatment backtest (non-random regimes) …")
    bl_treatment_series = run_bl_backtest(
        etf_returns, next_probs, regime_labels, use_random_regimes=False
    )

    # ── Common index ──────────────────────────────────────────────────
    predicted_shifted = get_predicted_regimes(next_probs)
    common_idx        = build_common_index(
        etf_returns, predicted_shifted, regime_labels
    )

    # ── MVO control strategies ────────────────────────────────────────
    mvo_series = run_mvo_backtest(etf_returns, common_idx)

    # ── SPY and EW benchmarks ─────────────────────────────────────────
    spy_raw    = etf_returns["SPY"].reindex(common_idx).dropna()
    ew_raw     = etf_returns.reindex(common_idx).mean(axis=1).dropna()
    spy_scaled = apply_vol_scaling(spy_raw)
    ew_scaled  = apply_vol_scaling(ew_raw)

    # ── Performance metrics ────────────────────────────────────────────
    bl_treatment_results = {n: performance_metrics(s)
                            for n, s in bl_treatment_series.items()}
    mvo_results          = {n: performance_metrics(s)
                            for n, s in mvo_series.items()}

    # ── Table 5 ───────────────────────────────────────────────────────
    table5 = build_table5(bl_treatment_series, mvo_series,
                          spy_scaled, ew_scaled)
    print("\n" + "=" * 70)
    print("TABLE 5 — Performance metrics (BL + MVO, vol-scaled 10%)")
    print("=" * 70)
    print(table5.to_string())
    table5.to_csv(OUTPUTS_DIR / "table5_bl_performance.csv")
    print(f"\nSaved: table5_bl_performance.csv")

    # ── Random BL baseline ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RANDOM REGIME BASELINE (BL control)")
    print("=" * 70)
    bl_random_runs = run_random_baseline(etf_returns, next_probs, regime_labels)

    # ── Table 3 Panel B ───────────────────────────────────────────────
    table3b = compute_table3_panel_b(bl_random_runs, bl_treatment_results)
    print("\n" + "=" * 70)
    print("TABLE 3 Panel B — Nemenyi Test (BL Random vs not Random)")
    print("=" * 70)
    display_cols = ["Control Rank", "Treatment Rank", "p-value", "sig"]
    print(table3b[display_cols].to_string())
    print("\nFull-precision p-values:")
    for idx in table3b.index:
        pf = table3b.loc[idx, "p-value-full"]
        print(f"  {idx:<20}: {pf:.8f}  {table3b.loc[idx,'sig']}")
    table3b.to_csv(OUTPUTS_DIR / "table3b_nemenyi_bl.csv")
    print(f"\nSaved: table3b_nemenyi_bl.csv")

    # ── Figures ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_figure8(mvo_results, bl_treatment_results, bl_random_runs)
    plot_figure12(bl_treatment_series, mvo_series, spy_scaled, ew_scaled)

    # ── Save BL returns ───────────────────────────────────────────────
    pd.DataFrame(bl_treatment_series).to_csv(
        PROCESSED_DIR / "bl_strategy_returns_paper_like.csv"
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("\nTop 5 BL strategies by Sharpe:")
    top5 = (table5["Sharpe"]
            .filter(like="bl_")
            .sort_values(ascending=False).head(5))
    print(top5.round(3).to_string())
    print("\nOutputs saved to outputs/:")
    for f in ["table5_bl_performance.csv", "table3b_nemenyi_bl.csv",
              "figure8_bl_boxplot.png", "figure12_bl_vs_benchmarks.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()