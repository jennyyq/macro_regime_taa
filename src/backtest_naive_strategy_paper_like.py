from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ETF_RETURNS_PATH = PROCESSED_DIR / "etf_monthly_returns_paper_like.csv"
NEXT_REGIME_PROBS_PATH = PROCESSED_DIR / "next_regime_probabilities_softmax_paper_like.csv"


# ============================================================
# 2. Backtest settings
# ============================================================

ETF_TICKERS = [
    "SPY", "XLB", "XLE", "XLF", "XLI",
    "XLK", "XLP", "XLU", "XLV", "XLY"
]

# Paper tests different l values. We include 2, 3, 4.
TOP_L_LIST = [2, 3, 4]

# Position sizing methods from the paper:
# lo  = long-only
# lns = long-and-short
# los = long-or-short
# mx  = mixed: los if predicted regime is 0, otherwise lo
POSITION_METHODS = ["lo", "lns", "los", "mx"]

# Need enough historical data before making decisions
MIN_HISTORY_MONTHS = 48

# Need enough same-regime observations to calculate regime-conditioned Sharpe
MIN_REGIME_OBS = 12

ANNUALIZATION = 12


# ============================================================
# 3. Load data
# ============================================================

def load_inputs():
    print("=" * 70)
    print("LOADING INPUT DATA")
    print("=" * 70)

    etf_returns = pd.read_csv(
        ETF_RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    next_probs = pd.read_csv(
        NEXT_REGIME_PROBS_PATH,
        index_col=0,
        parse_dates=True
    )

    etf_returns = etf_returns[ETF_TICKERS]

    print("\nETF monthly returns:")
    print(f"Shape: {etf_returns.shape}")
    print(f"Date range: {etf_returns.index.min()} to {etf_returns.index.max()}")

    print("\nNext regime probabilities:")
    print(f"Shape: {next_probs.shape}")
    print(f"Date range: {next_probs.index.min()} to {next_probs.index.max()}")

    if etf_returns.isna().sum().sum() > 0:
        raise ValueError("ETF returns contain missing values.")

    if next_probs.isna().sum().sum() > 0:
        raise ValueError("Next regime probabilities contain missing values.")

    return etf_returns, next_probs


# ============================================================
# 4. Align next-regime prediction with ETF return month
# ============================================================

def align_predicted_regime_to_returns(
    etf_returns: pd.DataFrame,
    next_probs: pd.DataFrame
) -> pd.Series:
    """
    Timing logic:

    next_probs at date t means:
        predicted regime distribution for month t+1.

    ETF return at date t+1 means:
        realized return from month t start to month t+1 start.

    Therefore, shift next_probs index forward by one month.
    """

    print("\n" + "=" * 70)
    print("ALIGNING NEXT REGIME PREDICTIONS WITH ETF RETURN DATES")
    print("=" * 70)

    predicted_next_regime = (
        next_probs
        .idxmax(axis=1)
        .str.replace("Next_Regime_", "")
        .astype(int)
    )

    predicted_next_regime.index = predicted_next_regime.index + pd.DateOffset(months=1)
    predicted_next_regime.name = "predicted_regime_for_return_month"

    predicted_next_regime = predicted_next_regime.loc[
        predicted_next_regime.index.intersection(etf_returns.index)
    ]

    aligned_returns = etf_returns.loc[predicted_next_regime.index]

    print("\nPredicted regime aligned to ETF returns:")
    print(f"Shape: {predicted_next_regime.shape}")
    print(f"Date range: {predicted_next_regime.index.min()} to {predicted_next_regime.index.max()}")

    print("\nPredicted regime counts:")
    print(predicted_next_regime.value_counts().sort_index())

    print("\nAligned ETF returns:")
    print(f"Shape: {aligned_returns.shape}")
    print(f"Date range: {aligned_returns.index.min()} to {aligned_returns.index.max()}")

    output_path = PROCESSED_DIR / "predicted_regime_for_return_month_paper_like.csv"
    predicted_next_regime.to_csv(output_path)

    print(f"\nSaved aligned predicted regimes to: {output_path}")

    return predicted_next_regime


# ============================================================
# 5. Naive forecast: regime-conditioned Sharpe vector
# ============================================================

def calculate_monthly_sharpe(returns: pd.DataFrame) -> pd.Series:
    """
    Forecast signal for the naive model.

    We use regime-conditioned historical Sharpe as y_hat.
    This gives signed forecasts:
        positive Sharpe  -> long candidate
        negative Sharpe  -> short candidate
    """

    mean_returns = returns.mean()
    std_returns = returns.std(ddof=1)

    sharpe = mean_returns / std_returns.replace(0, np.nan) * np.sqrt(ANNUALIZATION)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

    return sharpe.reindex(ETF_TICKERS)


def compute_naive_forecast_vector(
    current_date: pd.Timestamp,
    current_predicted_regime: int,
    etf_returns: pd.DataFrame,
    predicted_regime: pd.Series
):
    """
    For month t:
    1. Use only historical months before current_date.
    2. Select historical months whose predicted regime equals current predicted regime.
    3. Compute ETF Sharpe in those same-regime historical months.
    4. If too few same-regime observations, fall back to all historical months.

    Output:
        y_hat = signed Sharpe forecast vector for the 10 ETFs
    """

    historical_returns = etf_returns.loc[etf_returns.index < current_date]
    historical_regimes = predicted_regime.loc[predicted_regime.index < current_date]

    if len(historical_returns) < MIN_HISTORY_MONTHS:
        return None, "not_enough_total_history", 0

    same_regime_dates = historical_regimes[
        historical_regimes == current_predicted_regime
    ].index

    same_regime_dates = same_regime_dates.intersection(historical_returns.index)

    if len(same_regime_dates) >= MIN_REGIME_OBS:
        training_returns = historical_returns.loc[same_regime_dates]
        method = "regime_conditioned"
        n_regime_obs = len(same_regime_dates)
    else:
        training_returns = historical_returns
        method = "fallback_all_history"
        n_regime_obs = len(same_regime_dates)

    forecast = calculate_monthly_sharpe(training_returns)

    if forecast.dropna().empty:
        return None, "not_enough_valid_forecast", n_regime_obs

    return forecast, method, n_regime_obs


# ============================================================
# 6. Position sizing methods from the paper
# ============================================================

def normalize_signed_forecasts(selected_forecasts: pd.Series) -> pd.Series:
    """
    Convert selected signed forecasts into portfolio weights.

    Normalization uses gross exposure:
        weight_j = forecast_j / sum(abs(selected forecasts))

    This allows:
        positive weights for long positions
        negative weights for short positions
    """

    weights = pd.Series(0.0, index=ETF_TICKERS)

    selected_forecasts = selected_forecasts.dropna()

    if selected_forecasts.empty:
        return weights

    denom = selected_forecasts.abs().sum()

    if denom <= 0 or np.isclose(denom, 0):
        return weights

    weights.loc[selected_forecasts.index] = selected_forecasts / denom

    return weights


def position_long_only(forecast: pd.Series, top_l: int) -> pd.Series:
    """
    lo: Long-only.

    Select top l ETFs with positive forecasts.
    Weights are proportional to positive forecasts.
    No short positions.
    """

    positive = forecast[forecast > 0].dropna()

    if positive.empty:
        return pd.Series(0.0, index=ETF_TICKERS)

    selected = positive.sort_values(ascending=False).head(top_l)

    weights = pd.Series(0.0, index=ETF_TICKERS)
    denom = selected.sum()

    if denom <= 0 or np.isclose(denom, 0):
        return weights

    weights.loc[selected.index] = selected / denom

    return weights


def position_long_and_short(forecast: pd.Series, top_l: int) -> pd.Series:
    """
    lns: Long-and-short.

    Select:
        H = top l positive forecasts
        L = bottom l negative forecasts

    Take both long and short positions.
    Weights are normalized by gross exposure.
    """

    positive = forecast[forecast > 0].dropna().sort_values(ascending=False).head(top_l)
    negative = forecast[forecast < 0].dropna().sort_values(ascending=True).head(top_l)

    selected = pd.concat([positive, negative])

    return normalize_signed_forecasts(selected)


def position_long_or_short(forecast: pd.Series, top_l: int) -> pd.Series:
    """
    los: Long-or-short.

    Select the l ETFs with largest absolute forecast magnitude.
    If forecast is positive -> long.
    If forecast is negative -> short.
    Weights are normalized by gross exposure.
    """

    valid = forecast.dropna()

    if valid.empty:
        return pd.Series(0.0, index=ETF_TICKERS)

    selected_names = valid.abs().sort_values(ascending=False).head(top_l).index
    selected = valid.loc[selected_names]

    return normalize_signed_forecasts(selected)


def position_mixed(forecast: pd.Series, top_l: int, predicted_regime: int) -> pd.Series:
    """
    mx: Mixed strategy.

    If predicted next regime is Regime 0, use long-or-short.
    Otherwise, use long-only.

    This follows the paper's economic logic:
        Regime 0 = economic difficulty,
        shorts may help during stress periods.
    """

    if int(predicted_regime) == 0:
        return position_long_or_short(forecast, top_l)
    else:
        return position_long_only(forecast, top_l)


def build_weights(
    forecast: pd.Series,
    top_l: int,
    position_method: str,
    predicted_regime: int
) -> pd.Series:
    if forecast is None:
        return pd.Series(0.0, index=ETF_TICKERS)

    forecast = forecast.reindex(ETF_TICKERS)

    if position_method == "lo":
        return position_long_only(forecast, top_l)

    elif position_method == "lns":
        return position_long_and_short(forecast, top_l)

    elif position_method == "los":
        return position_long_or_short(forecast, top_l)

    elif position_method == "mx":
        return position_mixed(forecast, top_l, predicted_regime)

    else:
        raise ValueError(f"Unknown position_method: {position_method}")


def describe_positions(weights: pd.Series):
    long_etfs = weights[weights > 0].sort_values(ascending=False)
    short_etfs = weights[weights < 0].sort_values(ascending=True)

    long_names = ",".join(long_etfs.index.tolist())
    short_names = ",".join(short_etfs.index.tolist())

    gross_exposure = weights.abs().sum()
    net_exposure = weights.sum()

    return long_names, short_names, gross_exposure, net_exposure


# ============================================================
# 7. Backtest one strategy
# ============================================================

def backtest_naive_strategy(
    etf_returns: pd.DataFrame,
    predicted_regime: pd.Series,
    top_l: int,
    position_method: str
):
    strategy_name = f"naive_top{top_l}_{position_method}"

    print("\n" + "=" * 70)
    print(f"BACKTESTING {strategy_name}")
    print("=" * 70)

    common_dates = etf_returns.index.intersection(predicted_regime.index)

    etf_returns = etf_returns.loc[common_dates]
    predicted_regime = predicted_regime.loc[common_dates]

    strategy_rows = []
    weight_rows = []
    forecast_rows = []

    for current_date in common_dates:
        current_regime = int(predicted_regime.loc[current_date])

        forecast, method, n_regime_obs = compute_naive_forecast_vector(
            current_date=current_date,
            current_predicted_regime=current_regime,
            etf_returns=etf_returns,
            predicted_regime=predicted_regime
        )

        if forecast is None:
            weights = pd.Series(0.0, index=ETF_TICKERS)
            portfolio_return = np.nan
            long_etfs = ""
            short_etfs = ""
            gross_exposure = 0.0
            net_exposure = 0.0
        else:
            weights = build_weights(
                forecast=forecast,
                top_l=top_l,
                position_method=position_method,
                predicted_regime=current_regime
            )

            portfolio_return = float((weights * etf_returns.loc[current_date]).sum())

            long_etfs, short_etfs, gross_exposure, net_exposure = describe_positions(weights)

        strategy_rows.append({
            "date": current_date,
            "strategy_return": portfolio_return,
            "predicted_regime": current_regime,
            "position_method": position_method,
            "top_l": top_l,
            "forecast_method": method,
            "n_same_regime_obs": n_regime_obs,
            "long_etfs": long_etfs,
            "short_etfs": short_etfs,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
        })

        weight_row = {"date": current_date}
        weight_row.update(weights.to_dict())
        weight_rows.append(weight_row)

        if forecast is not None:
            forecast_row = {"date": current_date}
            forecast_row.update(forecast.reindex(ETF_TICKERS).to_dict())
            forecast_rows.append(forecast_row)

    returns_df = pd.DataFrame(strategy_rows).set_index("date")
    weights_df = pd.DataFrame(weight_rows).set_index("date")

    if forecast_rows:
        forecasts_df = pd.DataFrame(forecast_rows).set_index("date")
    else:
        forecasts_df = pd.DataFrame(index=returns_df.index, columns=ETF_TICKERS)

    returns_df = returns_df.dropna(subset=["strategy_return"])

    print("\nBacktest return data:")
    print(f"Shape: {returns_df.shape}")
    print(f"Date range: {returns_df.index.min()} to {returns_df.index.max()}")

    print("\nForecast method counts:")
    print(returns_df["forecast_method"].value_counts())

    print("\nGross exposure summary:")
    print(returns_df["gross_exposure"].describe())

    print("\nNet exposure summary:")
    print(returns_df["net_exposure"].describe())

    print("\nFirst 10 strategy rows:")
    print(returns_df.head(10))

    return strategy_name, returns_df, weights_df, forecasts_df


# ============================================================
# 8. Performance metrics
# ============================================================

def calculate_performance_metrics(return_series: pd.Series) -> dict:
    r = return_series.dropna()

    if len(r) == 0:
        raise ValueError("Return series is empty.")

    cumulative = (1 + r).cumprod()

    total_return = cumulative.iloc[-1] - 1

    annualized_return = (1 + total_return) ** (ANNUALIZATION / len(r)) - 1

    annualized_vol = r.std(ddof=1) * np.sqrt(ANNUALIZATION)

    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else np.nan

    downside = r[r < 0]
    downside_vol = downside.std(ddof=1) * np.sqrt(ANNUALIZATION)

    sortino = annualized_return / downside_vol if downside_vol != 0 else np.nan

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1

    avg_drawdown = drawdown[drawdown < 0].mean()
    max_drawdown = drawdown.min()

    positive_return_pct = (r > 0).mean()

    metrics = {
        "n_months": len(r),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "avg_drawdown": avg_drawdown,
        "max_drawdown": max_drawdown,
        "positive_return_pct": positive_return_pct,
        "mean_monthly_return": r.mean(),
        "monthly_volatility": r.std(ddof=1),
    }

    return metrics


def build_benchmarks(etf_returns: pd.DataFrame, backtest_index: pd.DatetimeIndex):
    aligned = etf_returns.loc[backtest_index]

    benchmarks = pd.DataFrame({
        "Equal_Weight_10_ETFs": aligned.mean(axis=1),
        "SPY": aligned["SPY"],
    })

    return benchmarks


# ============================================================
# 9. Save and plot results
# ============================================================

def save_strategy_outputs(
    all_strategy_returns: dict,
    all_weights: dict,
    all_forecasts: dict,
    etf_returns: pd.DataFrame
):
    print("\n" + "=" * 70)
    print("SAVING STRATEGY OUTPUTS")
    print("=" * 70)

    metrics_rows = []
    combined_returns = pd.DataFrame()

    for name, returns_df in all_strategy_returns.items():
        strategy_return = returns_df["strategy_return"]
        combined_returns[name] = strategy_return

        metrics = calculate_performance_metrics(strategy_return)
        metrics["strategy"] = name
        metrics_rows.append(metrics)

        returns_path = PROCESSED_DIR / f"{name}_returns_paper_like.csv"
        weights_path = PROCESSED_DIR / f"{name}_weights_paper_like.csv"
        forecasts_path = PROCESSED_DIR / f"{name}_forecasts_paper_like.csv"

        returns_df.to_csv(returns_path)
        all_weights[name].to_csv(weights_path)
        all_forecasts[name].to_csv(forecasts_path)

        print(f"Saved {name} returns to: {returns_path}")
        print(f"Saved {name} weights to: {weights_path}")
        print(f"Saved {name} forecasts to: {forecasts_path}")

    # Align all strategies to common non-missing dates
    common_index = combined_returns.dropna().index

    benchmarks = build_benchmarks(etf_returns, common_index)

    for col in benchmarks.columns:
        combined_returns[col] = benchmarks[col]

        metrics = calculate_performance_metrics(benchmarks[col])
        metrics["strategy"] = col
        metrics_rows.append(metrics)

    combined_returns = combined_returns.loc[common_index]

    combined_returns_path = PROCESSED_DIR / "naive_strategy_all_position_returns_paper_like.csv"
    combined_returns.to_csv(combined_returns_path)

    metrics_df = pd.DataFrame(metrics_rows).set_index("strategy")

    metric_order = [
        "n_months",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "avg_drawdown",
        "max_drawdown",
        "positive_return_pct",
        "mean_monthly_return",
        "monthly_volatility",
    ]

    metrics_df = metrics_df[metric_order]

    metrics_path = OUTPUTS_DIR / "naive_strategy_all_position_metrics_paper_like.csv"
    metrics_df.to_csv(metrics_path)

    print(f"\nSaved combined returns to: {combined_returns_path}")
    print(f"Saved metrics to: {metrics_path}")

    print("\nPerformance metrics:")
    print(metrics_df.round(4))

    cumulative_returns = (1 + combined_returns).cumprod()

    plot_path = OUTPUTS_DIR / "naive_strategy_all_position_cumulative_returns_paper_like.png"

    plt.figure(figsize=(14, 7))

    for col in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=1)

    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth of $1")
    plt.title("Naive Regime-Conditioned Strategy: All Position Sizing Methods")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved cumulative return plot to: {plot_path}")

    return metrics_df, combined_returns


# ============================================================
# 10. Main
# ============================================================

def main():
    print("=" * 70)
    print("NAIVE REGIME-CONDITIONED STRATEGY BACKTEST - FULL POSITION SIZING")
    print("=" * 70)

    etf_returns, next_probs = load_inputs()

    predicted_regime = align_predicted_regime_to_returns(
        etf_returns=etf_returns,
        next_probs=next_probs
    )

    all_strategy_returns = {}
    all_weights = {}
    all_forecasts = {}

    for top_l in TOP_L_LIST:
        for position_method in POSITION_METHODS:
            name, returns_df, weights_df, forecasts_df = backtest_naive_strategy(
                etf_returns=etf_returns,
                predicted_regime=predicted_regime,
                top_l=top_l,
                position_method=position_method
            )

            all_strategy_returns[name] = returns_df
            all_weights[name] = weights_df
            all_forecasts[name] = forecasts_df

    metrics_df, combined_returns = save_strategy_outputs(
        all_strategy_returns=all_strategy_returns,
        all_weights=all_weights,
        all_forecasts=all_forecasts,
        etf_returns=etf_returns
    )

    print("\n" + "=" * 70)
    print("FULL NAIVE STRATEGY BACKTEST COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")
    print("data/processed/naive_strategy_all_position_returns_paper_like.csv")
    print("outputs/naive_strategy_all_position_metrics_paper_like.csv")
    print("outputs/naive_strategy_all_position_cumulative_returns_paper_like.png")

    print("\nBest strategies by Sharpe:")
    print(metrics_df.sort_values("sharpe", ascending=False).head(10).round(4))


if __name__ == "__main__":
    main()