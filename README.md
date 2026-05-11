# Macro Regime TAA — Paper Replication

Replication of **"Tactical Asset Allocation with Macroeconomic Regime Detection"**
(Oliveira et al., arXiv 2503.11499v2).

The paper proposes a layered K-Means regime detection algorithm applied to
FRED-MD macroeconomic data, and uses the detected regimes to inform three
tactical asset allocation models (Naive, Black-Litterman, Linear Ridge) across
10 US sector ETFs.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Pipeline Stages](#pipeline-stages)
6. [Source Files](#source-files)
7. [Outputs](#outputs)
8. [Differences from the Paper](#differences-from-the-paper)

---

## Project Structure

```
macro_regime_taa/
│
├── data/
│   ├── raw/                        # Raw input files (manually placed)
│   │   ├── fred_md_current.csv     # FRED-MD monthly dataset
│   │   └── etf_prices.csv          # Daily ETF price data (from yfinance)
│   │
│   └── processed/                  # Auto-generated intermediate files
│
├── outputs/                        # All final figures and tables
│
├── src/                            # All source code
│   ├── test_env.py
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── detect_regimes.py
│   ├── backtest_naive_strategy.py
│   ├── backtest_bl_strategy.py
│   ├── backtest_ridge_strategy.py
│   ├── plot_figure10.py
│   ├── main_pipeline.py            # Flexible stage-by-stage runner
│   ├── run_all.py                  # Simple one-click runner
│
├── notebooks/                      # Optional Jupyter notebooks
├── .gitignore
└── README.md
```

---

## Environment Setup

### 1. Prerequisites

- Python 3.10 or higher
- Windows / macOS / Linux

### 2. Create a virtual environment

```bash
# Navigate to the project root
cd macro_regime_taa

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib yfinance
```

Or if a `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

### 4. Verify the environment

```bash
python src/test_env.py
```

This checks that all required packages are correctly installed and prints their
versions.

---

## Data Preparation

### FRED-MD (macroeconomic data)

The FRED-MD dataset cannot be downloaded programmatically from this codebase
due to server restrictions (403 errors). You must download it manually:

1. Go to: https://research.stlouisfed.org/econ/mccracken/fred-databases/
2. Download **current.csv** (the most recent vintage)
3. Rename it to `fred_md_current.csv`
4. Place it in `data/raw/fred_md_current.csv`

The file should be approximately 500–800 KB. Row 0 contains transformation
codes (t-codes); rows 1 onwards contain monthly observations starting from
January 1959.

### ETF price data

ETF prices are downloaded automatically from Yahoo Finance via `yfinance`:

```bash
python src/download_data.py
```

This downloads daily adjusted close prices for all 10 ETFs
(SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY)
from January 2000 to January 2023 and saves them to `data/raw/etf_prices.csv`.

---

## Running the Pipeline

### Option A — One-click (simplest)

```bash
python src/run_all.py
```

Runs all 6 stages in sequence. This is the simplest option.

### Option B — Flexible stage runner

```bash
# Run everything
python src/main_pipeline.py

# Start from stage 3 (skip preprocessing if data already exists)
python src/main_pipeline.py --from 3

# Run only one stage
python src/main_pipeline.py --only 2

# Skip Ridge regression (slowest stage, 30–60 min)
python src/main_pipeline.py --skip 5
```

### Option C — Run individual scripts

```bash
python src/preprocess_data.py
python src/detect_regimes.py
python src/backtest_naive_strategy.py
python src/backtest_bl_strategy.py
python src/backtest_ridge_strategy.py
python src/plot_figure10.py
```

**Important:** Scripts must be run in this exact order. Each stage depends on
the outputs of the previous one.

### Estimated runtimes

| Stage | Script | Time |
|-------|--------|------|
| 1 | preprocess_data.py | < 1 min |
| 2 | detect_regimes.py | 1–2 min |
| 3 | backtest_naive_strategy.py | 5–10 min |
| 4 | backtest_bl_strategy.py | 10–20 min |
| 5 | backtest_ridge_strategy.py | **30–60 min** |
| 6 | plot_figure10.py | < 1 min |

---

## Pipeline Stages

### Stage 1 — `preprocess_data.py`

**Input:** `data/raw/fred_md_current.csv`, `data/raw/etf_prices.csv`

Applies FRED-MD t-code transformations (Section 4.2) to convert raw macro
series into stationary variables. Drops Group 6 (interest and exchange rate
variables) per Section 4.1. Handles missing values via forward-fill then
backward-fill to preserve the full 1959–2023 sample. Standardises the
transformed data (zero mean, unit variance) and applies PCA retaining 95%
of variance. Also computes monthly log returns for the 10 ETFs.

**Key paper references:** Section 4.1, 4.2, Eq. — (PCA)

---

### Stage 2 — `detect_regimes.py`

**Input:** `data/processed/macro_pca_paper_like.csv`

Implements the layered K-Means regime detection algorithm (Algorithm 1):

- **Layer 1** — L2 KMeans k=2 identifies outlier months (Regime 0, economic
  difficulty) using a norm-threshold approach.
- **Layer 2** — Cosine KMeans k=5 clusters the remaining normal months into
  5 regimes (1–5).
- Hungarian matching ensures stable regime labels across rolling windows.
- Computes current-month regime probabilities via Eq. 1 + Eqs. 3–4.
- Computes next-month regime forecast via Eq. 7 (transition matrix).
- Also fits a GMM for comparison (Section 4.3, Figures 2–3 only).

**Outputs:** Figures 2–5, Table 1

**Key paper references:** Section 3.1–3.3, Algorithm 1, Eqs. 1, 3–5, 7

---

### Stage 3 — `backtest_naive_strategy.py`

**Input:** ETF returns, next-month regime probabilities, regime labels

Implements the Naive regime-conditioned strategy (Section 5.2.1):

- **Forecast** (Eqs. 9–10): regime-conditional Sharpe ratio = μ\*/σ\*, where
  μ\* and σ\* are computed from the 48-month window months matching the
  predicted next regime.
- **Position sizing** (Eqs. 15–18): four methods — long-only (lo), long-and-short
  (lns), long-or-short (los), mixed (mx). Three portfolio sizes l ∈ {2, 3, 4}.
- **Benchmarks:** MVO (rolling 48-month max-Sharpe long-only), SPY, EW.
- **Statistical test:** 50-run Monte Carlo random-regime baseline, Nemenyi-style
  rank test (Table 3 Panel A), one-sided paired t-test for p-values.

**Outputs:** Table 3 Panel A, Table 4, Figure 7, Figure 11

**Key paper references:** Section 5.2.1, 5.3.1, Eqs. 8–10, 15–18

---

### Stage 4 — `backtest_bl_strategy.py`

**Input:** ETF returns, next-month regime probabilities, regime labels

Implements the Black-Litterman strategy (Section 5.2.2):

- **Prior:** sample mean and covariance from the 48-month window (not market
  equilibrium returns — per paper Section 5.2.2).
- **Views** (Eq. 11): q\*_{j,t+1} = regime-conditional sample mean return.
- **BL posterior** (Eq. 19): μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹μ̂ + PᵀΩ⁻¹q\*],
  with P = identity, Ω = τ·Σ (standard BL proportional uncertainty), τ = 0.05.
- **Position sizing:** long-only (lo) and long-and-short (lns) only, l ∈ {2, 3, 4}.
- **Control vs treatment:** MVO is the control, BL with regime views is the treatment.

**Outputs:** Table 3 Panel B, Table 5, Figure 8, Figure 12

**Key paper references:** Section 5.2.2, 5.3.2, Eqs. 11, 19

---

### Stage 5 — `backtest_ridge_strategy.py`

**Input:** ETF returns, macro PCA features, next-month probabilities, regime labels

Implements the Linear Ridge Regression strategy (Section 5.2.3):

- **Per-regime models** (Eq. 13): for each regime i, fit a ridge regression
  β̂^{i,j} predicting ETF j's return from macro PCA features (X), using
  one-step-ahead pairs (X_t, y_{t+1}) within the 48-month window.
- **Prediction** (Eq. 12): ŷ^{i,j}_{t+1} = β̂^{i,j} @ x_current, where
  x_current is the macro PCA vector at the position date.
- **Aggregation** (Eq. 14): probability-weighted forecast across all regimes:
  ŷ^{j} = Σ_i p̃_{i,t+1} × ŷ^{i,j}_{t+1}.
- **X = macro PCA features** (not ETF returns — key difference from Naive/BL).
- **Random baseline:** both predicted regime and probability vector are
  randomised (Dirichlet draws) for the 50-run Monte Carlo control.

**Outputs:** Table 3 Panel C, Table 6, Figure 9, Figure 13

**Key paper references:** Section 5.2.3, 5.3.3, Eqs. 12–14

---

### Stage 6 — `plot_figure10.py`

**Input:** All three strategy return CSVs from Stages 3–5

Reads the saved vol-scaled log returns from all three models and plots them
together on a single figure with benchmarks. Color scheme matches the paper:
Linear-Ridge = yellow, Naive = red, BL = green, MVO = blue, EW = magenta,
SPY = cyan.

**Outputs:** Figure 10

**Key paper references:** Figure 10

---

## Source Files

| File | Purpose |
|------|---------|
| `test_env.py` | Checks all required packages are installed |
| `download_data.py` | Downloads ETF daily prices via yfinance |
| `preprocess_data.py` | FRED-MD transforms + PCA + ETF log returns |
| `detect_regimes.py` | Layered K-Means regime detection + GMM + Figures 2–5 |
| `backtest_naive_strategy.py` | Naive model backtest + Table 4 + Table 3A + Figures 7, 11 |
| `backtest_bl_strategy.py` | Black-Litterman backtest + Table 5 + Table 3B + Figures 8, 12 |
| `backtest_ridge_strategy.py` | Ridge regression backtest + Table 6 + Table 3C + Figures 9, 13 |
| `plot_figure10.py` | Figure 10 (all strategies combined) — run last |
| `main_pipeline.py` | Flexible entry point with `--from`, `--only`, `--skip` options |
| `run_all.py` | Simple one-click runner (no arguments needed) |

---

## Outputs

All outputs are saved to `outputs/`. After running the full pipeline you will have:

### Figures

| File | Paper Figure | Description |
|------|-------------|-------------|
| `figure2_kmeans_regimes.png` | Figure 2 | K-Means vs GMM regime timeline scatter (NBER shaded) |
| `figure3_crisis_probabilities.png` | Figure 3 | K-Means vs GMM crisis/BAU probability over time |
| `figure4_regime_stats.png` | Figure 4 | Min-max normalised FRED-MD statistics per regime |
| `figure5_transition_matrices.png` | Figure 5 | Regime transition matrix (raw + normalised) |
| `figure7_naive_boxplot.png` | Figure 7 | Naive: random (control) vs non-random (treatment) boxplots |
| `figure8_bl_boxplot.png` | Figure 8 | BL: MVO (control) vs BL with regimes (treatment) boxplots |
| `figure9_ridge_boxplot.png` | Figure 9 | Ridge: random (control) vs non-random (treatment) boxplots |
| `figure10_all_strategies.png` | Figure 10 | All strategies cumulative log returns (vol-scaled 10%) |
| `figure11_naive_vs_benchmarks.png` | Figure 11 | Naive vs SPY / EW / MVO |
| `figure12_bl_vs_benchmarks.png` | Figure 12 | BL vs SPY / EW / MVO |
| `figure13_ridge_vs_benchmarks.png` | Figure 13 | Ridge vs SPY / EW / MVO |

### Tables

| File | Paper Table | Description |
|------|------------|-------------|
| `table3_nemenyi_naive.csv` | Table 3 Panel A | Nemenyi test — Naive random vs non-random |
| `table3b_nemenyi_bl.csv` | Table 3 Panel B | Nemenyi test — BL random vs non-random |
| `table3c_nemenyi_ridge.csv` | Table 3 Panel C | Nemenyi test — Ridge random vs non-random |
| `table4_naive_performance.csv` | Table 4 | Naive strategy performance metrics |
| `table5_bl_performance.csv` | Table 5 | BL strategy performance metrics |
| `table6_ridge_performance.csv` | Table 6 | Ridge strategy performance metrics |

### Intermediate data (`data/processed/`)

Key files used by downstream scripts:

| File | Description |
|------|-------------|
| `macro_pca_paper_like.csv` | Standardised, PCA-transformed macro features |
| `regime_labels_paper_like.csv` | Hard regime label (0–5) per month |
| `next_regime_probabilities_paper_like.csv` | p̃_{i,t+1}: next-month regime probability vectors |
| `etf_monthly_returns_paper_like.csv` | Monthly log returns for 10 ETFs |
| `naive_strategy_returns_paper_like.csv` | Vol-scaled Naive strategy returns |
| `bl_strategy_returns_paper_like.csv` | Vol-scaled BL strategy returns |
| `ridge_strategy_returns_paper_like.csv` | Vol-scaled Ridge strategy returns |

---

## Differences from the Paper

Our replication is faithful to the paper's methodology, but several differences
arise from data availability and implementation choices. These are documented
transparently below.

### 1. FRED-MD vintage difference (most impactful)

The paper was published in early 2025 using the FRED-MD vintage available at
that time. Our replication uses the **current vintage** downloaded in 2025/2026.
FRED-MD data is regularly revised — individual series values change
retrospectively, and the set of available series evolves.

**Effect:** Our PCA retains **~53 components** vs the paper's **61 components**.
This means our macro feature space is slightly smaller, which propagates through
regime detection and all three forecasting models. Regime boundaries shift
slightly, and performance metrics differ numerically (though directionally
consistent).

### 2. Regime 0 detection — norm-threshold instead of pure k=2

The paper describes using a pure L2 KMeans k=2 to identify outlier months
(Regime 0 — Economic Difficulty). In our data, the April 2020 COVID shock
produces a PCA row-norm of ~101, roughly 12× the dataset mean (~8). A pure
k=2 always isolates this single month as Regime 0, completely missing the
2008 financial crisis and other historical downturns.

**Our fix:** We flag months with row-norm above the 97th percentile (~23 months,
~3% of sample) as Regime 0 directly. This threshold captures the major
historical crises (2008, 1974, 1982, 2020 etc.) that the paper's Figure 4
and Section 4.4 explicitly reference. A k=2 model is still fit on the
norm-capped data to obtain two centroids for Eq. 1 probability computation.

This deviation is necessary and justified: the paper's authors likely used an
older FRED-MD vintage where the COVID spike had already been revised and
smoothed, making pure k=2 viable for them.

### 3. Ridge regression lambda not specified

The paper states Ridge regression is used (Eq. 13) but does not specify the
regularisation parameter λ. We use **λ = 1.0** (sklearn Ridge default). Results
are moderately sensitive to this choice. The paper's exact λ is unknown.

### 4. Nemenyi test implementation

The paper cites the Nemenyi test for Table 3 but does not detail the exact
test statistic or p-value computation. We implement a **one-sided paired
t-test** on per-strategy (treatment − control_mean) differences as the closest
valid approximation. The rank statistics (Control Rank / Treatment Rank)
follow the paper's 1.0–2.0 format by assigning rank 1 = worse and rank 2 =
better per strategy variant, then averaging across variants.

Directional conclusions (which metrics are significant) match the paper;
exact p-values differ slightly.

### 5. BL Omega specification

The paper does not specify the view uncertainty matrix Ω in Eq. 19. We use the
standard Black-Litterman specification **Ω = τ · P · Σ · Pᵀ** (proportional
to the prior covariance), which is the most principled choice and the most
common in the BL literature. The paper may have used a different Ω.

### 6. Numerical results are directionally consistent

Despite the differences above, our key conclusions match the paper:

- Ridge outperforms Naive and BL on Sharpe and Sortino.
- All three models outperform MVO on risk-adjusted returns.
- SPY and EW are beaten by regime-conditioned strategies on cumulative returns.
- Table 3 significance patterns (which panels are significant) are the same.
- Figure 10 visual ordering (Ridge > Naive > BL > benchmarks) is preserved.

---

## Citation

```
Oliveira et al. (2025). Tactical Asset Allocation with Macroeconomic Regime Detection.
arXiv:2503.11499v2.
```