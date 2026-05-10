from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FRED_PATH = RAW_DIR / "fred_md_current.csv"
ETF_PATH  = RAW_DIR / "etf_prices.csv"


# ============================================================
# 2. Settings  (paper Sections 4.1, 4.2, 6)
# ============================================================

MACRO_START_DATE = "1959-12-01"   # paper: Dec 1959 – Jan 2023
MACRO_END_DATE   = "2023-01-01"

ETF_RETURN_START_DATE = "2000-02-01"   # paper: Feb 2000 – Dec 2022
ETF_RETURN_END_DATE   = "2022-12-01"

ETF_TICKERS = [
    "SPY", "XLB", "XLE", "XLF", "XLI",
    "XLK", "XLP", "XLU", "XLV", "XLY",
]

# Group 6: Interest and Exchange Rates — excluded per Section 4.1
GROUP_6_INTEREST_EXCHANGE = [
    "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10",
    "AAA", "BAA", "COMPAPFFx", "TB3SMFFM", "TB6SMFFM", "T1YFFM",
    "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM", "TWEXAFEGSMTHx",
    "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx",
]

# Missing-value handling thresholds
# Drop a column only if it is missing for more than 30% of the sample.
# This is the minimum intervention needed to stay faithful to the paper
# while avoiding columns that are structurally empty.
COL_NAN_THRESHOLD = 0.30


# ============================================================
# 3. Raw data checks
# ============================================================

def check_file_exists(path: Path, min_size_kb: float):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    size_kb = path.stat().st_size / 1024
    print(f"  Found: {path.name}  ({size_kb:.1f} KB)")
    if size_kb < min_size_kb:
        raise ValueError(
            f"{path.name} is only {size_kb:.1f} KB "
            f"(expected >= {min_size_kb} KB). Check the download."
        )


def check_fred_md():
    print("\n" + "=" * 70)
    print("CHECK 1/2 — FRED-MD RAW FILE")
    print("=" * 70)

    check_file_exists(FRED_PATH, min_size_kb=200)

    fred = pd.read_csv(FRED_PATH)
    print(f"  Raw shape : {fred.shape}")

    first_cell = str(fred.iloc[0, 0])
    if "transform" not in first_cell.lower():
        raise ValueError(
            f"Expected t-code row in row 0, got: '{first_cell}'. "
            "Check that the file is an unmodified FRED-MD current.csv."
        )
    print("  T-code row detected in row 0 — OK")

    dates = pd.to_datetime(fred.iloc[1:, 0], errors="coerce")
    print(f"  Date range : {dates.min().date()} to {dates.max().date()}")

    if dates.min() > pd.Timestamp("1960-01-01"):
        raise ValueError("FRED-MD dates start later than expected. Check the file.")


def check_etf_prices():
    print("\n" + "=" * 70)
    print("CHECK 2/2 — ETF PRICE FILE")
    print("=" * 70)

    check_file_exists(ETF_PATH, min_size_kb=100)

    etf = pd.read_csv(ETF_PATH, index_col=0, parse_dates=True)
    print(f"  Shape      : {etf.shape}")
    print(f"  Date range : {etf.index.min().date()} to {etf.index.max().date()}")

    missing_tickers = [t for t in ETF_TICKERS if t not in etf.columns]
    if missing_tickers:
        raise ValueError(f"Missing tickers in ETF file: {missing_tickers}")
    print("  All 10 ETF tickers present — OK")

    nan_counts = etf[ETF_TICKERS].isna().sum()
    if nan_counts.any():
        print(f"\n  Missing values:\n{nan_counts[nan_counts > 0].to_string()}")
    else:
        print("  No missing values — OK")


def run_checks():
    print("=" * 70)
    print("STARTING RAW DATA CHECKS")
    print("=" * 70)
    check_fred_md()
    check_etf_prices()
    print("\n" + "=" * 70)
    print("ALL RAW DATA CHECKS PASSED")
    print("=" * 70)


# ============================================================
# 4. FRED-MD t-code transformations  (Section 4.2)
# ============================================================

def transform_series(series: pd.Series, tcode: int) -> pd.Series:
    """
    Apply FRED-MD t-code transformation (Section 4.2).
      1 = level
      2 = first difference
      3 = second difference
      4 = log
      5 = first difference of log
      6 = second difference of log
      7 = first difference of pct change
    """
    x = pd.to_numeric(series, errors="coerce")

    if tcode == 1:
        return x
    elif tcode == 2:
        return x.diff()
    elif tcode == 3:
        return x.diff().diff()
    elif tcode == 4:
        return np.log(x)
    elif tcode == 5:
        return np.log(x).diff()
    elif tcode == 6:
        return np.log(x).diff().diff()
    elif tcode == 7:
        return x.pct_change().diff()
    else:
        raise ValueError(f"Unknown t-code: {tcode}")


# ============================================================
# 5. Preprocess FRED-MD  (Section 4.2)
# ============================================================

def preprocess_fred_md() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PREPROCESSING FRED-MD — MACRO PCA")
    print("=" * 70)

    fred_raw = pd.read_csv(FRED_PATH)

    # Row 0 = t-codes, rows 1+ = actual data
    transform_codes = fred_raw.iloc[0].copy()
    fred = fred_raw.iloc[1:].copy()

    fred = fred.rename(columns={fred.columns[0]: "date"})
    fred["date"] = pd.to_datetime(fred["date"])
    fred = fred.set_index("date").sort_index()

    # Paper: Dec 1959 – Jan 2023
    fred = fred.loc[MACRO_START_DATE:MACRO_END_DATE]
    print(f"  After date filter          : {fred.shape}")

    # Drop Group 6 (Interest & Exchange Rates) — Section 4.1
    drop_cols  = [c for c in GROUP_6_INTEREST_EXCHANGE if c in fred.columns]
    missing_g6 = [c for c in GROUP_6_INTEREST_EXCHANGE if c not in fred.columns]
    fred = fred.drop(columns=drop_cols)
    print(f"  Group 6 dropped            : {len(drop_cols)} columns")
    if missing_g6:
        print(f"  Group 6 not in file        : {missing_g6}")
    print(f"  After Group 6 removal      : {fred.shape}")

    # Apply t-code transformations column by column
    transformed = pd.DataFrame(
        {col: transform_series(fred[col], int(float(transform_codes[col])))
         for col in fred.columns},
        index=fred.index,
    )

    # Replace ±inf (e.g. log of zero) with NaN
    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    print(f"  After t-code transforms    : {transformed.shape}")

    # ── Missing value strategy (faithful to paper) ──────────────────────
    #
    # The paper does not mention dropping rows or winsorizing.
    # Many FRED-MD series simply start later than 1959 — their early NaNs
    # are structural, not errors.  Dropping rows with ANY NaN would remove
    # all months before the latest-starting series, shrinking the sample
    # from 1959 to ~1992 and wiping out historical recession data.
    #
    # Strategy (minimum intervention):
    #   Step 1: Drop columns missing > 30% of observations
    #           (series that barely exist add noise, not signal)
    #   Step 2: Forward-fill then backward-fill remaining NaNs
    #           (ffill is standard for macro time series;
    #            bfill handles the very first rows of late-starting series)
    #   Step 3: Verify no NaNs remain
    # ────────────────────────────────────────────────────────────────────

    # Step 1: drop high-NaN columns
    nan_rates   = transformed.isna().mean()
    high_nan    = nan_rates[nan_rates > COL_NAN_THRESHOLD].index.tolist()
    if high_nan:
        print(f"\n  Columns with >{COL_NAN_THRESHOLD*100:.0f}% NaN dropped ({len(high_nan)}):")
        for c in high_nan:
            print(f"    {c}  ({nan_rates[c]*100:.1f}%)")
        transformed = transformed.drop(columns=high_nan)

    print(f"  After dropping sparse cols : {transformed.shape}")

    # Step 2: forward-fill then backward-fill
    # ffill propagates last valid observation forward (handles mid-series gaps)
    # bfill fills the leading NaNs at the start of late-starting series
    transformed = transformed.ffill().bfill()
    print(f"  After ffill + bfill        : {transformed.shape}")
    print(f"  Date range                 : {transformed.index.min().date()} "
          f"to {transformed.index.max().date()}")

    # Step 3: verify clean
    remaining_nan = transformed.isna().sum().sum()
    if remaining_nan > 0:
        bad_cols = transformed.columns[transformed.isna().any()].tolist()
        raise ValueError(
            f"{remaining_nan} NaNs remain after ffill+bfill in: {bad_cols}"
        )
    print("  No remaining NaNs — OK")

    # Save transformed data
    transformed_path = PROCESSED_DIR / "macro_transformed_paper_like.csv"
    transformed.to_csv(transformed_path)
    print(f"\n  Saved transformed          : {transformed_path.name}")

    # Standardise: demean + unit variance (Section 4.2)
    scaler       = StandardScaler()
    scaled_array = scaler.fit_transform(transformed.values)
    macro_scaled = pd.DataFrame(scaled_array,
                                index=transformed.index,
                                columns=transformed.columns)

    scaled_path = PROCESSED_DIR / "macro_scaled_paper_like.csv"
    macro_scaled.to_csv(scaled_path)
    print(f"  Saved scaled               : {scaled_path.name}")

    # PCA: retain enough components for >= 95% variance
    # Paper Section 4.2 says this yields 61 components.
    # random_state=42 ensures reproducibility.
    #
    # NO winsorize, NO norm clipping before or after PCA.
    # Diagnostic confirmed that crisis months (2008, 2020) naturally have
    # the highest PCA row-norms (~18-20 vs mean ~8), which is exactly what
    # L2 KMeans k=2 needs to correctly identify them as "deviant months"
    # (Regime 0, Section 3.1.1). Any clipping destroys this signal.
    pca       = PCA(n_components=0.95, random_state=42)
    pca_array = pca.fit_transform(macro_scaled.values)

    pc_cols   = [f"PC{i+1}" for i in range(pca_array.shape[1])]
    macro_pca = pd.DataFrame(pca_array,
                             index=macro_scaled.index,
                             columns=pc_cols)

    print(f"\n  PCA components retained    : {macro_pca.shape[1]}  "
          f"(paper target: ~61)")
    print(f"  Cumulative variance        : "
          f"{pca.explained_variance_ratio_.sum():.4f}")

    if macro_pca.shape[1] < 40 or macro_pca.shape[1] > 90:
        print(f"  WARNING: expected ~61 PCs, got {macro_pca.shape[1]}. "
              "Check column count after Group 6 removal.")

    # Diagnostic: crisis months should have the highest row-norms
    row_norms = np.linalg.norm(macro_pca.values, axis=1)
    top5_idx  = np.argsort(row_norms)[::-1][:5]
    print(f"\n  PCA row-norm stats:")
    print(f"    mean={row_norms.mean():.2f}  "
          f"p95={np.percentile(row_norms,95):.2f}  "
          f"max={row_norms.max():.2f}")
    print(f"    Top 5 months by norm (should be crisis periods): "
          f"{[str(macro_pca.index[i].date()) for i in top5_idx]}")

    pca_path = PROCESSED_DIR / "macro_pca_paper_like.csv"
    macro_pca.to_csv(pca_path)
    print(f"  Saved PCA                  : {pca_path.name}")

    # Save variance breakdown (for Figure 1 reproduction)
    pca_info = pd.DataFrame({
        "component"                  : pc_cols,
        "explained_variance_ratio"   : pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    pca_info_path = PROCESSED_DIR / "pca_info_paper_like.csv"
    pca_info.to_csv(pca_info_path, index=False)
    print(f"  Saved PCA info             : {pca_info_path.name}")

    return macro_pca


# ============================================================
# 6. Preprocess ETF prices -> monthly log returns  (Section 6)
# ============================================================

def preprocess_etf_prices() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PREPROCESSING ETF PRICES — MONTHLY LOG RETURNS")
    print("=" * 70)

    prices = pd.read_csv(ETF_PATH, index_col=0, parse_dates=True)
    prices = prices[ETF_TICKERS].sort_index()
    print(f"  Raw daily prices shape     : {prices.shape}")

    # Resample to beginning-of-month price (first trading day of each month)
    monthly_prices = prices.resample("MS").first()
    print(f"  Monthly BOM prices         : {monthly_prices.shape}")

    # Log returns: ln(P_t / P_{t-1})
    monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

    # Clip to paper's ETF window: Feb 2000 – Dec 2022
    monthly_returns = monthly_returns.loc[ETF_RETURN_START_DATE:ETF_RETURN_END_DATE]

    print(f"  Monthly log returns        : {monthly_returns.shape}")
    print(f"  Date range                 : {monthly_returns.index.min().date()} "
          f"to {monthly_returns.index.max().date()}")

    nan_counts = monthly_returns.isna().sum()
    if nan_counts.any():
        print(f"\n  Missing values:\n{nan_counts[nan_counts > 0].to_string()}")
    else:
        print("  No missing values — OK")

    returns_path = PROCESSED_DIR / "etf_monthly_returns_paper_like.csv"
    monthly_returns.to_csv(returns_path)
    print(f"\n  Saved ETF log returns      : {returns_path.name}")

    return monthly_returns


# ============================================================
# 7. Merge macro PCA + ETF returns
# ============================================================

def merge_datasets(macro_pca: pd.DataFrame,
                   monthly_returns: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("MERGING MACRO PCA AND ETF RETURNS")
    print("=" * 70)

    merged = macro_pca.join(monthly_returns, how="inner")

    print(f"  Macro PCA shape            : {macro_pca.shape}")
    print(f"  ETF return shape           : {monthly_returns.shape}")
    print(f"  Merged shape               : {merged.shape}")
    print(f"  Date range                 : {merged.index.min().date()} "
          f"to {merged.index.max().date()}")

    if merged.isna().sum().sum() > 0:
        raise ValueError("NaN values in merged data — check index alignment.")

    merged_path = PROCESSED_DIR / "merged_data_paper_like.csv"
    merged.to_csv(merged_path)
    print(f"\n  Saved merged data          : {merged_path.name}")

    return merged


# ============================================================
# 8. Main
# ============================================================

def main():
    print("=" * 70)
    print("PREPROCESS_DATA.PY — RAW CHECKS + PREPROCESSING")
    print("=" * 70)

    run_checks()

    macro_pca       = preprocess_fred_md()
    monthly_returns = preprocess_etf_prices()
    merged          = merge_datasets(macro_pca, monthly_returns)

    print("\n" + "=" * 70)
    print("ALL PREPROCESSING COMPLETED")
    print("=" * 70)

    print("\nGenerated files in data/processed/:")
    print("  macro_transformed_paper_like.csv")
    print("  macro_scaled_paper_like.csv")
    print("  macro_pca_paper_like.csv")
    print("  pca_info_paper_like.csv")
    print("  etf_monthly_returns_paper_like.csv")
    print("  merged_data_paper_like.csv")

    print(f"\nFinal macro PCA shape  : {macro_pca.shape}")
    print(f"Final merged shape     : {merged.shape}")
    print(f"Date range (macro PCA) : {macro_pca.index.min().date()} "
          f"to {macro_pca.index.max().date()}")
    print(f"Date range (merged)    : {merged.index.min().date()} "
          f"to {merged.index.max().date()}")


if __name__ == "__main__":
    main()