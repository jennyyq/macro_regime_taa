from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FRED_PATH = RAW_DIR / "fred_md_current.csv"
ETF_PATH = RAW_DIR / "etf_prices.csv"


# ============================================================
# 2. Paper-like settings
# ============================================================

MACRO_START_DATE = "1959-12-01"
MACRO_END_DATE = "2023-01-01"

ETF_RETURN_START_DATE = "2000-02-01"
ETF_RETURN_END_DATE = "2022-12-01"



#Group 6 in FRED-MD: Interest and Exchange Rates
# The paper explicitly excludes group 6.
GROUP_6_INTEREST_EXCHANGE = [
    "FEDFUNDS",
    "CP3Mx",
    "TB3MS",
    "TB6MS",
    "GS1",
    "GS5",
    "GS10",
    "AAA",
    "BAA",
    "COMPAPFFx",
    "TB3SMFFM",
    "TB6SMFFM",
    "T1YFFM",
    "T5YFFM",
    "T10YFFM",
    "AAAFFM",
    "BAAFFM",
    "TWEXAFEGSMTHx",
    "EXSZUSx",
    "EXJPUSx",
    "EXUSUKx",
    "EXCAUSx",
]


ETF_TICKERS = [
    "SPY", "XLB", "XLE", "XLF", "XLI",
    "XLK", "XLP", "XLU", "XLV", "XLY"
]


# ============================================================
# 3. FRED-MD transformation function
# ============================================================

def transform_series(series: pd.Series, tcode: int) -> pd.Series:
    """
    Apply FRED-MD transformation based on transformation code.

    1: no transformation
    2: first difference
    3: second difference
    4: log
    5: first difference of log
    6: second difference of log
    7: first difference of percentage change
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
        raise ValueError(f"Unknown transformation code: {tcode}")


# ============================================================
# 4. Preprocess FRED-MD paper-like version
# ============================================================

def preprocess_fred_md_paper_like():
    print("\n" + "=" * 70)
    print("PREPROCESSING FRED-MD MACRO DATA - PAPER-LIKE VERSION")
    print("=" * 70)

    fred_raw = pd.read_csv(FRED_PATH)

    print("\nRaw FRED-MD shape:")
    print(fred_raw.shape)

    # First row contains transformation codes
    transform_codes = fred_raw.iloc[0].copy()

    # Actual data starts from row 1
    fred = fred_raw.iloc[1:].copy()

    # Rename date column
    fred = fred.rename(columns={"sasdate": "date"})

    # Convert date
    fred["date"] = pd.to_datetime(fred["date"])

    # Set date index
    fred = fred.set_index("date").sort_index()

    # Cut sample to paper macro period
    fred = fred.loc[MACRO_START_DATE:MACRO_END_DATE]

    print("\nAfter cutting macro sample to paper period:")
    print(fred.shape)
    print(f"Date range: {fred.index.min()} to {fred.index.max()}")

    # Exclude group 6 variables
    existing_group6 = [col for col in GROUP_6_INTEREST_EXCHANGE if col in fred.columns]
    missing_group6 = [col for col in GROUP_6_INTEREST_EXCHANGE if col not in fred.columns]

    print("\nGroup 6 variables found and removed:")
    print(existing_group6)
    print(f"Number removed: {len(existing_group6)}")

    if missing_group6:
        print("\nGroup 6 variables not found in current FRED-MD file:")
        print(missing_group6)

    fred = fred.drop(columns=existing_group6, errors="ignore")

    print("\nAfter excluding group 6:")
    print(fred.shape)

    # Apply transformations more efficiently
    transformed_dict = {}

    for col in fred.columns:
        tcode = int(float(transform_codes[col]))
        transformed_dict[col] = transform_series(fred[col], tcode)

    transformed = pd.DataFrame(transformed_dict, index=fred.index)

    # Replace inf with NaN
    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    
    print("\nAfter transformation:")
    print(transformed.shape)

    print("\nMissing ratio before winsorization, top 15:")
    print(transformed.isna().mean().sort_values(ascending=False).head(15))
    
    # Winsorize extreme values to reduce single-month dominance.
    # # This prevents one extreme COVID month from becoming the only Regime 0.
    
    lower = transformed.quantile(0.005)
    upper = transformed.quantile(0.995)
    transformed = transformed.clip(lower=lower, upper=upper, axis=1)

    print("\nAfter winsorization at 0.5% and 99.5%:")
    print(transformed.shape)

    print("\nMissing ratio after winsorization, top 15:")
    print(transformed.isna().mean().sort_values(ascending=False).head(15))

    # Paper does not specify exact missing-value treatment.
    # To stay closer to the paper, we do NOT drop columns just because missing > 20%.
    # We only drop columns that are completely missing.
    all_missing_cols = transformed.columns[transformed.isna().all()].tolist()

    if all_missing_cols:
        print("\nColumns fully missing and dropped:")
        print(all_missing_cols)
        transformed = transformed.drop(columns=all_missing_cols)

    # Fill remaining missing values
    # First forward fill, then backward fill.
    transformed = transformed.ffill().bfill()

    # Drop any remaining rows/columns with NaN as final safety check
    transformed = transformed.dropna(axis=0, how="any")
    transformed = transformed.dropna(axis=1, how="any")

    print("\nAfter filling missing values:")
    print(transformed.shape)
    print(f"Date range: {transformed.index.min()} to {transformed.index.max()}")

    print("\nAny remaining missing values?")
    print(transformed.isna().sum().sum())

    transformed_path = PROCESSED_DIR / "macro_transformed_paper_like.csv"
    transformed.to_csv(transformed_path)
    print(f"\nSaved transformed macro data to: {transformed_path}")

    # Standardize
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(transformed)

    macro_scaled = pd.DataFrame(
        scaled_array,
        index=transformed.index,
        columns=transformed.columns
    )

    scaled_path = PROCESSED_DIR / "macro_scaled_paper_like.csv"
    macro_scaled.to_csv(scaled_path)
    print(f"Saved scaled macro data to: {scaled_path}")

    # PCA with 95% explained variance
    pca = PCA(n_components=0.95)
    pca_array = pca.fit_transform(macro_scaled)

    pc_columns = [f"PC{i+1}" for i in range(pca_array.shape[1])]

    macro_pca = pd.DataFrame(
        pca_array,
        index=macro_scaled.index,
        columns=pc_columns
    )

    pca_path = PROCESSED_DIR / "macro_pca_paper_like.csv"
    macro_pca.to_csv(pca_path)

    explained_sum = pca.explained_variance_ratio_.sum()

    print("\nPCA completed.")
    print(f"Number of PCs retained: {macro_pca.shape[1]}")
    print(f"Explained variance ratio sum: {explained_sum:.4f}")
    print(f"Saved PCA macro data to: {pca_path}")

    # Save PCA variance detail
    pca_info = pd.DataFrame({
        "component": pc_columns,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)
    })

    pca_info_path = PROCESSED_DIR / "pca_info_paper_like.csv"
    pca_info.to_csv(pca_info_path, index=False)
    print(f"Saved PCA info to: {pca_info_path}")

    return macro_pca


# ============================================================
# 5. Preprocess ETF prices
# ============================================================

def preprocess_etf_prices_paper_like():
    print("\n" + "=" * 70)
    print("PREPROCESSING ETF PRICE DATA - PAPER-LIKE VERSION")
    print("=" * 70)

    prices = pd.read_csv(ETF_PATH, index_col=0, parse_dates=True)
    prices = prices.sort_index()

    prices = prices[ETF_TICKERS]

    print("\nRaw ETF price shape:")
    print(prices.shape)
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")

    print("\nMissing values in raw ETF prices:")
    print(prices.isna().sum())

    # Beginning-of-month price
    monthly_prices = prices.resample("MS").first()

    print("\nMonthly beginning-of-month prices:")
    print(monthly_prices.shape)
    print(monthly_prices.head())

    # Monthly beginning-of-month returns
    monthly_returns = monthly_prices.pct_change().dropna()

    # Paper says ETF data covers Feb 2000 to Dec 2022.
    # Our returns include 2023-01 because it is the return from Dec 2022 start to Jan 2023 start.
    # For closer paper experiment, we keep 2000-02 to 2023-01 for now.
    monthly_returns = monthly_returns.loc[ETF_RETURN_START_DATE:ETF_RETURN_END_DATE]

    print("\nMonthly ETF returns:")
    print(monthly_returns.shape)
    print(f"Date range: {monthly_returns.index.min()} to {monthly_returns.index.max()}")

    print("\nMissing values per ETF return:")
    print(monthly_returns.isna().sum())

    returns_path = PROCESSED_DIR / "etf_monthly_returns_paper_like.csv"
    monthly_returns.to_csv(returns_path)

    print(f"\nSaved ETF monthly returns to: {returns_path}")

    return monthly_returns


# ============================================================
# 6. Merge macro PCA and ETF returns
# ============================================================

def merge_paper_like(macro_pca: pd.DataFrame, monthly_returns: pd.DataFrame):
    print("\n" + "=" * 70)
    print("MERGING PAPER-LIKE MACRO PCA AND ETF RETURNS")
    print("=" * 70)

    merged = macro_pca.join(monthly_returns, how="inner")

    print("\nMerged data shape:")
    print(merged.shape)

    print("\nMerged date range:")
    print(f"Start: {merged.index.min()}")
    print(f"End:   {merged.index.max()}")

    print("\nFirst 5 rows:")
    print(merged.head())

    print("\nLast 5 rows:")
    print(merged.tail())

    merged_path = PROCESSED_DIR / "merged_data_paper_like.csv"
    merged.to_csv(merged_path)

    print(f"\nSaved merged paper-like data to: {merged_path}")

    return merged


# ============================================================
# 7. Main
# ============================================================

def main():
    print("=" * 70)
    print("STARTING PAPER-LIKE DATA PREPROCESSING")
    print("=" * 70)

    macro_pca = preprocess_fred_md_paper_like()
    monthly_returns = preprocess_etf_prices_paper_like()
    merged = merge_paper_like(macro_pca, monthly_returns)

    print("\n" + "=" * 70)
    print("PAPER-LIKE DATA PREPROCESSING COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")
    print("data/processed/macro_transformed_paper_like.csv")
    print("data/processed/macro_scaled_paper_like.csv")
    print("data/processed/macro_pca_paper_like.csv")
    print("data/processed/pca_info_paper_like.csv")
    print("data/processed/etf_monthly_returns_paper_like.csv")
    print("data/processed/merged_data_paper_like.csv")

    print("\nFinal merged paper-like data:")
    print(f"Shape: {merged.shape}")
    print(f"Date range: {merged.index.min()} to {merged.index.max()}")


if __name__ == "__main__":
    main()