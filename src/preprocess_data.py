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


# Paper sample setting
MACRO_END_DATE = "2023-01-01"
ETF_START_DATE = "2000-01-01"
ETF_END_DATE = "2023-01-31"


# ============================================================
# 2. FRED-MD transformation function
# ============================================================

def transform_series(series: pd.Series, tcode: int) -> pd.Series:
    """
    Apply FRED-MD transformation based on transformation code.

    Common FRED-MD transformation codes:
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
# 3. Preprocess FRED-MD
# ============================================================

def preprocess_fred_md():
    print("\n" + "=" * 70)
    print("PREPROCESSING FRED-MD MACRO DATA")
    print("=" * 70)

    fred_raw = pd.read_csv(FRED_PATH)

    print("\nRaw FRED-MD shape:")
    print(fred_raw.shape)

    # First row contains transformation codes
    transform_codes = fred_raw.iloc[0].copy()

    # Real macro data starts from row 1
    fred = fred_raw.iloc[1:].copy()

    # Rename date column
    fred = fred.rename(columns={"sasdate": "date"})

    # Convert date
    fred["date"] = pd.to_datetime(fred["date"])

    # Set date index
    fred = fred.set_index("date").sort_index()

    # Cut sample to paper period
    fred = fred.loc[:MACRO_END_DATE]

    print("\nFRED-MD after removing t-code row and cutting to 2023-01:")
    print(fred.shape)
    print(f"Date range: {fred.index.min()} to {fred.index.max()}")

    # Apply transformation code for each macro variable
    transformed = pd.DataFrame(index=fred.index)

    for col in fred.columns:
        tcode = int(float(transform_codes[col]))
        transformed[col] = transform_series(fred[col], tcode)

    # Replace inf with NaN
    transformed = transformed.replace([np.inf, -np.inf], np.nan)

    print("\nAfter transformation:")
    print(transformed.shape)

    print("\nMissing ratio before cleaning, top 10:")
    print(transformed.isna().mean().sort_values(ascending=False).head(10))

    # Drop columns with too many missing values
    # We use 20% threshold for the first practical replication.
    missing_threshold = 0.20
    transformed = transformed.loc[:, transformed.isna().mean() < missing_threshold]

    print("\nAfter dropping columns with >20% missing values:")
    print(transformed.shape)

    # Fill remaining missing values
    transformed = transformed.ffill().bfill()

    # Drop any remaining missing rows
    transformed = transformed.dropna()

    print("\nAfter filling and dropping remaining NaN:")
    print(transformed.shape)
    print(f"Date range: {transformed.index.min()} to {transformed.index.max()}")

    # Save transformed macro data
    transformed_path = PROCESSED_DIR / "macro_transformed.csv"
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

    scaled_path = PROCESSED_DIR / "macro_scaled.csv"
    macro_scaled.to_csv(scaled_path)
    print(f"Saved scaled macro data to: {scaled_path}")

    # PCA keeping 95% explained variance
    pca = PCA(n_components=0.95)
    pca_array = pca.fit_transform(macro_scaled)

    pc_columns = [f"PC{i+1}" for i in range(pca_array.shape[1])]

    macro_pca = pd.DataFrame(
        pca_array,
        index=macro_scaled.index,
        columns=pc_columns
    )

    pca_path = PROCESSED_DIR / "macro_pca.csv"
    macro_pca.to_csv(pca_path)

    print("\nPCA completed.")
    print(f"Number of PCs retained: {macro_pca.shape[1]}")
    print(f"Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Saved PCA macro data to: {pca_path}")

    return macro_pca


# ============================================================
# 4. Preprocess ETF prices
# ============================================================

def preprocess_etf_prices():
    print("\n" + "=" * 70)
    print("PREPROCESSING ETF PRICE DATA")
    print("=" * 70)

    prices = pd.read_csv(ETF_PATH, index_col=0, parse_dates=True)

    print("\nRaw ETF price shape:")
    print(prices.shape)
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")

    # Sort by date
    prices = prices.sort_index()

    # Use first available trading day of each month
    monthly_prices = prices.resample("MS").first()

    print("\nMonthly beginning-of-month prices:")
    print(monthly_prices.shape)
    print(monthly_prices.head())

    # Monthly return: price at current month start / previous month start - 1
    monthly_returns = monthly_prices.pct_change()

    # Drop first NaN row
    monthly_returns = monthly_returns.dropna()

    # Cut to paper-like ETF sample
    monthly_returns = monthly_returns.loc["2000-02-01":"2023-01-01"]

    print("\nMonthly ETF returns:")
    print(monthly_returns.shape)
    print(f"Date range: {monthly_returns.index.min()} to {monthly_returns.index.max()}")

    print("\nMissing values per ETF:")
    print(monthly_returns.isna().sum())

    returns_path = PROCESSED_DIR / "etf_monthly_returns.csv"
    monthly_returns.to_csv(returns_path)

    print(f"\nSaved ETF monthly returns to: {returns_path}")

    return monthly_returns


# ============================================================
# 5. Merge macro PCA and ETF returns
# ============================================================

def merge_macro_etf(macro_pca: pd.DataFrame, monthly_returns: pd.DataFrame):
    print("\n" + "=" * 70)
    print("MERGING MACRO PCA AND ETF RETURNS")
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

    merged_path = PROCESSED_DIR / "merged_data.csv"
    merged.to_csv(merged_path)

    print(f"\nSaved merged data to: {merged_path}")

    return merged


# ============================================================
# 6. Main
# ============================================================

def main():
    print("=" * 70)
    print("STARTING DATA PREPROCESSING")
    print("=" * 70)

    macro_pca = preprocess_fred_md()
    monthly_returns = preprocess_etf_prices()
    merged = merge_macro_etf(macro_pca, monthly_returns)

    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")
    print("data/processed/macro_transformed.csv")
    print("data/processed/macro_scaled.csv")
    print("data/processed/macro_pca.csv")
    print("data/processed/etf_monthly_returns.csv")
    print("data/processed/merged_data.csv")

    print("\nFinal merged data:")
    print(f"Shape: {merged.shape}")
    print(f"Date range: {merged.index.min()} to {merged.index.max()}")


if __name__ == "__main__":
    main()