from pathlib import Path
import pandas as pd
import yfinance as yf


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ETF tickers used in the paper (Table 2)
# ============================================================

ETF_TICKERS = [
    "SPY",  # S&P 500
    "XLB",  # Materials
    "XLE",  # Energy
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLK",  # Technology
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLV",  # Healthcare
    "XLY",  # Consumer Discretionary
]

# Paper uses Feb 2000 - Dec 2022 (Section 6)
START_DATE = "2000-01-01"
END_DATE   = "2023-01-01"

ETF_OUTPUT_PATH = RAW_DIR / "etf_prices.csv"


# ============================================================
# 3. Check manually downloaded FRED-MD
# ============================================================

def check_fred_md() -> Path:
    """
    Check whether FRED-MD has been manually downloaded.

    The direct St. Louis Fed link returned 403 in Python requests,
    so we manually download current.csv from the FRED-MD webpage,
    save it to data/raw/, and rename it to fred_md_current.csv.

    Expected file:
        data/raw/fred_md_current.csv
    """

    output_path = RAW_DIR / "fred_md_current.csv"

    print("\n[1/2] Checking FRED-MD file...")

    if not output_path.exists():
        raise FileNotFoundError(
            "\nFRED-MD file is missing.\n\n"
            "Please do this manually:\n"
            "1. Go to: https://research.stlouisfed.org/econ/mccracken/fred-databases/\n"
            "2. Download current.csv\n"
            "3. Put it into: data/raw/\n"
            "4. Rename it to: fred_md_current.csv\n\n"
            f"Expected path: {output_path}"
        )

    file_size_kb = output_path.stat().st_size / 1024

    print(f"  FRED-MD file found.")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size_kb:.2f} KB")

    if file_size_kb < 100:
        raise ValueError(
            "FRED-MD file looks too small. "
            "Please check whether the file was downloaded correctly."
        )

    # Quick preview check
    fred_preview = pd.read_csv(output_path, nrows=5)
    print("\n  FRED-MD preview:")
    print(fred_preview.head())
    print(f"\n  Preview shape: {fred_preview.shape}")

    return output_path


# ============================================================
# 4. Download ETF prices (one ticker at a time for reliability)
# ============================================================

def download_single_ticker(ticker: str) -> pd.Series:
    """
    Download adjusted close prices for one ETF ticker.

    Downloads one ticker at a time to avoid MultiIndex ambiguity
    and make it easier to debug per-ticker failures.

    Note: new yfinance returns MultiIndex even for single tickers,
    so we use .squeeze() to convert the Close column to a Series.
    """

    print(f"\n  Downloading {ticker}...")

    data = yf.download(
        tickers=ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        # threads parameter removed — deprecated in new yfinance
    )

    if data.empty:
        raise ValueError(f"{ticker} returned empty data.")

    if "Close" not in data.columns:
        raise ValueError(f"{ticker} has no Close column.")

    # New yfinance returns MultiIndex even for a single ticker,
    # so data["Close"] is a DataFrame — squeeze() converts it to a Series
    close = data["Close"].squeeze()
    close.name = ticker

    missing_count = close.isna().sum()
    print(f"  {ticker}: {len(close)} rows | missing: {missing_count} "
          f"| {close.index.min().date()} to {close.index.max().date()}")

    return close


def download_etf_prices() -> Path:
    """Download all 10 ETFs one by one and save to etf_prices.csv."""

    print("\n[2/2] Downloading ETF prices from Yahoo Finance...")
    print(f"  Tickers : {ETF_TICKERS}")
    print(f"  Start   : {START_DATE}")
    print(f"  End     : {END_DATE}")

    all_prices = []

    for ticker in ETF_TICKERS:
        close = download_single_ticker(ticker)
        all_prices.append(close)

    prices = pd.concat(all_prices, axis=1)
    prices = prices.sort_index()
    prices.index.name = "date"

    # Enforce paper ticker order
    prices = prices[ETF_TICKERS]

    print("\n  Combined ETF price data:")
    print(prices.head())
    print(prices.tail())
    print(f"\n  Shape: {prices.shape}")

    print("\n  Missing values per ETF:")
    print(prices.isna().sum())

    # Fail fast if any ticker is entirely missing
    fully_missing = prices.columns[prices.isna().all()].tolist()
    if fully_missing:
        raise ValueError(f"These tickers are fully missing: {fully_missing}")

    prices.to_csv(ETF_OUTPUT_PATH)

    print(f"\n  ETF prices saved to: {ETF_OUTPUT_PATH}")

    return ETF_OUTPUT_PATH


# ============================================================
# 5. Main
# ============================================================

def main():
    print("=" * 70)
    print("STARTING RAW DATA PREPARATION")
    print("=" * 70)

    fred_path = check_fred_md()
    etf_path  = download_etf_prices()

    print("\n" + "=" * 70)
    print("RAW DATA PREPARATION COMPLETED")
    print("=" * 70)

    print("\nFiles ready:")
    print(f"  FRED-MD   : {fred_path}")
    print(f"  ETF prices: {etf_path}")

    print("\nExpected files in data/raw/:")
    print("  data/raw/fred_md_current.csv")
    print("  data/raw/etf_prices.csv")


if __name__ == "__main__":
    main()