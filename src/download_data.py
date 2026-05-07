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
# 2. ETF tickers used in the paper
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
            "1. Go to the St. Louis Fed FRED-MD webpage\n"
            "2. Download current.csv\n"
            "3. Put it into: data/raw/\n"
            "4. Rename it to: fred_md_current.csv\n\n"
            f"Expected path:\n{output_path}"
        )

    file_size_kb = output_path.stat().st_size / 1024

    print("FRED-MD file found.")
    print(f"File path: {output_path}")
    print(f"File size: {file_size_kb:.2f} KB")

    if file_size_kb < 100:
        raise ValueError(
            "FRED-MD file looks too small. "
            "Please check whether the file was downloaded correctly."
        )

    # Quick preview check
    fred_preview = pd.read_csv(output_path, nrows=5)

    print("\nFRED-MD preview:")
    print(fred_preview.head())
    print(f"\nFRED-MD preview shape: {fred_preview.shape}")

    return output_path


# ============================================================
# 4. Download ETF prices
# ============================================================

def download_etf_prices() -> Path:
    """
    Download daily adjusted close prices for the 10 ETFs.

    We use yfinance as a practical replacement for WRDS ETF data.
    Later, we will convert daily prices to monthly beginning-of-month returns.
    """

    output_path = RAW_DIR / "etf_prices.csv"

    print("\n[2/2] Downloading ETF prices from Yahoo Finance...")
    print(f"Tickers: {ETF_TICKERS}")

    data = yf.download(
        tickers=ETF_TICKERS,
        start="2000-01-01",
        end="2023-01-31",
        auto_adjust=True,
        group_by="column",
        progress=True,
        threads=True,
    )

    if data.empty:
        raise ValueError("ETF download returned empty data. Please rerun the script later.")

    # With auto_adjust=True, the Close price is adjusted.
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data["Close"].copy()
    else:
        close_prices = data[["Close"]].copy()

    close_prices = close_prices.sort_index()
    close_prices.index.name = "date"

    # Make sure columns are in the same order as the paper.
    close_prices = close_prices[ETF_TICKERS]

    close_prices.to_csv(output_path)

    print("\nETF prices downloaded successfully.")
    print(f"Saved to: {output_path}")
    print(f"Shape: {close_prices.shape}")
    print(f"Date range: {close_prices.index.min()} to {close_prices.index.max()}")

    print("\nETF price preview:")
    print(close_prices.head())

    print("\nMissing values per ETF:")
    print(close_prices.isna().sum())

    return output_path


# ============================================================
# 5. Main function
# ============================================================

def main():
    print("=" * 70)
    print("STARTING RAW DATA PREPARATION")
    print("=" * 70)

    fred_path = check_fred_md()
    etf_path = download_etf_prices()

    print("\n" + "=" * 70)
    print("RAW DATA PREPARATION COMPLETED")
    print("=" * 70)

    print("\nFiles ready:")
    print(f"FRED-MD: {fred_path}")
    print(f"ETF prices: {etf_path}")

    print("\nYou should now have:")
    print("data/raw/fred_md_current.csv")
    print("data/raw/etf_prices.csv")


if __name__ == "__main__":
    main()