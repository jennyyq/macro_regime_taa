from pathlib import Path
import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

FRED_PATH = RAW_DIR / "fred_md_current.csv"
ETF_PATH = RAW_DIR / "etf_prices.csv"


EXPECTED_TICKERS = [
    "SPY", "XLB", "XLE", "XLF", "XLI",
    "XLK", "XLP", "XLU", "XLV", "XLY"
]


# ============================================================
# 2. Helper function
# ============================================================

def check_file_exists(path: Path):
    print("\nChecking file:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_size_kb = path.stat().st_size / 1024

    print("File exists.")
    print(f"File size: {file_size_kb:.2f} KB")

    if file_size_kb < 10:
        raise ValueError("File looks too small. Please check the download.")


# ============================================================
# 3. Check FRED-MD
# ============================================================

def check_fred_md():
    print("\n" + "=" * 70)
    print("CHECKING FRED-MD MACRO DATA")
    print("=" * 70)

    check_file_exists(FRED_PATH)

    # Read raw FRED-MD file
    fred = pd.read_csv(FRED_PATH)

    print("\nFRED-MD raw shape:")
    print(fred.shape)

    print("\nFirst 5 rows:")
    print(fred.head())

    print("\nFirst 10 columns:")
    print(fred.columns[:10].tolist())

    print("\nLast 5 rows:")
    print(fred.tail())

    first_col = fred.columns[0]
    print(f"\nFirst column name: {first_col}")

    print("\nImportant FRED-MD note:")
    print("The first data row usually contains transformation codes.")
    print("We will use these t-codes during preprocessing.")
    print("The actual macro data starts after that row.")


# ============================================================
# 4. Check ETF prices
# ============================================================

def check_etf_prices():
    print("\n" + "=" * 70)
    print("CHECKING ETF PRICE DATA")
    print("=" * 70)

    check_file_exists(ETF_PATH)

    etf = pd.read_csv(ETF_PATH, index_col=0, parse_dates=True)

    print("\nETF price shape:")
    print(etf.shape)

    print("\nETF columns:")
    print(etf.columns.tolist())

    print("\nETF date range:")
    print(f"Start: {etf.index.min()}")
    print(f"End:   {etf.index.max()}")

    missing_tickers = [ticker for ticker in EXPECTED_TICKERS if ticker not in etf.columns]

    if missing_tickers:
        print("\nMissing tickers:")
        print(missing_tickers)
    else:
        print("\nAll 10 expected ETF tickers are present.")

    print("\nMissing values per ETF:")
    print(etf.isna().sum())

    print("\nFirst 5 rows:")
    print(etf.head())

    print("\nLast 5 rows:")
    print(etf.tail())


# ============================================================
# 5. Main
# ============================================================

def main():
    print("=" * 70)
    print("STARTING RAW DATA CHECK")
    print("=" * 70)

    check_fred_md()
    check_etf_prices()

    print("\n" + "=" * 70)
    print("RAW DATA CHECK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()