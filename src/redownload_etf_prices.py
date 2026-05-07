from pathlib import Path
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ETF_TICKERS = [
    "SPY",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
]

START_DATE = "2000-01-01"
END_DATE = "2023-01-31"

OUTPUT_PATH = RAW_DIR / "etf_prices.csv"


def download_single_ticker(ticker: str) -> pd.Series:
    print(f"\nDownloading {ticker}...")

    data = yf.download(
        tickers=ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError(f"{ticker} returned empty data.")

    if "Close" not in data.columns:
        raise ValueError(f"{ticker} has no Close column.")

    close = data["Close"].copy()
    close.name = ticker

    missing_count = close.isna().sum()
    print(f"{ticker} downloaded. Rows: {len(close)}, Missing: {missing_count}")
    print(f"Date range: {close.index.min()} to {close.index.max()}")

    return close


def main():
    print("=" * 70)
    print("RE-DOWNLOADING ETF PRICES ONE BY ONE")
    print("=" * 70)

    all_prices = []

    for ticker in ETF_TICKERS:
        close = download_single_ticker(ticker)
        all_prices.append(close)

    prices = pd.concat(all_prices, axis=1)
    prices = prices.sort_index()
    prices.index.name = "date"

    print("\nCombined ETF price data:")
    print(prices.head())
    print(prices.tail())

    print("\nShape:")
    print(prices.shape)

    print("\nMissing values per ETF:")
    print(prices.isna().sum())

    # Check if any ticker is fully missing
    fully_missing = prices.columns[prices.isna().all()].tolist()
    if fully_missing:
        raise ValueError(f"These tickers are fully missing: {fully_missing}")

    prices.to_csv(OUTPUT_PATH)

    print("\nETF prices saved successfully.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()