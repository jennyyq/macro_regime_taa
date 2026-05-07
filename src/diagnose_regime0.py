from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MACRO_PCA_PATH = PROCESSED_DIR / "macro_pca_paper_like.csv"
REGIME_PATH = PROCESSED_DIR / "regime_labels_paper_like.csv"


def main():
    macro_pca = pd.read_csv(MACRO_PCA_PATH, index_col=0, parse_dates=True)
    regimes = pd.read_csv(REGIME_PATH, index_col=0, parse_dates=True)

    regime0_dates = regimes[regimes["regime"] == 0].index

    print("=" * 70)
    print("REGIME 0 DIAGNOSIS")
    print("=" * 70)

    print("\nRegime 0 dates:")
    print(regime0_dates)

    # Calculate Euclidean norm in PCA space
    pca_norm = np.linalg.norm(macro_pca.values, axis=1)
    norm_series = pd.Series(pca_norm, index=macro_pca.index, name="pca_norm")

    print("\nTop 20 most extreme months by PCA norm:")
    print(norm_series.sort_values(ascending=False).head(20))

    print("\nPCA values for Regime 0 month:")
    print(macro_pca.loc[regime0_dates])

    print("\nPCA norm for Regime 0 month:")
    print(norm_series.loc[regime0_dates])


if __name__ == "__main__":
    main()
    