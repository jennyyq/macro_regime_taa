from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MACRO_PCA_PATH = PROCESSED_DIR / "macro_pca_paper_like.csv"


# ============================================================
# 2. Settings
# ============================================================

RANDOM_STATE = 42

# Paper uses k = 2 in the first L2 KMeans layer
N_L2_CLUSTERS = 2

# Paper finds k = 5 for the second KMeans[Cosine] layer
N_NORMAL_REGIMES = 5

# Final regimes: 0, 1, 2, 3, 4, 5
N_TOTAL_REGIMES = N_NORMAL_REGIMES + 1


# ============================================================
# 3. Helper functions
# ============================================================

def load_macro_pca() -> pd.DataFrame:
    """
    Load paper-like macro PCA data.
    This should be generated from preprocess_data_paper_like.py.
    """

    if not MACRO_PCA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {MACRO_PCA_PATH}\n"
            "Please run: python src/preprocess_data_paper_like.py"
        )

    macro_pca = pd.read_csv(MACRO_PCA_PATH, index_col=0, parse_dates=True)

    print("\nLoaded macro PCA data:")
    print(f"Path: {MACRO_PCA_PATH}")
    print(f"Shape: {macro_pca.shape}")
    print(f"Date range: {macro_pca.index.min()} to {macro_pca.index.max()}")

    if macro_pca.isna().sum().sum() > 0:
        raise ValueError("macro_pca_paper_like.csv still contains missing values.")

    return macro_pca


def run_l2_kmeans_outlier_layer(macro_pca: pd.DataFrame):
    """
    First layer in the paper:
    KMeans with L2 distance and k=2.

    The smaller cluster is treated as abnormal/outlier months,
    and becomes Regime 0.
    """

    print("\n" + "=" * 70)
    print("STEP 1: L2 KMEANS, k=2, TO IDENTIFY OUTLIER MONTHS")
    print("=" * 70)

    model_l2 = KMeans(
        n_clusters=N_L2_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=50,
    )

    l2_labels = model_l2.fit_predict(macro_pca.values)

    l2_label_series = pd.Series(
        l2_labels,
        index=macro_pca.index,
        name="l2_cluster"
    )

    counts = l2_label_series.value_counts().sort_index()

    print("\nL2 cluster counts:")
    print(counts)

    # Paper says: if |A| <= |B|, A is outliers and B is typical months.
    outlier_cluster = counts.idxmin()
    normal_cluster = counts.idxmax()

    print(f"\nOutlier cluster selected as smaller cluster: {outlier_cluster}")
    print(f"Normal cluster selected as larger cluster: {normal_cluster}")

    is_outlier = l2_label_series == outlier_cluster

    print("\nNumber of outlier months:")
    print(int(is_outlier.sum()))

    print("\nNumber of normal months:")
    print(int((~is_outlier).sum()))

    return model_l2, l2_label_series, is_outlier


def run_cosine_kmeans_normal_layer(macro_pca: pd.DataFrame, is_outlier: pd.Series):
    """
    Second layer in the paper:
    On normal months only, run KMeans with cosine distance.

    sklearn KMeans does not directly support cosine distance.
    Common practical approximation:
    row-normalize each monthly state vector to unit length, then apply Euclidean KMeans.
    On unit-normalized vectors, Euclidean distance is monotonic with cosine distance.
    """

    print("\n" + "=" * 70)
    print("STEP 2: COSINE KMEANS, k=5, ON NORMAL MONTHS")
    print("=" * 70)

    normal_data = macro_pca.loc[~is_outlier].copy()

    print("\nNormal-month macro PCA data:")
    print(f"Shape: {normal_data.shape}")
    print(f"Date range: {normal_data.index.min()} to {normal_data.index.max()}")

    # Row-normalize monthly vectors
    normal_values = normal_data.values
    normal_values_unit = normalize(normal_values, norm="l2", axis=1)

    model_cosine = KMeans(
        n_clusters=N_NORMAL_REGIMES,
        random_state=RANDOM_STATE,
        n_init=50,
    )

    normal_labels_raw = model_cosine.fit_predict(normal_values_unit)

    # Convert labels 0-4 into Regime 1-5
    normal_regimes = normal_labels_raw + 1

    normal_regime_series = pd.Series(
        normal_regimes,
        index=normal_data.index,
        name="normal_regime"
    )

    print("\nNormal regime counts before combining with Regime 0:")
    print(normal_regime_series.value_counts().sort_index())

    return model_cosine, normal_regime_series


def combine_regime_labels(macro_pca: pd.DataFrame,
                          is_outlier: pd.Series,
                          normal_regime_series: pd.Series) -> pd.DataFrame:
    """
    Combine:
    - Outlier months => Regime 0
    - Normal months => Regime 1-5
    """

    print("\n" + "=" * 70)
    print("STEP 3: COMBINE REGIME LABELS")
    print("=" * 70)

    regime = pd.Series(index=macro_pca.index, dtype=int, name="regime")

    regime.loc[is_outlier] = 0
    regime.loc[normal_regime_series.index] = normal_regime_series

    regime = regime.astype(int)

    regime_df = pd.DataFrame({"regime": regime})

    print("\nFinal regime counts:")
    print(regime_df["regime"].value_counts().sort_index())

    print("\nFirst 10 regime labels:")
    print(regime_df.head(10))

    print("\nLast 10 regime labels:")
    print(regime_df.tail(10))

    # Safety checks
    expected_regimes = set(range(N_TOTAL_REGIMES))
    actual_regimes = set(regime_df["regime"].unique())

    print("\nActual regime set:")
    print(sorted(actual_regimes))

    if not actual_regimes.issubset(expected_regimes):
        raise ValueError(f"Unexpected regimes detected: {actual_regimes}")

    return regime_df


def compute_transition_matrix(regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transition matrix:
    e_ij = count(Regime i -> Regime j) / count(Regime i as current state)

    This follows PDF Section 3.3.
    """

    print("\n" + "=" * 70)
    print("STEP 4: COMPUTE REGIME TRANSITION MATRIX")
    print("=" * 70)

    regimes = regime_df["regime"].astype(int)

    current_regime = regimes.iloc[:-1].values
    next_regime = regimes.iloc[1:].values

    transition_counts = pd.DataFrame(
        0,
        index=range(N_TOTAL_REGIMES),
        columns=range(N_TOTAL_REGIMES),
        dtype=float
    )

    for i, j in zip(current_regime, next_regime):
        transition_counts.loc[i, j] += 1

    transition_matrix = transition_counts.div(
        transition_counts.sum(axis=1),
        axis=0
    )

    transition_matrix = transition_matrix.fillna(0.0)

    transition_matrix.index.name = "from_regime"
    transition_matrix.columns.name = "to_regime"

    print("\nTransition count matrix:")
    print(transition_counts)

    print("\nTransition probability matrix:")
    print(transition_matrix.round(4))

    print("\nRow sums, should be 1.0 except empty rows:")
    print(transition_matrix.sum(axis=1).round(4))

    return transition_counts, transition_matrix


def compute_normalized_transition_matrix(transition_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Paper also computes normalized transition matrix conditional on transition occurring.

    That means:
    - set diagonal to 0
    - divide each off-diagonal row by 1 - diagonal probability
    """

    print("\n" + "=" * 70)
    print("STEP 5: COMPUTE NORMALIZED TRANSITION MATRIX")
    print("=" * 70)

    normalized = transition_matrix.copy()

    for i in normalized.index:
        diag_prob = normalized.loc[i, i]

        # Remove self-transition
        normalized.loc[i, i] = 0.0

        denom = 1.0 - diag_prob

        if denom > 0:
            normalized.loc[i, :] = normalized.loc[i, :] / denom
        else:
            # If a regime never leaves itself, normalized row stays zero
            normalized.loc[i, :] = 0.0

    normalized.index.name = "from_regime"
    normalized.columns.name = "to_regime"

    print("\nNormalized transition matrix conditional on transition:")
    print(normalized.round(4))

    print("\nRow sums, should be 1.0 if transition is possible:")
    print(normalized.sum(axis=1).round(4))

    return normalized


def plot_regime_timeline(regime_df: pd.DataFrame):
    """
    Plot regime label over time.
    Similar idea to PDF Figure 2.
    """

    output_path = OUTPUTS_DIR / "regime_timeline_paper_like.png"

    plt.figure(figsize=(14, 4))
    plt.scatter(
        regime_df.index,
        regime_df["regime"],
        s=12
    )
    plt.yticks(range(N_TOTAL_REGIMES))
    plt.xlabel("Date")
    plt.ylabel("Regime")
    plt.title("Detected Macroeconomic Regimes - Paper-like Modified KMeans")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved regime timeline plot to: {output_path}")


def plot_transition_matrix(matrix: pd.DataFrame, title: str, output_name: str):
    """
    Plot transition matrix as heatmap.
    Similar idea to PDF Figure 5.
    """

    output_path = OUTPUTS_DIR / output_name

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix.values, aspect="auto")
    plt.colorbar(label="Probability")
    plt.xticks(range(N_TOTAL_REGIMES), matrix.columns)
    plt.yticks(range(N_TOTAL_REGIMES), matrix.index)
    plt.xlabel("To Regime")
    plt.ylabel("From Regime")
    plt.title(title)

    # Add text labels
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved transition matrix plot to: {output_path}")


def save_outputs(regime_df: pd.DataFrame,
                 transition_counts: pd.DataFrame,
                 transition_matrix: pd.DataFrame,
                 normalized_transition_matrix: pd.DataFrame):
    """
    Save all outputs to data/processed and outputs folders.
    """

    print("\n" + "=" * 70)
    print("STEP 6: SAVE OUTPUTS")
    print("=" * 70)

    regime_path = PROCESSED_DIR / "regime_labels_paper_like.csv"
    transition_count_path = PROCESSED_DIR / "regime_transition_counts_paper_like.csv"
    transition_matrix_path = PROCESSED_DIR / "regime_transition_matrix_paper_like.csv"
    normalized_matrix_path = PROCESSED_DIR / "regime_transition_matrix_normalized_paper_like.csv"

    regime_df.to_csv(regime_path)
    transition_counts.to_csv(transition_count_path)
    transition_matrix.to_csv(transition_matrix_path)
    normalized_transition_matrix.to_csv(normalized_matrix_path)

    counts = regime_df["regime"].value_counts().sort_index()
    counts_path = OUTPUTS_DIR / "regime_counts_paper_like.csv"
    counts.to_csv(counts_path, header=["count"])

    print(f"Saved regime labels to: {regime_path}")
    print(f"Saved transition counts to: {transition_count_path}")
    print(f"Saved transition matrix to: {transition_matrix_path}")
    print(f"Saved normalized transition matrix to: {normalized_matrix_path}")
    print(f"Saved regime counts to: {counts_path}")

    plot_regime_timeline(regime_df)

    plot_transition_matrix(
        transition_matrix,
        title="Regime Transition Matrix",
        output_name="regime_transition_matrix_paper_like.png"
    )

    plot_transition_matrix(
        normalized_transition_matrix,
        title="Normalized Regime Transition Matrix | Transition Occurs",
        output_name="regime_transition_matrix_normalized_paper_like.png"
    )


# ============================================================
# 4. Main
# ============================================================

def main():
    print("=" * 70)
    print("DETECT ECONOMIC REGIMES USING MODIFIED KMEANS")
    print("=" * 70)

    macro_pca = load_macro_pca()

    model_l2, l2_labels, is_outlier = run_l2_kmeans_outlier_layer(macro_pca)

    model_cosine, normal_regime_series = run_cosine_kmeans_normal_layer(
        macro_pca,
        is_outlier
    )

    regime_df = combine_regime_labels(
        macro_pca,
        is_outlier,
        normal_regime_series
    )

    transition_counts, transition_matrix = compute_transition_matrix(regime_df)

    normalized_transition_matrix = compute_normalized_transition_matrix(
        transition_matrix
    )

    save_outputs(
        regime_df,
        transition_counts,
        transition_matrix,
        normalized_transition_matrix
    )

    print("\n" + "=" * 70)
    print("REGIME DETECTION COMPLETED")
    print("=" * 70)

    print("\nImportant output files:")
    print("data/processed/regime_labels_paper_like.csv")
    print("data/processed/regime_transition_matrix_paper_like.csv")
    print("data/processed/regime_transition_matrix_normalized_paper_like.csv")
    print("outputs/regime_timeline_paper_like.png")
    print("outputs/regime_transition_matrix_paper_like.png")
    print("outputs/regime_transition_matrix_normalized_paper_like.png")


if __name__ == "__main__":
    main()