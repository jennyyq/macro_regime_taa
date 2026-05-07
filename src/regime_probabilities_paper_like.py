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
TRANSITION_MATRIX_PATH = PROCESSED_DIR / "regime_transition_matrix_paper_like.csv"


# ============================================================
# 2. Settings
# ============================================================

RANDOM_STATE = 42

N_L2_CLUSTERS = 2
N_NORMAL_REGIMES = 5
N_TOTAL_REGIMES = 6

REGIME_COLUMNS = [f"Regime_{i}" for i in range(N_TOTAL_REGIMES)]
NEXT_REGIME_COLUMNS = [f"Next_Regime_{i}" for i in range(N_TOTAL_REGIMES)]


# ============================================================
# 3. Probability helper functions
# ============================================================

def distances_to_paper_probabilities(distances: np.ndarray) -> np.ndarray:
    """
    Convert centroid distances into probabilities.

    This follows the paper's core idea:
    - smaller distance to centroid -> higher regime probability
    - larger distance to centroid -> lower regime probability

    Paper-style score:
        score_i = 1 - d_i / sum_j(d_j)
        p_i = score_i / sum_m(score_m)

    We add numerical safeguards for extreme edge cases.
    """

    distances = np.asarray(distances, dtype=float)

    # Numerical safety
    distances = np.maximum(distances, 0.0)

    # If one distance is basically zero, assign probability 1 to that centroid
    eps = 1e-12
    if distances.min() < eps:
        probs = np.zeros_like(distances)
        probs[np.argmin(distances)] = 1.0
        return probs

    distance_sum = distances.sum()

    if distance_sum < eps:
        # If all distances are zero, use uniform probability
        return np.ones_like(distances) / len(distances)

    scores = 1.0 - distances / distance_sum

    # Avoid tiny negative values due to numerical issues
    scores = np.maximum(scores, 0.0)

    score_sum = scores.sum()

    if score_sum < eps:
        return np.ones_like(distances) / len(distances)

    probs = scores / score_sum

    return probs


def combine_l2_and_cosine_probabilities(p_regime0_l2: float,
                                        normal_probs: np.ndarray) -> np.ndarray:
    """
    Combine:
    - L2 outlier probability for Regime 0
    - Cosine KMeans probabilities for Regime 1-5

    This follows the logic in paper Section 3.2.2.

    Paper idea:
    1. Compute normal-regime distribution over Regime 1-5.
    2. Let Pmax = max probability among Regime 1-5.
    3. Scale Regime 0 probability using:
           PR0 = -Pmax * log2(1 - P(Regime 0))
    4. Normalize [PR0, Regime 1-5 probabilities] to sum to 1.
    """

    eps = 1e-12

    p0 = float(np.clip(p_regime0_l2, 0.0, 1.0 - eps))

    normal_probs = np.asarray(normal_probs, dtype=float)
    normal_probs = np.maximum(normal_probs, 0.0)

    # Normalize normal probabilities first
    normal_sum = normal_probs.sum()
    if normal_sum <= eps:
        normal_probs = np.ones_like(normal_probs) / len(normal_probs)
    else:
        normal_probs = normal_probs / normal_sum

    pmax = normal_probs.max()

    # Paper-style Regime 0 scaling
    pr0 = -pmax * np.log2(1.0 - p0)

    combined = np.concatenate([[pr0], normal_probs])

    combined_sum = combined.sum()

    if combined_sum <= eps:
        combined = np.ones(6) / 6
    else:
        combined = combined / combined_sum

    return combined


# ============================================================
# 4. Load data
# ============================================================

def load_inputs():
    print("=" * 70)
    print("LOADING INPUT DATA")
    print("=" * 70)

    if not MACRO_PCA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {MACRO_PCA_PATH}\n"
            "Please run: python src/preprocess_data_paper_like.py"
        )

    if not TRANSITION_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {TRANSITION_MATRIX_PATH}\n"
            "Please run: python src/detect_regimes_paper_like.py"
        )

    macro_pca = pd.read_csv(MACRO_PCA_PATH, index_col=0, parse_dates=True)
    transition_matrix = pd.read_csv(TRANSITION_MATRIX_PATH, index_col=0)

    transition_matrix.index = transition_matrix.index.astype(int)
    transition_matrix.columns = transition_matrix.columns.astype(int)

    print("\nMacro PCA:")
    print(f"Shape: {macro_pca.shape}")
    print(f"Date range: {macro_pca.index.min()} to {macro_pca.index.max()}")

    print("\nTransition matrix:")
    print(transition_matrix.round(4))

    print("\nTransition matrix row sums:")
    print(transition_matrix.sum(axis=1).round(6))

    if macro_pca.isna().sum().sum() > 0:
        raise ValueError("macro_pca_paper_like.csv contains missing values.")

    if transition_matrix.isna().sum().sum() > 0:
        raise ValueError("transition matrix contains missing values.")

    return macro_pca, transition_matrix


# ============================================================
# 5. Refit modified KMeans for probability calculation
# ============================================================

def fit_modified_kmeans(macro_pca: pd.DataFrame):
    """
    Refit the same two-layer modified KMeans structure:

    Layer 1:
        L2 KMeans, k=2.
        Smaller cluster is Regime 0 / outlier cluster.

    Layer 2:
        Cosine KMeans, k=5, applied to normal months.
        Implemented by row-normalizing vectors and using Euclidean KMeans.
    """

    print("\n" + "=" * 70)
    print("STEP 1: FIT TWO-LAYER MODIFIED KMEANS")
    print("=" * 70)

    # -------------------------------
    # Layer 1: L2 KMeans
    # -------------------------------

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

    l2_counts = l2_label_series.value_counts().sort_index()

    outlier_cluster = l2_counts.idxmin()
    normal_cluster = l2_counts.idxmax()

    is_outlier = l2_label_series == outlier_cluster

    print("\nL2 cluster counts:")
    print(l2_counts)

    print(f"\nOutlier cluster: {outlier_cluster}")
    print(f"Normal cluster: {normal_cluster}")

    # -------------------------------
    # Layer 2: Cosine KMeans
    # -------------------------------

    normal_data = macro_pca.loc[~is_outlier].copy()
    normal_values_unit = normalize(normal_data.values, norm="l2", axis=1)

    model_cosine = KMeans(
        n_clusters=N_NORMAL_REGIMES,
        random_state=RANDOM_STATE,
        n_init=50,
    )

    normal_labels_raw = model_cosine.fit_predict(normal_values_unit)

    normal_regimes = normal_labels_raw + 1

    normal_regime_series = pd.Series(
        normal_regimes,
        index=normal_data.index,
        name="normal_regime"
    )

    print("\nNormal regime counts:")
    print(normal_regime_series.value_counts().sort_index())

    # -------------------------------
    # Hard label sanity check
    # -------------------------------

    hard_regime = pd.Series(index=macro_pca.index, dtype=int, name="hard_regime")
    hard_regime.loc[is_outlier] = 0
    hard_regime.loc[normal_regime_series.index] = normal_regime_series
    hard_regime = hard_regime.astype(int)

    print("\nHard regime counts from refitted models:")
    print(hard_regime.value_counts().sort_index())

    return {
        "model_l2": model_l2,
        "model_cosine": model_cosine,
        "outlier_cluster": outlier_cluster,
        "normal_cluster": normal_cluster,
        "is_outlier": is_outlier,
        "hard_regime": hard_regime,
    }


# ============================================================
# 6. Compute current regime probabilities
# ============================================================

def compute_current_regime_probabilities(macro_pca: pd.DataFrame,
                                         fitted_models: dict) -> pd.DataFrame:
    """
    Compute current regime probabilities for every month.

    For each month:
    1. Compute L2 probability of belonging to outlier cluster.
    2. Compute cosine probability over normal regimes 1-5.
    3. Combine them using paper-style Regime 0 scaling.
    """

    print("\n" + "=" * 70)
    print("STEP 2: COMPUTE CURRENT REGIME PROBABILITIES")
    print("=" * 70)

    model_l2 = fitted_models["model_l2"]
    model_cosine = fitted_models["model_cosine"]
    outlier_cluster = fitted_models["outlier_cluster"]

    X = macro_pca.values

    l2_centroids = model_l2.cluster_centers_
    cosine_centroids = model_cosine.cluster_centers_

    probability_rows = []

    for x in X:
        # ---------------------------------------
        # Layer 1: L2 probabilities over 2 clusters
        # ---------------------------------------
        l2_distances = np.linalg.norm(l2_centroids - x, axis=1)
        l2_probs = distances_to_paper_probabilities(l2_distances)

        p_regime0_l2 = l2_probs[outlier_cluster]

        # ---------------------------------------
        # Layer 2: Cosine probabilities over Regime 1-5
        # ---------------------------------------
        x_unit = normalize(x.reshape(1, -1), norm="l2", axis=1)[0]

        cosine_distances = np.linalg.norm(cosine_centroids - x_unit, axis=1)
        normal_probs = distances_to_paper_probabilities(cosine_distances)

        # ---------------------------------------
        # Combine into final 6-regime distribution
        # ---------------------------------------
        combined_probs = combine_l2_and_cosine_probabilities(
            p_regime0_l2=p_regime0_l2,
            normal_probs=normal_probs
        )

        probability_rows.append(combined_probs)

    current_probs = pd.DataFrame(
        probability_rows,
        index=macro_pca.index,
        columns=REGIME_COLUMNS
    )

    print("\nCurrent regime probabilities:")
    print(f"Shape: {current_probs.shape}")

    print("\nFirst 5 rows:")
    print(current_probs.head())

    print("\nLast 5 rows:")
    print(current_probs.tail())

    print("\nRow sum check:")
    print(current_probs.sum(axis=1).describe())

    print("\nAverage regime probabilities:")
    print(current_probs.mean().round(4))

    output_path = PROCESSED_DIR / "regime_probabilities_paper_like.csv"
    current_probs.to_csv(output_path)

    print(f"\nSaved current regime probabilities to: {output_path}")

    return current_probs


# ============================================================
# 7. Compute next-month regime probabilities
# ============================================================

def compute_next_regime_probabilities(current_probs: pd.DataFrame,
                                      transition_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute next-month regime probabilities:

        p_{t+1} = p_t * transition_matrix

    This follows the paper's Markov-chain style update.
    """

    print("\n" + "=" * 70)
    print("STEP 3: COMPUTE NEXT-MONTH REGIME PROBABILITIES")
    print("=" * 70)

    E = transition_matrix.loc[range(N_TOTAL_REGIMES), range(N_TOTAL_REGIMES)].values

    P_current = current_probs.values
    P_next = P_current @ E

    next_probs = pd.DataFrame(
        P_next,
        index=current_probs.index,
        columns=NEXT_REGIME_COLUMNS
    )

    print("\nNext-month regime probabilities:")
    print(f"Shape: {next_probs.shape}")

    print("\nFirst 5 rows:")
    print(next_probs.head())

    print("\nLast 5 rows:")
    print(next_probs.tail())

    print("\nRow sum check:")
    print(next_probs.sum(axis=1).describe())

    print("\nAverage next-month regime probabilities:")
    print(next_probs.mean().round(4))

    output_path = PROCESSED_DIR / "next_regime_probabilities_paper_like.csv"
    next_probs.to_csv(output_path)

    print(f"\nSaved next-month regime probabilities to: {output_path}")

    return next_probs


# ============================================================
# 8. Sanity check
# ============================================================

def sanity_check_against_hard_labels(fitted_models: dict,
                                     current_probs: pd.DataFrame):
    """
    Compare the hard regime labels from the refitted KMeans models
    with the max-probability regime.

    They do not need to match perfectly because this is a soft probability model,
    but match rate should be informative.
    """

    print("\n" + "=" * 70)
    print("STEP 4: SANITY CHECK AGAINST HARD LABELS")
    print("=" * 70)

    hard_regime = fitted_models["hard_regime"]

    max_prob_regime = (
        current_probs
        .idxmax(axis=1)
        .str.replace("Regime_", "")
        .astype(int)
    )

    comparison = pd.DataFrame({
        "hard_regime": hard_regime,
        "max_probability_regime": max_prob_regime,
        "match": hard_regime == max_prob_regime
    })

    match_rate = comparison["match"].mean()

    print("\nMatch rate:")
    print(f"{match_rate:.4f}")

    print("\nComparison head:")
    print(comparison.head(10))

    print("\nMismatches head:")
    print(comparison[~comparison["match"]].head(10))

    output_path = PROCESSED_DIR / "regime_probability_hard_label_check_paper_like.csv"
    comparison.to_csv(output_path)

    print(f"\nSaved hard-label check to: {output_path}")

    return comparison


# ============================================================
# 9. Plots
# ============================================================

def plot_regime_probabilities(current_probs: pd.DataFrame):
    """
    Plot current regime probabilities over time.
    """

    output_path = OUTPUTS_DIR / "regime_probabilities_paper_like.png"

    plt.figure(figsize=(14, 6))

    for col in current_probs.columns:
        plt.plot(current_probs.index, current_probs[col], label=col, linewidth=1)

    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.title("Current Regime Probabilities")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved current regime probability plot to: {output_path}")


def plot_next_regime_probabilities(next_probs: pd.DataFrame):
    """
    Plot next-month regime probabilities over time.
    """

    output_path = OUTPUTS_DIR / "next_regime_probabilities_paper_like.png"

    plt.figure(figsize=(14, 6))

    for col in next_probs.columns:
        plt.plot(next_probs.index, next_probs[col], label=col, linewidth=1)

    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.title("Next-Month Regime Probabilities")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved next-month regime probability plot to: {output_path}")


# ============================================================
# 10. Main
# ============================================================

def main():
    print("=" * 70)
    print("COMPUTE REGIME PROBABILITY DISTRIBUTIONS")
    print("=" * 70)

    macro_pca, transition_matrix = load_inputs()

    fitted_models = fit_modified_kmeans(macro_pca)

    current_probs = compute_current_regime_probabilities(
        macro_pca=macro_pca,
        fitted_models=fitted_models
    )

    next_probs = compute_next_regime_probabilities(
        current_probs=current_probs,
        transition_matrix=transition_matrix
    )

    sanity_check_against_hard_labels(
        fitted_models=fitted_models,
        current_probs=current_probs
    )

    plot_regime_probabilities(current_probs)
    plot_next_regime_probabilities(next_probs)

    print("\n" + "=" * 70)
    print("REGIME PROBABILITY COMPUTATION COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")
    print("data/processed/regime_probabilities_paper_like.csv")
    print("data/processed/next_regime_probabilities_paper_like.csv")
    print("data/processed/regime_probability_hard_label_check_paper_like.csv")
    print("outputs/regime_probabilities_paper_like.png")
    print("outputs/next_regime_probabilities_paper_like.png")


if __name__ == "__main__":
    main()