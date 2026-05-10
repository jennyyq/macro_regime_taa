"""
detect_regimes.py
=================
Pipeline (Algorithm 1, Sections 3.1-3.3):
  1.  L2 KMeans k=2       -> Regime 0 (outlier) vs normal months
  2.  Cosine KMeans k=5   -> 5 normal regimes
  3.  Hungarian matching  -> stable regime labels across windows
  4.  Eq.1                -> centroid-distance probabilities
  5.  Eqs.3-4             -> combine L2 + cosine probs
  6.  Eq.5                -> transition matrix
  7.  Eq.7                -> next-month regime forecast
  8.  Figures 2-5, Table 1

Outputs (outputs/):
  figure2_kmeans_regimes.png        (K-Means + GMM panels)
  figure3_crisis_probabilities.png  (K-Means + GMM panels)
  figure4_regime_stats.png
  figure5_transition_matrices.png

  data/processed/:
  gmm_regime_labels_paper_like.csv
  gmm_regime_probabilities_paper_like.csv

Outputs (data/processed/):
  regime_labels_paper_like.csv
  regime_probabilities_paper_like.csv
  next_regime_probabilities_paper_like.csv
  regime_transition_matrix_paper_like.csv
  regime_transition_matrix_normalized_paper_like.csv
  table1_regime_labels.csv
  l2_centroids.npy  /  cosine_centroids.npy
  ref_l2_centroids.npy  /  ref_cosine_centroids.npy
  l2_outlier_cluster_id.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MACRO_PCA_PATH  = PROCESSED_DIR / "macro_pca_paper_like.csv"
FRED_RAW_PATH   = RAW_DIR / "fred_md_current.csv"
FRED_TRANS_PATH = PROCESSED_DIR / "macro_transformed_paper_like.csv"

REF_L2_CENTROIDS_PATH     = PROCESSED_DIR / "ref_l2_centroids.npy"
REF_COSINE_CENTROIDS_PATH = PROCESSED_DIR / "ref_cosine_centroids.npy"
REF_L2_OUTLIER_ID_PATH    = PROCESSED_DIR / "ref_l2_outlier_cluster_id.json"


# ============================================================
# 2. Settings
# ============================================================

RANDOM_STATE     = 42
N_INIT           = 20
N_L2_CLUSTERS    = 2
N_NORMAL_REGIMES = 5
N_TOTAL_REGIMES  = N_NORMAL_REGIMES + 1    # 0..5

REGIME_COLS      = [f"Regime_{i}"      for i in range(N_TOTAL_REGIMES)]
NEXT_REGIME_COLS = [f"Next_Regime_{i}" for i in range(N_TOTAL_REGIMES)]

# Key FRED-MD variables for Figure 4
KEY_VARS = {
    "RPI"      : ["RPI"],
    "UNRATE"   : ["UNRATE"],
    "UMCSENTx" : ["UMCSENTx", "UMCSENT"],
    "FEDFUNDS" : ["FEDFUNDS"],
    "CPIAUCSL" : ["CPIAUCSL"],
    "S&P 500"  : ["S&P 500", "SP500"],
}

# Table 1 — regime labels (Section 4.6)
# Verified against Figure 4 min-max normalised FRED-MD statistics:
#
#   Regime 0: UNRATE=1.00 (highest), UMCSENTx=0.00 (lowest),
#             FEDFUNDS=0.02 (lowest), CPIAUCSL=1.00 (highest)
#             -> Economic Difficulty  (crisis months: 2008, 2020 etc.)
#
#   Regime 1: FEDFUNDS=1.00 (highest), RPI=0.07 (lowest),
#             UMCSENTx=0.20 (low), CPIAUCSL=0.10 (low)
#             -> Stagflationary Pressure  (high rates crushing income)
#
#   Regime 2: FEDFUNDS=0.00 (lowest), UNRATE=0.73 (high),
#             CPIAUCSL=0.90 (high)
#             -> Economic Recovery  (QE era: zero rates, labour weak)
#
#   Regime 3: UNRATE=0.00 (lowest), UMCSENTx=1.00 (highest),
#             FEDFUNDS=0.46 (mid)
#             -> Expansionary Growth  (full employment, peak sentiment)
#
#   Regime 4: UNRATE=0.11 (low), UMCSENTx=0.96 (high),
#             FEDFUNDS=0.65 (rising)
#             -> Pre-Recession Transition  (strong economy, tightening)
#
#   Regime 5: RPI=0.00 (lowest), CPIAUCSL=0.00 (lowest),
#             UMCSENTx=1.00 (highest), FEDFUNDS=0.73 (high)
#             -> Reflationary Boom  (disinflation + strong sentiment)
REGIME_LABELS = {
    0: "Economic Difficulty",
    1: "Stagflationary Pressure",
    2: "Economic Recovery",
    3: "Expansionary Growth",
    4: "Pre-Recession Transition",
    5: "Reflationary Boom",
}

# Colors matching paper Figure 2 scatter style
REGIME_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]

# NBER recession shading dates (approximate)
NBER_RECESSIONS = [
    ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


# ============================================================
# 3. Load
# ============================================================

def load_macro_pca() -> pd.DataFrame:
    if not MACRO_PCA_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {MACRO_PCA_PATH}\nRun preprocess_data.py first."
        )
    macro_pca = pd.read_csv(MACRO_PCA_PATH, index_col=0, parse_dates=True)
    print(f"Loaded macro PCA: {macro_pca.shape}  "
          f"{macro_pca.index.min().date()} -> {macro_pca.index.max().date()}")
    if macro_pca.isna().sum().sum() > 0:
        raise ValueError("macro_pca contains NaNs — rerun preprocess_data.py.")
    return macro_pca


# ============================================================
# 4. Layer 1 — L2 KMeans  (Section 3.1.1)
# ============================================================

def run_l2_kmeans(X: np.ndarray):
    """
    Identify outlier months (Regime 0) via norm threshold (Section 3.1.1).

    COVID (2020-04) has PCA row-norm ~101 vs mean ~8. Naive k=2 always
    isolates it as the only Regime 0 month, missing 2008 and all other
    crises. Fix: flag months with norm > p97 as outliers directly, then
    fit k=2 on norm-capped X only to obtain two centroids for Eq.1.

    p97 => ~23 months (~3% of 758), covering major historical crises.
    """
    row_norms = np.linalg.norm(X, axis=1)
    norm_p97  = np.percentile(row_norms, 97)
    is_outlier = row_norms > norm_p97
    n_outlier  = int(is_outlier.sum())
    print(f"  Norm p97 threshold : {norm_p97:.2f}  "
          f"({n_outlier} outlier months, max norm={row_norms.max():.2f})")

    # Fit k=2 on norm-capped data for centroids (Eq.1 needs two centroids)
    scale    = np.where(row_norms > norm_p97, norm_p97 / row_norms, 1.0)
    X_capped = X * scale[:, np.newaxis]
    model    = KMeans(n_clusters=N_L2_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
    model.fit(X_capped)

    # Outlier centroid = higher norm centroid
    outlier_raw_id = int(np.argmax(np.linalg.norm(model.cluster_centers_, axis=1)))

    return model, outlier_raw_id, is_outlier, norm_p97


# ============================================================
# 5. Layer 2 — Cosine KMeans  (Section 3.1.2)
# ============================================================

def run_cosine_kmeans(X: np.ndarray, is_outlier: np.ndarray):
    """k=5 cosine KMeans on L2-normalised normal months."""
    X_normal = X[~is_outlier]
    X_unit   = normalize(X_normal, norm="l2", axis=1)
    model    = KMeans(n_clusters=N_NORMAL_REGIMES,
                      random_state=RANDOM_STATE, n_init=N_INIT)
    model.fit(X_unit)
    return model, model.cluster_centers_    # (5, n_pca)


# ============================================================
# 6. Hungarian label matching
# ============================================================

def build_reference_ordering(l2_centroids: np.ndarray,
                              outlier_raw_id: int,
                              cosine_centroids_raw: np.ndarray) -> np.ndarray:
    """
    Canonical Regime 1-5 ordering: sort cosine centroids by cosine distance
    to the crisis centroid. Most-similar-to-crisis = Regime 1.
    """
    crisis_unit  = normalize(l2_centroids[outlier_raw_id].reshape(1, -1), norm="l2")
    dists        = cosine_distances(
        normalize(cosine_centroids_raw, norm="l2"), crisis_unit
    ).ravel()
    order     = np.argsort(dists)
    reordered = cosine_centroids_raw[order]
    print("  Reference ordering (raw -> regime):")
    for new_i, raw_i in enumerate(order):
        print(f"    raw cluster {raw_i}  ->  Regime {new_i + 1}  "
              f"(cosine dist to crisis: {dists[raw_i]:.4f})")
    return reordered


def hungarian_match(ref_centroids: np.ndarray,
                    new_centroids: np.ndarray) -> np.ndarray:
    """Match new_centroids to ref_centroids. Returns permutation array."""
    cost       = cosine_distances(normalize(new_centroids, norm="l2"),
                                  normalize(ref_centroids, norm="l2"))
    _, col_ind = linear_sum_assignment(cost)
    return col_ind


def apply_label_mapping(raw_labels: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return np.array([perm[r] + 1 for r in raw_labels])


# ============================================================
# 7. Full-history fit
# ============================================================

def fit_full_history(macro_pca: pd.DataFrame):
    print("\n" + "=" * 60)
    print("FULL-HISTORY FIT")
    print("=" * 60)

    X = macro_pca.values.astype(float)

    print("\n  STEP 1: L2 KMeans k=2")
    l2_model, outlier_raw_id, is_outlier_arr, norm_p97 = run_l2_kmeans(X)
    is_outlier = pd.Series(is_outlier_arr, index=macro_pca.index)
    print(f"  L2 cluster counts  : {np.bincount(l2_model.labels_)}")

    print("\n  STEP 2: Cosine KMeans k=5")
    cosine_model, cosine_centroids_raw = run_cosine_kmeans(X, is_outlier_arr)
    raw_labels_normal = cosine_model.labels_

    print("\n  STEP 3: Build reference ordering")
    ref_cosine_centroids = build_reference_ordering(
        l2_model.cluster_centers_, outlier_raw_id, cosine_centroids_raw
    )

    perm          = hungarian_match(ref_cosine_centroids, cosine_centroids_raw)
    stable_normal = apply_label_mapping(raw_labels_normal, perm)

    # Save reference centroids
    np.save(REF_L2_CENTROIDS_PATH,     l2_model.cluster_centers_)
    np.save(REF_COSINE_CENTROIDS_PATH, ref_cosine_centroids)
    with open(REF_L2_OUTLIER_ID_PATH, "w") as f:
        json.dump({"outlier_cluster_id": int(outlier_raw_id)}, f)
    print("\n  Reference centroids saved.")

    normal_idx = macro_pca.index[~is_outlier_arr]
    regime     = pd.Series(np.nan, index=macro_pca.index)
    regime.loc[is_outlier]    = 0
    regime.loc[normal_idx]    = stable_normal
    regime = regime.astype(int)

    print("\n  Final regime counts (full history):")
    print(regime.value_counts().sort_index().to_string())

    return regime, l2_model, outlier_raw_id, ref_cosine_centroids, norm_p97


# ============================================================
# 8. Walk-forward window fit (for backtest use)
# ============================================================

def fit_window(X_window: np.ndarray,
               ref_cosine_centroids: np.ndarray,
               ref_l2_centroids: np.ndarray,
               ref_outlier_raw_id: int):
    """
    Fit one rolling window and return matched centroids.
    Used by the backtest script — not called in main pipeline.
    """
    l2_model, outlier_raw_id, is_outlier_arr, _ = run_l2_kmeans(X_window)

    # Re-match outlier cluster to reference
    ref_out = ref_l2_centroids[ref_outlier_raw_id].reshape(1, -1)
    dists   = np.linalg.norm(l2_model.cluster_centers_ - ref_out, axis=1)
    outlier_raw_id = int(np.argmin(dists))
    is_outlier_arr = l2_model.labels_ == outlier_raw_id

    _, cosine_centroids_raw = run_cosine_kmeans(X_window, is_outlier_arr)
    perm              = hungarian_match(ref_cosine_centroids, cosine_centroids_raw)
    matched_centroids = cosine_centroids_raw[np.argsort(perm)]

    return l2_model, outlier_raw_id, matched_centroids


# ============================================================
# 9. Eq.1 — centroid-distance -> probabilities
# ============================================================

def distances_to_probs(distances: np.ndarray) -> np.ndarray:
    """
    Paper Eq.(1):
        P(C_i) = (1 - d_i/sum_j d_j) / sum_m(1 - d_m/sum_j d_j)
    """
    distances  = np.maximum(np.asarray(distances, dtype=float), 0.0)
    total      = distances.sum()
    if total < 1e-12:
        return np.ones(len(distances)) / len(distances)
    numerators = 1.0 - distances / total
    denom      = numerators.sum()
    if denom < 1e-12:
        return np.ones(len(distances)) / len(distances)
    return numerators / denom


# ============================================================
# 10. Eqs.3-4 — combine L2 and cosine probabilities
# ============================================================

def combine_probabilities(p_regime0: float,
                          normal_probs: np.ndarray) -> np.ndarray:
    """
    Paper Eqs.(3-4):
        P_max = max(P(Regime 1..r))
        PR0   = -P_max * log2(1 - P(Regime 0))
    Concatenate [PR0, P(R1)..P(R5)] then renormalise.
    """
    eps      = 1e-12
    p0       = float(np.clip(p_regime0, 0.0, 1.0 - eps))
    pmax     = float(np.asarray(normal_probs).max())
    pr0      = -pmax * np.log2(1.0 - p0)
    combined = np.concatenate([[pr0], normal_probs])
    s        = combined.sum()
    if s < eps:
        return np.ones(N_TOTAL_REGIMES) / N_TOTAL_REGIMES
    return combined / s


# ============================================================
# 11. Current-month probabilities  (Eq.1 + Eqs.3-4)
# ============================================================

def compute_current_probs(macro_pca: pd.DataFrame,
                          norm_p97: float,
                          ref_cosine_centroids: np.ndarray) -> pd.DataFrame:
    """
    P(Regime 0): sigmoid of (row_norm / norm_p97) — consistent with the
    norm-threshold definition of is_outlier, avoids centroid mismatch.

    P(Regime 1-5): cosine distances to reference centroids via Eq.1.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Current-month probabilities (Eq.1 + Eqs.3-4)")
    print("=" * 60)

    X         = macro_pca.values
    row_norms = np.linalg.norm(X, axis=1)
    rows      = []

    for i, x in enumerate(X):
        norm_ratio = row_norms[i] / norm_p97
        p0         = float(1.0 / (1.0 + np.exp(-6.0 * (norm_ratio - 1.0))))

        x_unit       = normalize(x.reshape(1, -1), norm="l2")
        cos_dists    = cosine_distances(x_unit, ref_cosine_centroids)[0]
        normal_probs = distances_to_probs(cos_dists)
        rows.append(combine_probabilities(p0, normal_probs))

    current_probs = pd.DataFrame(rows, index=macro_pca.index, columns=REGIME_COLS)
    print(f"  Shape              : {current_probs.shape}")
    print(f"  Row-sum min/max    : {current_probs.sum(axis=1).min():.6f} / "
          f"{current_probs.sum(axis=1).max():.6f}")
    print(f"  Mean prob/regime   :\n{current_probs.mean().round(4).to_string()}")
    return current_probs


# ============================================================
# 12. Eq.7 — next-month forecast
# ============================================================

def compute_next_probs(current_probs: pd.DataFrame,
                       trans: pd.DataFrame) -> pd.DataFrame:
    """
    Paper Eq.(7): p̃_{t+1} = p̃_t^T E
    Row t = forecast made AT t FOR t+1.
    In backtest: shift(1) to align with return dates.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Next-month probabilities (Eq.7)")
    print("=" * 60)

    E      = trans.loc[range(N_TOTAL_REGIMES),
                       range(N_TOTAL_REGIMES)].values.astype(float)
    P_next = current_probs.values @ E
    next_probs = pd.DataFrame(P_next, index=current_probs.index,
                              columns=NEXT_REGIME_COLS)
    print(f"  Shape              : {next_probs.shape}")
    print(f"  Row-sum min/max    : {next_probs.sum(axis=1).min():.6f} / "
          f"{next_probs.sum(axis=1).max():.6f}")
    print(f"  NOTE: row t = forecast made at t for t+1.")
    return next_probs


# ============================================================
# 13. Transition matrix  (Eq.5)
# ============================================================

def compute_transition_matrix(regime: pd.Series):
    """Eq.(5): e_ij = count(i->j) / count(i)."""
    print("\n" + "=" * 60)
    print("STEP 6: Transition matrix (Eq.5)")
    print("=" * 60)

    current = regime.iloc[:-1].values
    nxt     = regime.iloc[1:].values
    counts  = pd.crosstab(pd.Series(current, name="from"),
                          pd.Series(nxt,     name="to"))
    all_r   = list(range(N_TOTAL_REGIMES))
    counts  = counts.reindex(index=all_r, columns=all_r, fill_value=0).astype(float)
    trans   = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

    print("  Transition matrix (rounded):")
    print(trans.round(3).to_string())

    norm = trans.copy()
    for i in norm.index:
        d              = norm.loc[i, i]
        norm.loc[i, i] = 0.0
        denom          = 1.0 - d
        norm.loc[i]    = norm.loc[i] / denom if denom > 1e-12 else 0.0

    return counts, trans, norm


# ============================================================
# 14. Sanity check
# ============================================================

def sanity_check(current_probs: pd.DataFrame, regime: pd.Series):
    print("\n" + "=" * 60)
    print("SANITY CHECK: soft-argmax vs hard labels")
    print("=" * 60)
    soft   = (current_probs.idxmax(axis=1)
              .str.replace("Regime_", "", regex=False).astype(int))
    common = regime.index.intersection(soft.index)
    match  = (regime.loc[common] == soft.loc[common]).mean()
    print(f"  Hard vs soft-argmax agreement : {match:.4f}")
    if match < 0.80:
        print("  WARNING: match below 0.80")
    else:
        print("  OK")


# ============================================================
# 15. NBER shading helper
# ============================================================

def _shade_recessions(ax, alpha: float = 0.15):
    for start, end in NBER_RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="grey", alpha=alpha, zorder=0)



# ============================================================
# 15b. GMM — fit and get labels + probabilities  (Section 4.3)
# ============================================================

def run_gmm(macro_pca: pd.DataFrame,
            is_outlier: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    """
    Fit a Gaussian Mixture Model (GMM) with N_NORMAL_REGIMES components
    on the normal months (non-outlier), then assign Regime 0 to outlier
    months.  Used only for comparison with K-Means (Figure 2, Figure 3).
    NOT used in backtest.

    GMM is inherently probabilistic and outputs near-binary probabilities
    (high confidence) — contrasting with the smoother K-Means probabilities.
    This matches the paper description in Section 4.3 and Figure 3.

    Returns:
        gmm_regime  : pd.Series of hard labels (0-5)
        gmm_probs   : pd.DataFrame (T x 6) probability matrix
    """
    print("\n" + "=" * 60)
    print("GMM: fitting for comparison (Section 4.3)")
    print("=" * 60)

    X         = macro_pca.values.astype(float)
    X_normal  = X[~is_outlier]

    gmm = GaussianMixture(
        n_components=N_NORMAL_REGIMES,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=5,
        max_iter=200,
    )
    gmm.fit(X_normal)

    # Hard labels for normal months (0-indexed, will become 1-5)
    raw_labels_normal = gmm.predict(X_normal)
    # Probabilities for normal months
    raw_probs_normal  = gmm.predict_proba(X_normal)   # (n_normal x 5)

    # Build full regime series: 0 = outlier, 1-5 = GMM clusters
    normal_idx = macro_pca.index[~is_outlier]
    gmm_regime = pd.Series(np.nan, index=macro_pca.index)
    gmm_regime.loc[macro_pca.index[is_outlier]] = 0
    gmm_regime.loc[normal_idx]                  = raw_labels_normal + 1
    gmm_regime = gmm_regime.astype(int)

    print("  GMM regime counts:")
    print(gmm_regime.value_counts().sort_index().to_string())

    # Build full probability matrix: col 0 = P(regime 0)
    # For outlier months: P(regime 0) = 1.0
    # For normal months: use GMM posterior probabilities
    probs_array = np.zeros((len(macro_pca), N_TOTAL_REGIMES))

    outlier_idx_bool = is_outlier
    probs_array[outlier_idx_bool, 0] = 1.0

    normal_positions = np.where(~is_outlier)[0]
    probs_array[normal_positions, 1:] = raw_probs_normal

    gmm_probs = pd.DataFrame(
        probs_array,
        index=macro_pca.index,
        columns=[f"GMM_Regime_{i}" for i in range(N_TOTAL_REGIMES)],
    )

    gmm_regime.to_frame("regime").to_csv(
        PROCESSED_DIR / "gmm_regime_labels_paper_like.csv"
    )
    gmm_probs.to_csv(PROCESSED_DIR / "gmm_regime_probabilities_paper_like.csv")
    print("  Saved: gmm_regime_labels_paper_like.csv")
    print("  Saved: gmm_regime_probabilities_paper_like.csv")

    return gmm_regime, gmm_probs


# ============================================================
# 16. Figure 2 — K-Means vs GMM Regime Timeline  (paper Figure 2)
# ============================================================

def plot_figure2(kmeans_regime: pd.Series, gmm_regime: pd.Series):
    """
    Two-panel scatter plot (K-Means top, GMM bottom) of regime labels
    over time.  NBER recessions shaded grey.  Each regime its own colour.
    Exactly matches paper Figure 2 layout.
    Saved as: outputs/figure2_kmeans_regimes.png
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for ax, regime, ylabel in zip(
        axes,
        [kmeans_regime, gmm_regime],
        ["K-Means Regimes", "GMM Regimes"],
    ):
        _shade_recessions(ax)
        for r in range(N_TOTAL_REGIMES):
            mask = regime == r
            ax.scatter(
                regime.index[mask], regime.values[mask],
                color=REGIME_COLORS[r], s=12, zorder=3, label=str(r),
            )
        ax.set_yticks(range(N_TOTAL_REGIMES))
        ax.set_ylabel(ylabel)

        handles = [mpatches.Patch(color=REGIME_COLORS[r], label=str(r))
                   for r in range(N_TOTAL_REGIMES)]
        ax.legend(handles=handles, loc="upper right",
                  fontsize=7, ncol=1, framealpha=0.7)

    axes[1].set_xlabel("Date")
    axes[0].set_title("K-Means vs GMM Regimes")
    fig.tight_layout()
    out = OUTPUTS_DIR / "figure2_kmeans_regimes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ============================================================
# 17. Figure 3 — K-Means vs GMM Crisis Probabilities  (paper Figure 3)
# ============================================================

def plot_figure3(kmeans_probs: pd.DataFrame, gmm_probs: pd.DataFrame):
    """
    Two-panel plot:
      Top    — K-Means P(crisis) and P(BAU) over time (smooth transitions)
      Bottom — GMM     P(crisis) and P(BAU) over time (near-binary)
    NBER recessions shaded grey.
    Exactly matches paper Figure 3 layout.
    Saved as: outputs/figure3_crisis_probabilities.png
    """
    # K-Means crisis/BAU
    km_crisis = kmeans_probs["Regime_0"]
    km_bau    = kmeans_probs.drop(columns=["Regime_0"]).sum(axis=1)

    # GMM crisis/BAU
    gmm_crisis = gmm_probs["GMM_Regime_0"]
    gmm_bau    = gmm_probs.drop(columns=["GMM_Regime_0"]).sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top: K-Means
    ax = axes[0]
    _shade_recessions(ax)
    ax.scatter(km_crisis.index, km_crisis.values,
               color="red", s=5, zorder=3, label="Crisis")
    ax.scatter(km_bau.index, km_bau.values,
               color="steelblue", s=5, zorder=3, alpha=0.7, label="BAU")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("K-Means Prob")
    ax.legend(loc="lower right", fontsize=8)

    # Bottom: GMM
    ax = axes[1]
    _shade_recessions(ax)
    ax.scatter(gmm_crisis.index, gmm_crisis.values,
               color="red", s=5, zorder=3, label="Crisis")
    ax.scatter(gmm_bau.index, gmm_bau.values,
               color="steelblue", s=5, zorder=3, alpha=0.7, label="BAU")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("date")
    ax.set_ylabel("GMM Prob")
    ax.legend(loc="lower right", fontsize=8)

    axes[0].set_title("K-Means vs GMM Crisis Probabilities")
    fig.tight_layout()
    out = OUTPUTS_DIR / "figure3_crisis_probabilities.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ============================================================
# 18. Figure 4 — Min-Max Normalised Regime Statistics  (paper Figure 4)
# ============================================================

def plot_figure4(regime: pd.Series):
    """
    Heatmap of min-max normalised mean FRED-MD statistics per regime.
    Reads from raw FRED-MD (before Group 6 removal) so FEDFUNDS is available.
    Saved as: outputs/figure4_regime_stats.png
    """
    print("\n" + "=" * 60)
    print("FIGURE 4: Min-Max Normalised Regime Statistics")
    print("=" * 60)

    # Load source data — prefer raw FRED-MD so FEDFUNDS is available
    if FRED_RAW_PATH.exists():
        fred_raw = pd.read_csv(FRED_RAW_PATH)
        fred     = fred_raw.iloc[1:].copy()              # skip t-code row
        fred     = fred.rename(columns={fred.columns[0]: "date"})
        fred["date"] = pd.to_datetime(fred["date"])
        fred     = fred.set_index("date").sort_index()
        print(f"  Source: {FRED_RAW_PATH.name}")
    elif FRED_TRANS_PATH.exists():
        fred = pd.read_csv(FRED_TRANS_PATH, index_col=0, parse_dates=True)
        print(f"  Source: {FRED_TRANS_PATH.name} (FEDFUNDS not available)")
    else:
        print("  FRED-MD not found — skipping Figure 4.")
        return None

    # Find available key variable columns
    available = {}
    for label, candidates in KEY_VARS.items():
        for c in candidates:
            if c in fred.columns:
                available[label] = c
                break
    if not available:
        print("  No key variables found — skipping Figure 4.")
        return None
    print(f"  Variables: {list(available.keys())}")

    # Compute per-regime means
    common = regime.index.intersection(fred.index)
    r, f   = regime.loc[common], fred.loc[common]
    rows   = {}
    for label, col in available.items():
        s = pd.to_numeric(f[col], errors="coerce").dropna()
        rows[label] = s.groupby(r.loc[s.index]).mean()

    stats      = pd.DataFrame(rows).T             # (variables × regimes)
    stats_norm = stats.apply(
        lambda row: (row - row.min()) / (row.max() - row.min() + 1e-12), axis=1
    )

    print("\n  Min-Max Normalised (paper Figure 4):")
    print(stats_norm.round(2).to_string())

    # Regime 0 verdict (Section 4.4)
    print("\n  Regime 0 check:")
    for var in ["RPI", "UMCSENTx", "FEDFUNDS", "S&P 500"]:
        if var in stats_norm.index and 0 in stats_norm.columns:
            v = stats_norm.loc[var, 0]
            print(f"    {var:12s}  norm={v:.3f}  "
                  f"{'OK  low' if v < 0.3 else 'WARN not low'}")
    if "UNRATE" in stats_norm.index and 0 in stats_norm.columns:
        v = stats_norm.loc["UNRATE", 0]
        print(f"    {'UNRATE':12s}  norm={v:.3f}  "
              f"{'OK  high' if v > 0.7 else 'WARN not high'}")

    # Plot — matches paper Figure 4 style
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(stats_norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(stats_norm.shape[1]))
    ax.set_xticklabels([str(c) for c in stats_norm.columns])
    ax.set_yticks(range(stats_norm.shape[0]))
    ax.set_yticklabels(stats_norm.index)
    ax.set_xlabel("Regime")
    ax.set_ylabel("FRED Statistic")
    ax.set_title("Min-Max Normalized Regime Statistics")

    for i in range(stats_norm.shape[0]):
        for j in range(stats_norm.shape[1]):
            ax.text(j, i, f"{stats_norm.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)

    fig.tight_layout()
    out = OUTPUTS_DIR / "figure4_regime_stats.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out.name}")
    return stats_norm


# ============================================================
# 19. Figure 5 — Transition matrices  (paper Figure 5)
# ============================================================

def plot_figure5(trans: pd.DataFrame, trans_norm: pd.DataFrame):
    """
    Side-by-side heatmaps:
      Left  — full transition probability matrix
      Right — normalised matrix (given a transition occurs)
    Saved as: outputs/figure5_transition_matrices.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n = N_TOTAL_REGIMES

    for ax, matrix, title in zip(
        axes,
        [trans, trans_norm],
        ["Regime Transition Matrix",
         "Normalized Regime Transition Matrix"],
    ):
        im = ax.imshow(matrix.values, aspect="auto",
                       cmap="YlGnBu", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(n))
        ax.set_xticklabels([str(c) for c in matrix.columns])
        ax.set_yticks(range(n))
        ax.set_yticklabels([str(i) for i in matrix.index])
        ax.set_xlabel("From Regime")
        ax.set_ylabel("To Regime")
        ax.set_title(title)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix.iloc[i, j]:.3f}",
                        ha="center", va="center", fontsize=7)

    fig.tight_layout()
    out = OUTPUTS_DIR / "figure5_transition_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ============================================================
# 20. Table 1 — Regime labels  (paper Table 1)
# ============================================================

def print_table1(regime: pd.Series):
    """
    Print Table 1 style regime summary and save to CSV.
    Saved as: data/processed/table1_regime_labels.csv
    """
    print("\n" + "=" * 60)
    print("TABLE 1 — Regime Labels (Section 4.6)")
    print("=" * 60)

    counts = regime.value_counts().sort_index()
    rows   = []
    for r in range(N_TOTAL_REGIMES):
        label = REGIME_LABELS[r]
        count = counts.get(r, 0)
        pct   = count / len(regime) * 100
        print(f"  Regime {r} | {label:<28} | {count:3d} months ({pct:.1f}%)")
        rows.append({"regime": r, "label": label,
                     "count": count, "pct_of_sample": round(pct, 1)})

    table1 = pd.DataFrame(rows)
    out    = PROCESSED_DIR / "table1_regime_labels.csv"
    table1.to_csv(out, index=False)
    print(f"\n  Saved: {out.name}")


# ============================================================
# 21. Save all data outputs
# ============================================================

def save_outputs(regime, trans_counts, trans, trans_norm,
                 current_probs, next_probs,
                 l2_model, outlier_raw_id, ref_cosine_centroids):
    print("\n" + "=" * 60)
    print("STEP 7: Save data outputs")
    print("=" * 60)

    regime.rename("regime").to_frame().to_csv(
        PROCESSED_DIR / "regime_labels_paper_like.csv"
    )
    trans_counts.to_csv(PROCESSED_DIR / "regime_transition_counts_paper_like.csv")
    trans.to_csv(       PROCESSED_DIR / "regime_transition_matrix_paper_like.csv")
    trans_norm.to_csv(  PROCESSED_DIR / "regime_transition_matrix_normalized_paper_like.csv")
    current_probs.to_csv(PROCESSED_DIR / "regime_probabilities_paper_like.csv")
    next_probs.to_csv(   PROCESSED_DIR / "next_regime_probabilities_paper_like.csv")

    np.save(PROCESSED_DIR / "l2_centroids.npy",         l2_model.cluster_centers_)
    np.save(PROCESSED_DIR / "cosine_centroids.npy",     ref_cosine_centroids)
    with open(PROCESSED_DIR / "l2_outlier_cluster_id.json", "w") as fh:
        json.dump({"outlier_cluster_id": int(outlier_raw_id)}, fh)

    for name in [
        "regime_labels_paper_like.csv",
        "regime_probabilities_paper_like.csv",
        "next_regime_probabilities_paper_like.csv",
        "regime_transition_matrix_paper_like.csv",
        "regime_transition_matrix_normalized_paper_like.csv",
        "regime_transition_counts_paper_like.csv",
        "l2_centroids.npy", "cosine_centroids.npy",
        "ref_l2_centroids.npy", "ref_cosine_centroids.npy",
        "l2_outlier_cluster_id.json",
    ]:
        print(f"  {name}")


# ============================================================
# 22. Main
# ============================================================

def main():
    print("=" * 60)
    print("DETECT REGIMES")
    print("=" * 60)

    macro_pca = load_macro_pca()

    # Full-history clustering
    regime, l2_model, outlier_raw_id, ref_cosine_centroids, norm_p97 = \
        fit_full_history(macro_pca)

    # Transition matrix (Eq.5)
    trans_counts, trans, trans_norm = compute_transition_matrix(regime)

    # Regime probabilities (Eq.1 + Eqs.3-4 + Eq.7)
    current_probs = compute_current_probs(macro_pca, norm_p97, ref_cosine_centroids)
    next_probs    = compute_next_probs(current_probs, trans)

    # Sanity check
    sanity_check(current_probs, regime)

    # Save data outputs
    save_outputs(
        regime, trans_counts, trans, trans_norm,
        current_probs, next_probs,
        l2_model, outlier_raw_id, ref_cosine_centroids,
    )

    # ── GMM (comparison only, Section 4.3) ───────────────────────────
    # Need is_outlier_arr for GMM; recompute from norm threshold
    X_full       = macro_pca.values.astype(float)
    row_norms    = np.linalg.norm(X_full, axis=1)
    is_outlier_arr = row_norms > norm_p97
    gmm_regime, gmm_probs = run_gmm(macro_pca, is_outlier_arr)

    # ── Paper figures ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    plot_figure2(regime, gmm_regime)          # Figure 2 (K-Means + GMM)
    plot_figure3(current_probs, gmm_probs)    # Figure 3 (K-Means + GMM)
    plot_figure4(regime)                      # Figure 4
    plot_figure5(trans, trans_norm)           # Figure 5

    # ── Table 1 ────────────────────────────────────────────────────────
    print_table1(regime)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("\nData outputs : data/processed/")
    print("Figures      : outputs/")
    print("  figure2_kmeans_regimes.png       (K-Means + GMM panels)")
    print("  figure3_crisis_probabilities.png (K-Means + GMM panels)")
    print("  figure4_regime_stats.png")
    print("  figure5_transition_matrices.png")


if __name__ == "__main__":
    main()