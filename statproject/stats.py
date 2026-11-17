"""Ranking metrics and resampling inference for detector scores.

Scores are "higher means more AI-like" throughout. Nothing here assumes a
score is a calibrated probability.
"""

from collections.abc import Callable
from itertools import product

import numpy as np
from scipy.stats import rankdata

Statistic = Callable[[np.ndarray], float | np.ndarray]


def roc_auc(scores: np.ndarray, is_ai: np.ndarray) -> float:
    """Chance that a random AI text outscores a random human text (ties count half)."""
    is_ai = np.asarray(is_ai, dtype=bool)
    n_ai, n_human = is_ai.sum(), (~is_ai).sum()
    if n_ai == 0 or n_human == 0:
        return np.nan
    ranks = rankdata(scores)
    u = ranks[is_ai].sum() - n_ai * (n_ai + 1) / 2
    return u / (n_ai * n_human)


def roc_curve(scores: np.ndarray, is_ai: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False- and true-positive rates as the cutoff sweeps from high to low."""
    scores, is_ai = np.asarray(scores), np.asarray(is_ai, dtype=bool)
    cutoffs = np.unique(scores)[::-1]
    fpr = [np.mean(scores[~is_ai] >= c) for c in cutoffs]
    tpr = [np.mean(scores[is_ai] >= c) for c in cutoffs]
    return np.array([0.0, *fpr]), np.array([0.0, *tpr])


def threshold_at_fpr(human_scores: np.ndarray, max_fpr: float) -> float:
    """Lowest cutoff such that flagging scores strictly above it flags at most
    max_fpr of the human texts."""
    ordered = np.sort(human_scores)
    allowed = int(np.floor(max_fpr * len(ordered)))
    return ordered[len(ordered) - allowed - 1]


def tpr_at_fpr(scores: np.ndarray, is_ai: np.ndarray, max_fpr: float) -> float:
    scores, is_ai = np.asarray(scores), np.asarray(is_ai, dtype=bool)
    if is_ai.all() or not is_ai.any():
        return np.nan
    return float(np.mean(scores[is_ai] > threshold_at_fpr(scores[~is_ai], max_fpr)))


def hit_rate(scores: np.ndarray, is_ai: np.ndarray, threshold: float) -> float:
    return float(np.mean((np.asarray(scores) >= threshold) == np.asarray(is_ai, dtype=bool)))


def cluster_bootstrap(
    statistic: Statistic,
    clusters: np.ndarray,
    strata: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap replicates of statistic(row_indices), resampling whole clusters.

    The statistic may return one number or an array of several.

    Rows that share a cluster (chunks of one essay, a passage and its
    paraphrase) move together, so their correlation shows up as wider
    intervals instead of being ignored. With strata, clusters are resampled
    within each stratum, which keeps the human/AI group sizes fixed.
    """
    clusters = np.asarray(clusters)
    strata = np.zeros(len(clusters), dtype=int) if strata is None else np.asarray(strata)
    rows_by_stratum = []
    for stratum in np.unique(strata):
        in_stratum = np.flatnonzero(strata == stratum)
        names = np.unique(clusters[in_stratum])
        rows_by_stratum.append([in_stratum[clusters[in_stratum] == name] for name in names])

    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(n_boot):
        drawn = [
            groups[i]
            for groups in rows_by_stratum
            for i in rng.integers(len(groups), size=len(groups))
        ]
        replicates.append(statistic(np.concatenate(drawn)))
    return np.array(replicates)


def percentile_interval(replicates: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(replicates, [tail, 100 - tail])
    return float(lo), float(hi)


def stratified_correlation_pvalue(
    x: np.ndarray, y: np.ndarray, strata: np.ndarray, n_perm: int = 10000, seed: int = 0
) -> float:
    """Two-sided permutation p-value for the correlation of x and y, shuffling y
    only among rows of the same stratum."""
    x, y, strata = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(strata)
    members = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    rng = np.random.default_rng(seed)
    observed = abs(np.corrcoef(x, y)[0, 1])
    if np.isnan(observed):
        return np.nan
    hits = 0
    for _ in range(n_perm):
        shuffled = y.copy()
        for rows in members:
            shuffled[rows] = y[rng.permutation(rows)]
        hits += abs(np.corrcoef(x, shuffled)[0, 1]) >= observed
    return (hits + 1) / (n_perm + 1)


def cluster_permutation_pvalue(
    statistic: Callable[[np.ndarray], float],
    in_group: np.ndarray,
    clusters: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided p-value for statistic(in_group), shuffling group membership
    between clusters. Every cluster must sit wholly inside or outside the group."""
    in_group = np.asarray(in_group, dtype=bool)
    names, cluster_of_row = np.unique(clusters, return_inverse=True)
    cluster_in_group = np.zeros(len(names), dtype=bool)
    cluster_in_group[cluster_of_row[in_group]] = True
    if not np.array_equal(cluster_in_group[cluster_of_row], in_group):
        raise ValueError("a cluster straddles the two groups")

    rng = np.random.default_rng(seed)
    observed = abs(statistic(in_group))
    hits = sum(
        abs(statistic(rng.permutation(cluster_in_group)[cluster_of_row])) >= observed
        for _ in range(n_perm)
    )
    # The observed labelling counts as one permutation, so p is never zero.
    return (hits + 1) / (n_perm + 1)


def sign_flip_pvalue(differences: np.ndarray) -> float:
    """Exact two-sided paired test: under the null each difference is as likely
    to be negative as positive, so enumerate all 2^n sign patterns."""
    differences = np.asarray(differences, dtype=float)
    observed = abs(differences.sum())
    sums = [abs(np.dot(signs, differences)) for signs in product((1, -1), repeat=len(differences))]
    return float(np.mean(np.array(sums) >= observed - 1e-12))


def paired_permutation_pvalue(differences: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    """Monte Carlo version of sign_flip_pvalue for when 2^n is too many."""
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    observed = abs(differences.sum())
    signs = rng.choice((1.0, -1.0), size=(n_perm, len(differences)))
    hits = np.sum(np.abs(signs @ differences) >= observed - 1e-12)
    return float((hits + 1) / (n_perm + 1))


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in the original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    scaled = pvalues[order] * (len(pvalues) - np.arange(len(pvalues)))
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return adjusted
