from collections.abc import Callable
from itertools import product

import numpy as np
from scipy.stats import rankdata

Statistic = Callable[[np.ndarray], float | np.ndarray]


def roc_auc(scores: np.ndarray, is_ai: np.ndarray) -> float:
    is_ai = np.asarray(is_ai, dtype=bool)
    n_ai, n_human = is_ai.sum(), (~is_ai).sum()
    if n_ai == 0 or n_human == 0:
        return np.nan
    ranks = rankdata(scores)
    u = ranks[is_ai].sum() - n_ai * (n_ai + 1) / 2
    return u / (n_ai * n_human)


def roc_curve(scores: np.ndarray, is_ai: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores, is_ai = np.asarray(scores), np.asarray(is_ai, dtype=bool)
    cutoffs = np.unique(scores)[::-1]
    fpr = [np.mean(scores[~is_ai] >= c) for c in cutoffs]
    tpr = [np.mean(scores[is_ai] >= c) for c in cutoffs]
    return np.array([0.0, *fpr]), np.array([0.0, *tpr])


def threshold_at_fpr(human_scores: np.ndarray, max_fpr: float) -> float:
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


def stratified_group_pvalue(
    statistic: Callable[[np.ndarray], float],
    in_group: np.ndarray,
    strata: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    in_group = np.asarray(in_group, dtype=bool)
    members = [np.flatnonzero(np.asarray(strata) == value) for value in np.unique(strata)]
    rng = np.random.default_rng(seed)
    observed = abs(statistic(in_group))
    hits = 0
    for _ in range(n_perm):
        shuffled = in_group.copy()
        for rows in members:
            shuffled[rows] = in_group[rng.permutation(rows)]
        hits += abs(statistic(shuffled)) >= observed
    return (hits + 1) / (n_perm + 1)


def sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(differences.sum())
    sums = [abs(np.dot(signs, differences)) for signs in product((1, -1), repeat=len(differences))]
    return float(np.mean(np.array(sums) >= observed - 1e-12))


def paired_permutation_pvalue(differences: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    observed = abs(differences.sum())
    signs = rng.choice((1.0, -1.0), size=(n_perm, len(differences)))
    hits = np.sum(np.abs(signs @ differences) >= observed - 1e-12)
    return float((hits + 1) / (n_perm + 1))


def holm(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    scaled = pvalues[order] * (len(pvalues) - np.arange(len(pvalues)))
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return adjusted
