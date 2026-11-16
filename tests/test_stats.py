import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from statproject import stats


def test_auc_by_hand():
    # AI 2.5 beats two of three humans, AI 4 beats all three: 5 of 6 pairs.
    scores = np.array([1, 2, 3, 2.5, 4])
    is_ai = np.array([0, 0, 0, 1, 1], dtype=bool)
    assert stats.roc_auc(scores, is_ai) == pytest.approx(5 / 6)


def test_auc_extremes_and_ties():
    is_ai = np.array([0, 0, 1, 1], dtype=bool)
    assert stats.roc_auc(np.array([1, 2, 3, 4]), is_ai) == 1.0
    assert stats.roc_auc(np.array([3, 4, 1, 2]), is_ai) == 0.0
    assert stats.roc_auc(np.array([7, 7, 7, 7]), is_ai) == 0.5


def test_auc_matches_mann_whitney():
    rng = np.random.default_rng(1)
    human, ai = rng.normal(0, 1, 40), rng.normal(0.8, 1, 55)
    scores = np.concatenate([human, ai])
    is_ai = np.arange(95) >= 40
    u = mannwhitneyu(ai, human).statistic
    assert stats.roc_auc(scores, is_ai) == pytest.approx(u / (40 * 55))


def test_auc_needs_both_classes():
    assert np.isnan(stats.roc_auc(np.array([1.0, 2.0]), np.array([True, True])))


def test_tpr_at_fpr_by_hand():
    human = np.arange(1, 101)
    ai = np.array([90, 95, 96, 99, 150, 200])
    scores = np.concatenate([human, ai])
    is_ai = np.arange(106) >= 100
    # 5% of 100 humans may be flagged, so the cutoff is the 95th human score.
    assert stats.threshold_at_fpr(human, 0.05) == 95
    assert stats.tpr_at_fpr(scores, is_ai, 0.05) == pytest.approx(4 / 6)
    # At 1% only one human may be flagged: cutoff 99, and AI must beat it strictly.
    assert stats.tpr_at_fpr(scores, is_ai, 0.01) == pytest.approx(2 / 6)


def test_tpr_at_fpr_too_few_humans_allows_no_false_positives():
    scores = np.array([0.1, 0.2, 0.9, 0.5, 0.95])
    is_ai = np.array([0, 0, 0, 1, 1], dtype=bool)
    assert stats.tpr_at_fpr(scores, is_ai, 0.01) == pytest.approx(0.5)


def test_hit_rate():
    scores = np.array([0.1, 0.6, 0.5, 0.4])
    is_ai = np.array([0, 0, 1, 1], dtype=bool)
    assert stats.hit_rate(scores, is_ai, 0.5) == 0.5


def test_cluster_bootstrap_two_constant_clusters():
    # Two clusters of four identical rows. A replicate draws two clusters with
    # replacement, so the mean can only be 0, 0.5 or 1 with chances 1/4, 1/2, 1/4.
    values = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    clusters = np.array(list("aaaabbbb"))
    reps = stats.cluster_bootstrap(lambda rows: values[rows].mean(), clusters, n_boot=4000, seed=3)
    assert set(np.unique(reps)) == {0.0, 0.5, 1.0}
    assert np.mean(reps == 0.5) == pytest.approx(0.5, abs=0.03)
    assert np.mean(reps == 0.0) == pytest.approx(0.25, abs=0.03)
    # Treating the 8 rows as independent would give a variance of 0.25/8.
    assert reps.var() == pytest.approx(0.125, abs=0.01)


def test_cluster_bootstrap_singletons_match_ordinary_bootstrap():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    reps = stats.cluster_bootstrap(
        lambda rows: values[rows].mean(), np.arange(200), n_boot=3000, seed=1
    )
    assert reps.std() == pytest.approx(values.std() / np.sqrt(200), rel=0.1)


def test_cluster_bootstrap_strata_keep_group_sizes():
    strata = np.array([0] * 6 + [1] * 4)
    sizes = stats.cluster_bootstrap(
        lambda rows: np.sum(strata[rows] == 1), np.arange(10), strata=strata, n_boot=50
    )
    assert np.all(sizes == 4)


def test_percentile_interval():
    lo, hi = stats.percentile_interval(np.arange(101), level=0.9)
    assert (lo, hi) == pytest.approx((5.0, 95.0))


def test_sign_flip_exact():
    # All five differences positive: only the all-plus and all-minus patterns
    # are as extreme, 2 of 32.
    assert stats.sign_flip_pvalue(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(2 / 32)
    assert stats.sign_flip_pvalue(np.array([1.0, -1.0])) == 1.0


def test_paired_permutation_agrees_with_exact():
    differences = np.array([0.3, 0.5, -0.1, 0.4, 0.2, 0.6, -0.2, 0.1])
    exact = stats.sign_flip_pvalue(differences)
    approx = stats.paired_permutation_pvalue(differences, n_perm=20000, seed=5)
    assert approx == pytest.approx(exact, abs=0.01)


def test_cluster_permutation_detects_shift_and_respects_null():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(40), 3)
    in_group = clusters < 20
    noise = rng.normal(size=120)

    def mean_gap(values):
        return lambda group: values[group].mean() - values[~group].mean()

    shifted = noise + 2.0 * in_group
    assert stats.cluster_permutation_pvalue(mean_gap(shifted), in_group, clusters, n_perm=999) < 0.01
    assert stats.cluster_permutation_pvalue(mean_gap(noise), in_group, clusters, n_perm=999) > 0.05


def test_cluster_permutation_rejects_split_cluster():
    with pytest.raises(ValueError):
        stats.cluster_permutation_pvalue(
            lambda group: 0.0, np.array([True, False]), np.array(["a", "a"])
        )


def test_holm_by_hand():
    # Sorted: .01*3, .03*2, .04*1 -> .03, .06, .04, then made monotone.
    adjusted = stats.holm(np.array([0.01, 0.04, 0.03]))
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm(np.array([0.5, 0.9])).max() <= 1.0


def test_roc_curve_by_hand():
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    is_ai = np.array([0, 0, 1, 1], dtype=bool)
    fpr, tpr = stats.roc_curve(scores, is_ai)
    assert fpr.tolist() == [0.0, 0.0, 0.5, 0.5, 1.0]
    assert tpr.tolist() == [0.0, 0.5, 0.5, 1.0, 1.0]
    # The area under those points is the AUC: 3 of 4 pairs ordered correctly.
    assert np.trapezoid(tpr, fpr) == pytest.approx(stats.roc_auc(scores, is_ai)) == 0.75


def test_cluster_bootstrap_vector_statistic():
    values = np.arange(10.0)
    reps = stats.cluster_bootstrap(
        lambda rows: np.array([values[rows].mean(), values[rows].max()]), np.arange(10), n_boot=20
    )
    assert reps.shape == (20, 2)


def test_stratified_correlation_ignores_between_stratum_differences():
    # x and y both jump between strata but are unrelated inside each one, so
    # the pooled correlation is large and the stratified test should not care.
    rng = np.random.default_rng(4)
    strata = np.repeat([0, 1], 60)
    x = rng.normal(size=120) + 5 * strata
    y = rng.normal(size=120) + 5 * strata
    assert np.corrcoef(x, y)[0, 1] > 0.8
    assert stats.stratified_correlation_pvalue(x, y, strata, n_perm=999) > 0.05
    related = x + rng.normal(scale=0.5, size=120)
    assert stats.stratified_correlation_pvalue(x, related, strata, n_perm=999) < 0.01
