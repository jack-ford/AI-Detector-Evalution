"""Turn detector scores into the tables under results/tables."""

import json
import logging
import platform
from importlib.metadata import version
from itertools import combinations

import numpy as np
import pandas as pd

from . import paths, registry, stats
from .corpus import load_metadata, load_rewrite_chunks, load_scores, load_texts
from .features import feature_table

log = logging.getLogger(__name__)

DETECTORS = registry.NAMES
PROBABILITY = [name for name in DETECTORS if name in registry.PROBABILITY_NAMES]
FLAG_THRESHOLD = 0.5
STRICT_FPR, LOOSE_FPR = 0.01, 0.05
RANKING_METRICS = ["auc", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"]
CUTOFF_METRICS = ["hit_rate_at_0.5", "fpr_at_0.5", "tpr_at_0.5"]
LENGTH_BINS = {"under 150 words": 0, "150-249 words": 150, "250+ words": 250}
UNPAIRED_STYLES = ["student", "anti_ai", "casual"]
ALL_STYLES = ["plain", *UNPAIRED_STYLES, "paraphrase"]
GENRES = ["student", "forum", "fiction"]
EM_DASH = chr(0x2014)
N_BOOT = 2000
N_PERM = 10_000
SEED = 0


def _metrics(scores: np.ndarray, is_ai: np.ndarray, with_cutoff: bool) -> np.ndarray:
    values = [
        stats.roc_auc(scores, is_ai),
        stats.tpr_at_fpr(scores, is_ai, STRICT_FPR),
        stats.tpr_at_fpr(scores, is_ai, LOOSE_FPR),
    ]
    if with_cutoff:
        flagged = scores >= FLAG_THRESHOLD
        values += [np.mean(flagged == is_ai), flagged[~is_ai].mean(), flagged[is_ai].mean()]
    return np.array(values)


def detection_row(frame: pd.DataFrame, detector: str, clusters: pd.Series, stratify: bool = True) -> dict:
    """Point estimates and cluster-bootstrap intervals for one detector on one slice.

    stratify=True resamples human and AI clusters separately; turn it off when
    a cluster holds both labels (the essay rewrites).
    """
    scores = frame[detector].to_numpy(dtype=float)
    is_ai = (frame["label"] == "ai").to_numpy()
    with_cutoff = detector in registry.PROBABILITY_NAMES
    names = RANKING_METRICS + CUTOFF_METRICS if with_cutoff else RANKING_METRICS

    def compute(rows: np.ndarray) -> np.ndarray:
        return _metrics(scores[rows], is_ai[rows], with_cutoff)

    point = compute(np.arange(len(frame)))
    replicates = stats.cluster_bootstrap(
        compute, clusters.to_numpy(), strata=is_ai if stratify else None, n_boot=N_BOOT, seed=SEED
    )
    row = {"detector": detector, "n_human": int((~is_ai).sum()), "n_ai": int(is_ai.sum())}
    for k, name in enumerate(names):
        lo, hi = stats.percentile_interval(replicates[:, k])
        row |= {name: point[k], f"{name}_lo": lo, f"{name}_hi": hi}
    return row


def overall(corpus: pd.DataFrame) -> pd.DataFrame:
    is_ai = (corpus["label"] == "ai").to_numpy()
    rows = []
    for detector in DETECTORS:
        scores = corpus[detector].to_numpy()
        row = detection_row(corpus, detector, corpus["cluster"])
        row["p_value"] = stats.cluster_permutation_pvalue(
            lambda group: stats.roc_auc(scores, group) - 0.5,
            is_ai, corpus["cluster"].to_numpy(), n_perm=N_PERM, seed=SEED,
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = stats.holm(table["p_value"])
    return table


def auc_contrasts(corpus: pd.DataFrame) -> pd.DataFrame:
    """Every pair of detectors on the same passages, with the AUC difference
    bootstrapped over the same resamples so the shared passages cancel out."""
    scores = corpus[DETECTORS].to_numpy(dtype=float)
    is_ai = (corpus["label"] == "ai").to_numpy()

    def aucs(rows: np.ndarray) -> np.ndarray:
        return np.array([stats.roc_auc(scores[rows, k], is_ai[rows]) for k in range(len(DETECTORS))])

    replicates = stats.cluster_bootstrap(
        aucs, corpus["cluster"].to_numpy(), strata=is_ai, n_boot=N_BOOT, seed=SEED
    )
    point = aucs(np.arange(len(corpus)))
    rows = []
    for i, j in combinations(range(len(DETECTORS)), 2):
        diff = replicates[:, i] - replicates[:, j]
        lo, hi = stats.percentile_interval(diff)
        # Share of resamples on the other side of zero, doubled for a two-sided p.
        tail = min(np.mean(diff <= 0), np.mean(diff >= 0))
        rows.append({
            "detector_a": DETECTORS[i], "detector_b": DETECTORS[j],
            "auc_difference": point[i] - point[j], "lo": lo, "hi": hi,
            "p_value": max(2 * tail, 1 / N_BOOT),
        })
    table = pd.DataFrame(rows)
    table["p_holm"] = stats.holm(table["p_value"])
    return table


def breakdown(corpus: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = [
        {column: value} | detection_row(part, detector, part["cluster"])
        for value, part in corpus.groupby(column, sort=False)
        for detector in DETECTORS
    ]
    return pd.DataFrame(rows)


def plain_by_genre(corpus: pd.DataFrame) -> pd.DataFrame:
    """The easiest case, AI text written with no style instructions, genre by genre."""
    rows = []
    for genre, part in corpus.groupby("genre"):
        part = part[(part["label"] == "human") | (part["prompt_style"] == "plain")]
        rows += [{"genre": genre} | detection_row(part, d, part["cluster"]) for d in DETECTORS]
    return pd.DataFrame(rows)


def by_style(corpus: pd.DataFrame) -> pd.DataFrame:
    """Each style's AI passages against all human passages. The humans are the
    same in every row, so the 5% cutoff is shared across styles."""
    humans = corpus[corpus["label"] == "human"]
    rows = []
    for style in ALL_STYLES:
        part = pd.concat([humans, corpus[corpus["prompt_style"] == style]])
        rows += [
            {"prompt_style": style} | detection_row(part, detector, part["cluster"])
            for detector in DETECTORS
        ]
    return pd.DataFrame(rows)


def style_tests(corpus: pd.DataFrame) -> pd.DataFrame:
    """Does a style score lower than plain AI text?

    Three styles are separate passages, compared to the plain ones by
    permutation. Paraphrases are compared to their own source passage.
    """
    ai = corpus[corpus["label"] == "ai"].set_index("id")
    plain = ai[ai["prompt_style"] == "plain"]
    paraphrases = ai[ai["prompt_style"] == "paraphrase"]
    rows = []
    for detector in DETECTORS:
        for style in UNPAIRED_STYLES:
            both = pd.concat([plain, ai[ai["prompt_style"] == style]])
            scores = both[detector].to_numpy()
            is_plain = (both["prompt_style"] == "plain").to_numpy()
            rows.append({
                "detector": detector,
                "prompt_style": style,
                "test": "permutation vs plain",
                "n": int((~is_plain).sum()),
                "p_plain_scores_higher": stats.roc_auc(scores, is_plain),
                "mean_score_drop": scores[is_plain].mean() - scores[~is_plain].mean(),
                "p_value": stats.cluster_permutation_pvalue(
                    lambda group: stats.roc_auc(scores, group) - 0.5,
                    is_plain, both.index.to_numpy(), n_perm=N_PERM, seed=SEED,
                ),
            })
        drops = ai.loc[paraphrases["cluster"], detector].to_numpy() - paraphrases[detector].to_numpy()
        rows.append({
            "detector": detector,
            "prompt_style": "paraphrase",
            "test": "paired sign-flip vs source",
            "n": len(drops),
            "p_plain_scores_higher": np.mean(drops > 0) + 0.5 * np.mean(drops == 0),
            "mean_score_drop": drops.mean(),
            "p_value": stats.paired_permutation_pvalue(drops, n_perm=N_PERM, seed=SEED),
        })
    table = pd.DataFrame(rows)
    table["p_holm"] = stats.holm(table["p_value"])
    return table


def matched_pairs(corpus: pd.DataFrame) -> pd.DataFrame:
    """How often an AI passage outscores the human passage it was matched to."""
    ai = corpus[(corpus["label"] == "ai") & (corpus["prompt_style"] != "paraphrase")]
    by_id = corpus.set_index("id")
    rows = []
    for detector in DETECTORS:
        gap = ai[detector].to_numpy() - by_id.loc[ai["pair_id"], detector].to_numpy()
        wins = (gap > 0) + 0.5 * (gap == 0)
        replicates = stats.cluster_bootstrap(
            lambda rows_: wins[rows_].mean(), np.arange(len(wins)), n_boot=N_BOOT, seed=SEED
        )
        lo, hi = stats.percentile_interval(replicates)
        rows.append({
            "detector": detector, "n_pairs": len(wins),
            "ai_outscores_twin": wins.mean(), "lo": lo, "hi": hi,
        })
    return pd.DataFrame(rows)


def by_writer(corpus: pd.DataFrame) -> pd.DataFrame:
    """AUC per writing session, as a check that no single batch of AI passages
    is driving the results."""
    humans = corpus[corpus["label"] == "human"]
    rows = []
    for writer, written in corpus[corpus["label"] == "ai"].groupby("writer"):
        part = pd.concat([humans, written])
        for detector in DETECTORS:
            row = detection_row(part, detector, part["cluster"])
            rows.append({"writer": writer} | {k: row[k] for k in ("detector", "n_ai", "auc", "auc_lo", "auc_hi")})
    return pd.DataFrame(rows)


def human_thresholds(corpus: pd.DataFrame, max_fpr: float = LOOSE_FPR) -> dict[str, float]:
    humans = corpus[corpus["label"] == "human"]
    return {d: stats.threshold_at_fpr(humans[d].to_numpy(), max_fpr) for d in DETECTORS}


def threshold_transfer(corpus: pd.DataFrame) -> pd.DataFrame:
    """Fix the cutoff to allow 5% false positives on the human text of two
    genres, then use it on the third. If a cutoff set in one place carried over,
    the held-out false positive rate would stay near 5%."""
    is_ai = (corpus["label"] == "ai").to_numpy()
    genre = corpus["genre"].to_numpy()
    plain = (corpus["prompt_style"] == "plain").to_numpy()
    rows = []
    for detector in DETECTORS:
        scores = corpus[detector].to_numpy(dtype=float)
        for held_out in GENRES:
            def evaluate_cutoff(rows_: np.ndarray) -> np.ndarray:
                s, a, g, p = scores[rows_], is_ai[rows_], genre[rows_], plain[rows_]
                cutoff = stats.threshold_at_fpr(s[~a & (g != held_out)], LOOSE_FPR)
                target = g == held_out
                return np.array([
                    np.mean(s[~a & target] > cutoff),
                    np.mean(s[a & target] > cutoff),
                    np.mean(s[a & target & p] > cutoff),
                ])

            point = evaluate_cutoff(np.arange(len(corpus)))
            replicates = stats.cluster_bootstrap(
                evaluate_cutoff, corpus["cluster"].to_numpy(), strata=is_ai, n_boot=N_BOOT, seed=SEED
            )
            row = {"detector": detector, "held_out_genre": held_out}
            for k, name in enumerate(["fpr", "tpr_all_ai", "tpr_plain_ai"]):
                lo, hi = stats.percentile_interval(replicates[:, k])
                row |= {name: point[k], f"{name}_lo": lo, f"{name}_hi": hi}
            rows.append(row)
    return pd.DataFrame(rows)


def false_positives_by_genre(corpus: pd.DataFrame) -> pd.DataFrame:
    humans = corpus[corpus["label"] == "human"]
    thresholds = human_thresholds(corpus)
    rows = []
    for genre, part in humans.groupby("genre"):
        for detector in DETECTORS:
            row = {
                "genre": genre, "detector": detector, "n_human": len(part),
                "median_score": part[detector].median(),
                "share_above_5pct_cutoff": (part[detector] > thresholds[detector]).mean(),
            }
            if detector in registry.PROBABILITY_NAMES:
                row["share_flagged_at_0.5"] = (part[detector] >= FLAG_THRESHOLD).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def flagged_humans(corpus: pd.DataFrame) -> pd.DataFrame:
    """Human passages that at least one probability-style detector calls AI."""
    humans = corpus[corpus["label"] == "human"].copy()
    humans["n_detectors_flagging"] = (humans[PROBABILITY] >= FLAG_THRESHOLD).sum(axis=1)
    flagged = humans[humans["n_detectors_flagging"] > 0].sort_values("n_detectors_flagging", ascending=False)
    return flagged[["id", "genre", "n_words", "source", "n_detectors_flagging", *PROBABILITY]]


def _within_group_ranks(values: pd.Series, groups: pd.Series) -> np.ndarray:
    return values.groupby(groups).rank(pct=True).to_numpy()


def feature_associations(corpus: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation between each detector's score and each surface feature,
    over human passages only, computed within genre so that "student essays
    are short and misspelled" doesn't masquerade as a property of the detector."""
    humans = corpus[corpus["label"] == "human"].reset_index(drop=True)
    features = feature_table(load_texts(humans["id"]))
    rows = []
    for detector in DETECTORS:
        score_ranks = _within_group_ranks(humans[detector], humans["genre"])
        for feature in features.columns:
            feature_ranks = _within_group_ranks(features[feature], humans["genre"])
            rows.append({
                "detector": detector,
                "feature": feature,
                "pooled_spearman": humans[detector].corr(features[feature], method="spearman"),
                "within_genre_spearman": np.corrcoef(score_ranks, feature_ranks)[0, 1],
                "p_value": stats.stratified_correlation_pvalue(
                    score_ranks, feature_ranks, humans["genre"].to_numpy(),
                    n_perm=N_PERM, seed=SEED,
                ),
            })
    table = pd.DataFrame(rows)
    table["p_holm"] = stats.holm(table["p_value"])
    return table


def ell_probe(probe: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """English learners against matched non-learners; all of it human writing."""
    is_ell = probe["detail"].str.contains("ell=Yes").to_numpy()
    rows = []
    for detector in DETECTORS:
        scores = probe[detector].to_numpy()

        def learner_scores_higher(rows_: np.ndarray) -> float:
            return stats.roc_auc(scores[rows_], is_ell[rows_])

        replicates = stats.cluster_bootstrap(
            learner_scores_higher, np.arange(len(probe)), strata=is_ell, n_boot=N_BOOT, seed=SEED
        )
        lo, hi = stats.percentile_interval(replicates)
        row = {
            "detector": detector,
            "n_ell": int(is_ell.sum()),
            "n_other": int((~is_ell).sum()),
            "p_ell_scores_higher": stats.roc_auc(scores, is_ell),
            "lo": lo,
            "hi": hi,
            "p_value": stats.cluster_permutation_pvalue(
                lambda group: stats.roc_auc(scores, group) - 0.5,
                is_ell, np.arange(len(probe)), n_perm=N_PERM, seed=SEED,
            ),
            "ell_above_5pct_cutoff": np.mean(scores[is_ell] > thresholds[detector]),
            "other_above_5pct_cutoff": np.mean(scores[~is_ell] > thresholds[detector]),
        }
        if detector in registry.PROBABILITY_NAMES:
            row["ell_flagged_at_0.5"] = np.mean(scores[is_ell] >= FLAG_THRESHOLD)
            row["other_flagged_at_0.5"] = np.mean(scores[~is_ell] >= FLAG_THRESHOLD)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = stats.holm(table["p_value"])
    return table


def ell_quality_matched(probe: pd.DataFrame) -> pd.DataFrame:
    """The English-learner gap again, comparing learners and non-learners only
    within the same holistic score (percentile-rank difference, learners minus
    others, averaged over scores weighted by how many essays have them)."""
    is_ell = probe["detail"].str.contains("ell=Yes").to_numpy()
    quality = probe["detail"].str.extract(r"holistic_score=(\d)")[0].astype(float).to_numpy()
    rows = []
    for detector in DETECTORS:
        ranks = probe[detector].rank(pct=True).to_numpy()

        def matched_gap(group: np.ndarray) -> float:
            gaps, sizes = [], []
            for level in np.unique(quality):
                at_level = quality == level
                if group[at_level].any() and (~group[at_level]).any():
                    gaps.append(ranks[at_level & group].mean() - ranks[at_level & ~group].mean())
                    sizes.append(at_level.sum())
            return float(np.average(gaps, weights=sizes))

        rows.append({
            "detector": detector,
            "mean_quality_ell": quality[is_ell].mean(),
            "mean_quality_other": quality[~is_ell].mean(),
            "rank_gap_unmatched": ranks[is_ell].mean() - ranks[~is_ell].mean(),
            "rank_gap_matched": matched_gap(is_ell),
            "p_value": stats.stratified_group_pvalue(matched_gap, is_ell, quality, n_perm=N_PERM, seed=SEED),
        })
    table = pd.DataFrame(rows)
    table["p_holm"] = stats.holm(table["p_value"])
    return table


def quality_probe(subsets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Do better human essays look more like AI? Rank correlation between the
    PERSUADE holistic score (1-6) and each detector's score, within prompt,
    over every human student essay in the corpus."""
    essays = pd.concat([
        subsets[name][subsets[name]["label"] == "human"] for name in ("main", "ell_probe", "imitation_probe")
    ])
    essays = essays[essays["source"] == "PERSUADE 2.0"].reset_index(drop=True)
    quality = essays["detail"].str.extract(r"holistic_score=(\d)")[0].astype(float)
    quality_ranks = _within_group_ranks(quality, essays["title"])
    rows = []
    for detector in DETECTORS:
        score_ranks = _within_group_ranks(essays[detector], essays["title"])

        def correlation(rows_: np.ndarray) -> float:
            return np.corrcoef(score_ranks[rows_], quality_ranks[rows_])[0, 1]

        lo, hi = stats.percentile_interval(
            stats.cluster_bootstrap(correlation, np.arange(len(essays)), n_boot=N_BOOT, seed=SEED)
        )
        rows.append({
            "detector": detector,
            "n_essays": len(essays),
            "within_prompt_spearman": correlation(np.arange(len(essays))),
            "lo": lo,
            "hi": hi,
            "p_value": stats.stratified_correlation_pvalue(
                score_ranks, quality_ranks, essays["title"].to_numpy(), n_perm=N_PERM, seed=SEED
            ),
        })
    table = pd.DataFrame(rows)
    table["p_holm"] = stats.holm(table["p_value"])
    return table


def imitation_probe(probe: pd.DataFrame) -> pd.DataFrame:
    """Real student essays against AI passages written to imitate each one."""
    humans = probe[probe["label"] == "human"].set_index("id")
    ai = probe[probe["label"] == "ai"]
    rows = []
    for detector in DETECTORS:
        row = detection_row(probe, detector, probe["cluster"])
        gap = ai[detector].to_numpy() - humans.loc[ai["pair_id"], detector].to_numpy()
        row["ai_outscores_twin"] = np.mean(gap > 0) + 0.5 * np.mean(gap == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def corpus_summary(subsets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    corpus = pd.concat(subsets.values())
    grouped = corpus.groupby(["subset", "genre", "label", "prompt_style"], sort=False)
    return grouped["n_words"].agg(n="size", mean_words="mean", sd_words="std").reset_index()


def length_match(corpus: pd.DataFrame) -> pd.DataFrame:
    """Written length of each AI passage as a share of the word count it was asked for."""
    plan = pd.read_csv(paths.ASSIGNMENTS, usecols=["id", "target_words"]).set_index("id")
    ai = corpus[corpus["label"] == "ai"].set_index("id")
    ratio = ai["n_words"] / plan.loc[ai.index, "target_words"]
    return ratio.groupby(ai["prompt_style"]).agg(n="size", mean="mean", min="min", max="max").reset_index()


def em_dash_check(corpus: pd.DataFrame) -> pd.DataFrame:
    """Human fiction came from Gutenberg, whose plain text writes dashes as
    double hyphens, which were turned into em dashes. Do the detectors treat
    passages with an em dash differently from those without?"""
    fiction = corpus[(corpus["label"] == "human") & (corpus["genre"] == "fiction")].reset_index(drop=True)
    has_dash = load_texts(fiction["id"]).str.contains(EM_DASH).to_numpy()
    rows = []
    for detector in DETECTORS:
        scores = fiction[detector].to_numpy()

        def dash_scores_higher(rows_: np.ndarray) -> float:
            return stats.roc_auc(scores[rows_], has_dash[rows_])

        replicates = stats.cluster_bootstrap(
            dash_scores_higher, np.arange(len(fiction)), strata=has_dash, n_boot=N_BOOT, seed=SEED
        )
        lo, hi = stats.percentile_interval(replicates)
        rows.append({
            "detector": detector, "n_with_dash": int(has_dash.sum()), "n_without": int((~has_dash).sum()),
            "p_dash_scores_higher": stats.roc_auc(scores, has_dash), "lo": lo, "hi": hi,
        })
    return pd.DataFrame(rows)


def rewrites_by_version(chunks: pd.DataFrame) -> pd.DataFrame:
    """Mean score for each essay at each level of AI involvement, plus the
    within-essay rank trend across levels."""
    rows = []
    for detector in DETECTORS:
        means = chunks.groupby(["essay", "percent_ai"])[detector].mean().unstack()
        score_ranks = _within_group_ranks(chunks[detector], chunks["essay"])
        level_ranks = _within_group_ranks(chunks["percent_ai"], chunks["essay"])
        rows.append({
            "detector": detector,
            **{f"mean_at_{level}": means[level].mean() for level in means.columns},
            **{f"essay_{essay}_at_{level}": means.loc[essay, level] for essay in means.index for level in means.columns},
            "trend_spearman": np.corrcoef(score_ranks, level_ranks)[0, 1],
            "trend_p": stats.stratified_correlation_pvalue(
                score_ranks, level_ranks, chunks["essay"].to_numpy(), n_perm=N_PERM, seed=SEED
            ),
        })
    table = pd.DataFrame(rows)
    table["trend_p_holm"] = stats.holm(table["trend_p"])
    return table


def rewrites_versus_human(chunks: pd.DataFrame) -> pd.DataFrame:
    """Each AI version of the essays against the human originals. The 25 and 75
    versions are hybrids, so only the 100 comparison has a clean meaning; the
    others show how detection grows with the share of AI involvement.

    Both versions of an essay share a cluster, and there are five essays, so
    the intervals are rough."""
    rows = []
    for percent_ai in (25, 75, 100):
        part = chunks[chunks["percent_ai"].isin([0, percent_ai])].copy()
        part["label"] = np.where(part["percent_ai"] == 0, "human", "ai")
        for detector in DETECTORS:
            row = detection_row(part, detector, part["essay"], stratify=False)
            rows.append({"version": percent_ai} | {k: row[k] for k in ("detector", "n_human", "n_ai", "auc", "auc_lo", "auc_hi", "tpr_at_5pct_fpr")})
    return pd.DataFrame(rows)


def load_corpus() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Returns the passage corpus split by subset, and the essay-rewrite chunks,
    each with the detector scores attached."""
    scores = load_scores()
    metadata = load_metadata().join(scores, on="id")
    labels = [*LENGTH_BINS]
    edges = [*LENGTH_BINS.values(), np.inf]
    metadata["length"] = pd.cut(metadata["n_words"], bins=edges, labels=labels, right=False).astype(str)
    subsets = {name: part.reset_index(drop=True) for name, part in metadata.groupby("subset")}
    chunks = load_rewrite_chunks().join(scores, on="id")
    return subsets, chunks


def write_manifest(subsets: dict[str, pd.DataFrame]) -> None:
    models_file = paths.SCORES_DIR / "models.json"
    manifest = {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "torch", "transformers")},
        "seed": SEED,
        "bootstrap_resamples": N_BOOT,
        "permutations": N_PERM,
        "detectors": {
            d.name: {"label": d.label, "family": d.family, "year": d.year, "license": d.license}
            for d in registry.DETECTORS
        },
        "model_revisions": json.loads(models_file.read_text(encoding="utf-8")),
        "passages": {name: len(part) for name, part in subsets.items()},
    }
    paths.MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run() -> None:
    subsets, chunks = load_corpus()
    main = subsets["main"]
    tables = {
        "corpus_summary": corpus_summary(subsets),
        "length_match": length_match(pd.concat(subsets.values())),
        "overall": overall(main),
        "auc_contrasts": auc_contrasts(main),
        "length_baseline": pd.DataFrame([detection_row(main, "n_words", main["cluster"])]),
        "em_dash_check": em_dash_check(main),
        "by_genre": breakdown(main, "genre"),
        "plain_by_genre": plain_by_genre(main),
        "by_length": breakdown(main, "length"),
        "by_style": by_style(main),
        "style_tests": style_tests(main),
        "matched_pairs": matched_pairs(main),
        "by_writer": by_writer(main),
        "threshold_transfer": threshold_transfer(main),
        "false_positives_by_genre": false_positives_by_genre(main),
        "flagged_humans": flagged_humans(main),
        "feature_associations": feature_associations(main),
        "ell_probe": ell_probe(subsets["ell_probe"], human_thresholds(main)),
        "ell_quality_matched": ell_quality_matched(subsets["ell_probe"]),
        "quality_probe": quality_probe(subsets),
        "imitation_probe": imitation_probe(subsets["imitation_probe"]),
        "detector_correlations": main[DETECTORS].corr(method="spearman").reset_index(names="detector"),
        "rewrites_by_version": rewrites_by_version(chunks),
        "rewrites_versus_human": rewrites_versus_human(chunks),
    }
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(paths.TABLES / f"{name}.csv", index=False, float_format="%.4f")
        log.info("wrote %s (%d rows)", name, len(table))
    write_manifest(subsets)
