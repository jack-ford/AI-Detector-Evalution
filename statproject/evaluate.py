"""Turn detector scores into the tables under results/tables."""

import logging

import numpy as np
import pandas as pd

from . import paths, stats
from .corpus import load_legacy_chunks, load_metadata, load_scores, load_texts
from .features import feature_table

log = logging.getLogger(__name__)

LOCAL_DETECTORS = ["roberta_fakespot", "roberta_openai", "fast_detectgpt", "log_likelihood"]
# Only these output something meant to be read as P(AI), so only they get a 0.5 cutoff.
PROBABILITY_DETECTORS = {"sapling", "roberta_fakespot", "roberta_openai"}
FLAG_THRESHOLD = 0.5
STRICT_FPR, LOOSE_FPR = 0.01, 0.05
RANKING_METRICS = ["auc", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"]
CUTOFF_METRICS = ["hit_rate_at_0.5", "fpr_at_0.5", "tpr_at_0.5"]
LENGTH_BINS = {"under 150 words": 0, "150-249 words": 150, "250+ words": 250}
UNPAIRED_STYLES = ["student", "anti_ai", "casual"]
ALL_STYLES = ["plain", *UNPAIRED_STYLES, "paraphrase"]
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
    a cluster holds both labels (the original essays and their AI rewrites).
    """
    scores = frame[detector].to_numpy(dtype=float)
    is_ai = (frame["label"] == "ai").to_numpy()
    with_cutoff = detector in PROBABILITY_DETECTORS
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
    for detector in LOCAL_DETECTORS:
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


def breakdown(corpus: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = [
        {column: value} | detection_row(part, detector, part["cluster"])
        for value, part in corpus.groupby(column, sort=False)
        for detector in LOCAL_DETECTORS
    ]
    return pd.DataFrame(rows)


def plain_by_genre(corpus: pd.DataFrame) -> pd.DataFrame:
    """The easiest case, AI text written with no style instructions, genre by genre."""
    rows = []
    for genre, part in corpus.groupby("genre"):
        part = part[(part["label"] == "human") | (part["prompt_style"] == "plain")]
        rows += [{"genre": genre} | detection_row(part, d, part["cluster"]) for d in LOCAL_DETECTORS]
    return pd.DataFrame(rows)


def by_style(corpus: pd.DataFrame) -> pd.DataFrame:
    """Each style's AI passages against all human passages. Because the humans
    are the same in every row, tpr_at_5pct_fpr uses one shared cutoff."""
    humans = corpus[corpus["label"] == "human"]
    rows = []
    for style in ALL_STYLES:
        part = pd.concat([humans, corpus[corpus["prompt_style"] == style]])
        rows += [
            {"prompt_style": style} | detection_row(part, detector, part["cluster"])
            for detector in LOCAL_DETECTORS
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
    for detector in LOCAL_DETECTORS:
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
    for detector in LOCAL_DETECTORS:
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
        for detector in LOCAL_DETECTORS:
            row = detection_row(part, detector, part["cluster"])
            rows.append({"writer": writer} | {k: row[k] for k in ("detector", "n_ai", "auc", "auc_lo", "auc_hi")})
    return pd.DataFrame(rows)


def human_thresholds(corpus: pd.DataFrame) -> dict[str, float]:
    humans = corpus[corpus["label"] == "human"]
    return {d: stats.threshold_at_fpr(humans[d].to_numpy(), LOOSE_FPR) for d in LOCAL_DETECTORS}


def false_positives_by_genre(corpus: pd.DataFrame) -> pd.DataFrame:
    humans = corpus[corpus["label"] == "human"]
    thresholds = human_thresholds(corpus)
    rows = []
    for genre, part in humans.groupby("genre"):
        for detector in LOCAL_DETECTORS:
            row = {
                "genre": genre, "detector": detector, "n_human": len(part),
                "median_score": part[detector].median(),
                "share_above_5pct_cutoff": (part[detector] > thresholds[detector]).mean(),
            }
            if detector in PROBABILITY_DETECTORS:
                row["share_flagged_at_0.5"] = (part[detector] >= FLAG_THRESHOLD).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def flagged_humans(corpus: pd.DataFrame, detector: str) -> pd.DataFrame:
    humans = corpus[corpus["label"] == "human"]
    flagged = humans[humans[detector] >= FLAG_THRESHOLD].sort_values(detector, ascending=False)
    return flagged[["id", "genre", "n_words", "source", "title", detector]]


def _within_genre_ranks(values: pd.Series, genre: pd.Series) -> np.ndarray:
    return values.groupby(genre).rank(pct=True).to_numpy()


def feature_associations(corpus: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation between each detector's score and each surface feature,
    over human passages only, computed within genre so that "student essays
    are short and misspelled" doesn't masquerade as a property of the detector."""
    humans = corpus[corpus["label"] == "human"].reset_index(drop=True)
    features = feature_table(load_texts(humans["id"]))
    rows = []
    for detector in LOCAL_DETECTORS:
        score_ranks = _within_genre_ranks(humans[detector], humans["genre"])
        for feature in features.columns:
            feature_ranks = _within_genre_ranks(features[feature], humans["genre"])
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
    for detector in LOCAL_DETECTORS:
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
        if detector in PROBABILITY_DETECTORS:
            row["ell_flagged_at_0.5"] = np.mean(scores[is_ell] >= FLAG_THRESHOLD)
            row["other_flagged_at_0.5"] = np.mean(scores[~is_ell] >= FLAG_THRESHOLD)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = stats.holm(table["p_value"])
    return table


def corpus_summary(subsets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    corpus = pd.concat(subsets.values())
    grouped = corpus.groupby(["subset", "genre", "label", "prompt_style"], sort=False)
    return grouped["n_words"].agg(n="size", mean_words="mean", sd_words="std").reset_index()


def em_dash_check(corpus: pd.DataFrame) -> pd.DataFrame:
    """Human fiction came from Gutenberg, whose plain text writes dashes as
    double hyphens, which were turned into em dashes. Do the detectors treat
    passages with an em dash differently from those without?"""
    fiction = corpus[(corpus["label"] == "human") & (corpus["genre"] == "fiction")].reset_index(drop=True)
    has_dash = load_texts(fiction["id"]).str.contains(EM_DASH).to_numpy()
    rows = []
    for detector in LOCAL_DETECTORS:
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


def imitation_probe(probe: pd.DataFrame) -> pd.DataFrame:
    """Real student essays against AI passages written to imitate each one."""
    humans = probe[probe["label"] == "human"].set_index("id")
    ai = probe[probe["label"] == "ai"]
    rows = []
    for detector in LOCAL_DETECTORS:
        row = detection_row(probe, detector, probe["cluster"])
        gap = ai[detector].to_numpy() - humans.loc[ai["pair_id"], detector].to_numpy()
        row["ai_outscores_twin"] = np.mean(gap > 0) + 0.5 * np.mean(gap == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_summary(chunks: pd.DataFrame) -> pd.DataFrame:
    """Sapling and the local detectors on the original human and fully-AI chunks.

    Chunks of one essay are not independent, and each AI essay was generated
    from its human counterpart, so the essay (both versions together) is the
    resampling unit. There are five of them.
    """
    binary = chunks[chunks["label"] != "mixed"]
    essay_means = binary.pivot_table(index="essay", columns="label", aggfunc="mean", values=["sapling", *LOCAL_DETECTORS])
    rows = []
    for detector in ["sapling", *LOCAL_DETECTORS]:
        row = detection_row(binary, detector, binary["essay"], stratify=False)
        gaps = essay_means[detector]["ai"] - essay_means[detector]["human"]
        row["essays_where_ai_scores_higher"] = int((gaps > 0).sum())
        row["sign_flip_p"] = stats.sign_flip_pvalue(gaps.to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_by_version(chunks: pd.DataFrame) -> pd.DataFrame:
    grouped = chunks.groupby(["essay", "percent_ai"])["sapling"]
    table = grouped.agg(n_chunks="size", mean_score="mean")
    table["share_flagged_at_0.5"] = grouped.apply(lambda s: (s >= FLAG_THRESHOLD).mean())
    return table.reset_index()


def legacy_mixed(chunks: pd.DataFrame) -> pd.DataFrame:
    """Side analysis: the hybrid versions against the human originals. These
    have no clean label, so this only asks whether Sapling scores them higher."""
    rows = []
    for percent_ai in (25, 75, 100):
        part = chunks[chunks["percent_ai"].isin([0, percent_ai])].copy()
        part["label"] = np.where(part["percent_ai"] == 0, "human", "ai")
        row = detection_row(part, "sapling", part["essay"], stratify=False)
        rows.append({"version": f"{percent_ai}% vs 0%"} | {
            key: row[key] for key in ("n_human", "n_ai", "auc", "auc_lo", "auc_hi", "tpr_at_0.5", "fpr_at_0.5")
        })
    return pd.DataFrame(rows)


def legacy_agreement(chunks: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between Sapling and each local detector over all
    119 chunks, with intervals from resampling essays."""
    rows = []
    for detector in LOCAL_DETECTORS:
        def rho(rows_: np.ndarray) -> float:
            part = chunks.iloc[rows_]
            return part["sapling"].corr(part[detector], method="spearman")

        replicates = stats.cluster_bootstrap(rho, chunks["essay"].to_numpy(), n_boot=N_BOOT, seed=SEED)
        lo, hi = stats.percentile_interval(replicates)
        row = {"detector": detector, "spearman_with_sapling": rho(np.arange(len(chunks))), "lo": lo, "hi": hi}
        if detector in PROBABILITY_DETECTORS:
            row["same_call_at_0.5"] = np.mean(
                (chunks["sapling"] >= FLAG_THRESHOLD) == (chunks[detector] >= FLAG_THRESHOLD)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _length_bin(n_words: pd.Series) -> pd.Series:
    labels = list(LENGTH_BINS)
    edges = [*LENGTH_BINS.values(), np.inf]
    return pd.cut(n_words, bins=edges, labels=labels, right=False).astype(str)


def load_corpus() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Returns the passage corpus split by subset, and the original Sapling
    chunks, each with the local detector scores attached."""
    scores = load_scores()
    metadata = load_metadata().join(scores, on="id")
    metadata["length"] = _length_bin(metadata["n_words"])
    subsets = {
        name: part.reset_index(drop=True) for name, part in metadata.groupby("subset")
    }
    chunks = load_legacy_chunks().join(scores, on="id")
    return subsets, chunks


def run() -> None:
    subsets, chunks = load_corpus()
    main = subsets["main"]
    tables = {
        "corpus_summary": corpus_summary(subsets),
        "overall": overall(main),
        "length_baseline": pd.DataFrame([detection_row(main, "n_words", main["cluster"])]),
        "em_dash_check": em_dash_check(main),
        "by_genre": breakdown(main, "genre"),
        "by_length": breakdown(main, "length"),
        "plain_by_genre": plain_by_genre(main),
        "by_style": by_style(main),
        "style_tests": style_tests(main),
        "matched_pairs": matched_pairs(main),
        "by_writer": by_writer(main),
        "false_positives_by_genre": false_positives_by_genre(main),
        "flagged_humans_fakespot": flagged_humans(main, "roberta_fakespot"),
        "feature_associations": feature_associations(main),
        "ell_probe": ell_probe(subsets["ell_probe"], human_thresholds(main)),
        "imitation_probe": imitation_probe(subsets["imitation_probe"]),
        "detector_correlations": main[LOCAL_DETECTORS].corr(method="spearman").reset_index(names="detector"),
        "legacy_summary": legacy_summary(chunks),
        "legacy_by_version": legacy_by_version(chunks),
        "legacy_mixed": legacy_mixed(chunks),
        "legacy_agreement": legacy_agreement(chunks),
    }
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(paths.TABLES / f"{name}.csv", index=False, float_format="%.4f")
        log.info("wrote %s (%d rows)", name, len(table))
