"""Figures for the README, drawn from the tables that evaluate.py wrote."""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import paths, stats
from .evaluate import FLAG_THRESHOLD, LOCAL_DETECTORS, load_corpus

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
# One colour and one line style per detector, the same in every figure. The
# line styles are there for readers who can't separate the hues.
DETECTOR_STYLE = {
    "sapling": ("Sapling", "#2a78d6", "-"),
    "roberta_fakespot": ("RoBERTa (Fakespot, 2025)", "#eb6834", "--"),
    "roberta_openai": ("RoBERTa (OpenAI, 2019)", "#1baf7a", "-."),
    "fast_detectgpt": ("Fast-DetectGPT", "#eda100", ":"),
    "log_likelihood": ("Log-likelihood", "#e87ba4", (0, (5, 1, 1, 1, 1, 1))),
}
GENRE_ORDER = ["student", "forum", "fiction"]
STYLE_LABELS = {
    "plain": "plain",
    "student": '"write like a student"',
    "anti_ai": '"don\'t sound like AI"',
    "casual": "casual and sloppy",
    "paraphrase": "paraphrased",
}
DPI = 200


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 9.5, "legend.frameon": False,
    })


def _save(fig: plt.Figure, name: str) -> None:
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s.png", name)


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.TABLES / f"{name}.csv")


def _roc_panel(ax: plt.Axes, frame: pd.DataFrame, detectors: list[str], title: str) -> None:
    is_ai = (frame["label"] == "ai").to_numpy()
    for detector in detectors:
        label, color, dashes = DETECTOR_STYLE[detector]
        scores = frame[detector].to_numpy()
        fpr, tpr = stats.roc_curve(scores, is_ai)
        auc = stats.roc_auc(scores, is_ai)
        ax.plot(fpr, tpr, color=color, linestyle=dashes, linewidth=2, label=f"{label}  {auc:.2f}")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, zorder=0)
    ax.set(
        title=title, xlabel="Share of human texts flagged", ylabel="Share of AI texts flagged",
        xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), aspect="equal",
    )
    ax.legend(
        title="Detector and AUC", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
        fontsize=8.5, title_fontsize=8.5, handlelength=3.2,
    )


def roc_curves(main: pd.DataFrame, chunks: pd.DataFrame) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 6))
    binary = chunks[chunks["label"] != "mixed"]
    _roc_panel(left, binary, ["sapling", *LOCAL_DETECTORS], "Original essays: 30 human vs 30 AI chunks")
    _roc_panel(right, main, LOCAL_DETECTORS, f"New corpus: {(main['label'] == 'human').sum()} human vs {(main['label'] == 'ai').sum()} AI passages")
    _save(fig, "roc_curves")


def sapling_by_version(chunks: pd.DataFrame) -> None:
    versions = {0: "Human original", 25: "Lightly AI-edited", 75: "AI draft, human-edited", 100: "Fully AI"}
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.4), sharey=True)
    _, color, _ = DETECTOR_STYLE["sapling"]
    for ax, (percent_ai, title) in zip(axes, versions.items()):
        part = chunks[chunks["percent_ai"] == percent_ai]
        ax.axhline(FLAG_THRESHOLD, color=INK_SECONDARY, linewidth=0.8)
        ax.scatter(
            part["essay"], part["sapling"], s=42, color=color,
            edgecolor=SURFACE, linewidth=1.2, clip_on=False, zorder=3,
        )
        flagged = (part["sapling"] >= FLAG_THRESHOLD).mean()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"essay\n{flagged:.0%} of chunks flagged")
        ax.set(xticks=range(1, 6), xlim=(0.4, 5.6), ylim=(-0.03, 1.03))
        ax.grid(axis="y")
    axes[0].set_ylabel("Sapling score")
    # Labelled in a panel where no dots sit near the line.
    axes[2].annotate(
        "0.5 cutoff", xy=(0.45, FLAG_THRESHOLD), xytext=(0, 3), textcoords="offset points",
        fontsize=8, color=INK_SECONDARY,
    )
    _save(fig, "sapling_by_version")


def _dot_rows(ax: plt.Axes, table: pd.DataFrame, row_column: str, row_labels: dict[str, str], value: str) -> None:
    """One row per category, one dot with a 95% interval per detector."""
    rows = list(row_labels)
    offsets = np.linspace(0.27, -0.27, len(LOCAL_DETECTORS))
    for detector, offset in zip(LOCAL_DETECTORS, offsets):
        label, color, _ = DETECTOR_STYLE[detector]
        part = table[table["detector"] == detector].set_index(row_column).loc[rows]
        y = np.arange(len(rows)) + offset
        ax.hlines(y, part[f"{value}_lo"], part[f"{value}_hi"], color=color, linewidth=1.6, alpha=0.55)
        ax.scatter(part[value], y, s=44, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=label)
    ax.set_yticks(range(len(rows)), [row_labels[row] for row in rows])
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)


def auc_by_genre() -> None:
    table = _table("by_genre")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    _dot_rows(ax, table, "genre", {genre: genre for genre in GENRE_ORDER}, "auc")
    ax.axvline(0.5, color=INK_SECONDARY, linewidth=0.8)
    ax.annotate("coin flip", xy=(0.5, -0.5), xytext=(3, 0), textcoords="offset points", fontsize=8, color=INK_SECONDARY, va="top")
    ax.set(title="Detection by genre (AUC, 95% bootstrap interval)", xlabel="AUC", xlim=(0, 1.02))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _save(fig, "auc_by_genre")


def detection_by_style() -> None:
    table = _table("by_style")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _dot_rows(ax, table, "prompt_style", STYLE_LABELS, "tpr_at_5pct_fpr")
    ax.set(
        title="AI passages caught when 5% of human passages are allowed to be flagged",
        xlabel="Share of AI passages caught", xlim=(-0.02, 1.02),
    )
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    _save(fig, "detection_by_style")


def human_scores_by_genre(main: pd.DataFrame, detector: str = "roberta_fakespot") -> None:
    humans = main[main["label"] == "human"]
    label, color, _ = DETECTOR_STYLE[detector]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for row, genre in enumerate(GENRE_ORDER):
        scores = humans.loc[humans["genre"] == genre, detector].to_numpy()
        jitter = rng.uniform(-0.28, 0.28, len(scores))
        ax.scatter(scores, row + jitter, s=22, color=color, alpha=0.75, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        flagged = (scores >= FLAG_THRESHOLD).mean()
        ax.annotate(f"{flagged:.0%} flagged", xy=(1.03, row), fontsize=8.5, color=INK, va="center", annotation_clip=False)
    ax.axvline(FLAG_THRESHOLD, color=INK_SECONDARY, linewidth=0.8)
    ax.set_yticks(range(len(GENRE_ORDER)), GENRE_ORDER)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.set(
        title=f"Human passages only: {label} score, one dot per passage",
        xlabel="Score (0.5 and above is called AI)", xlim=(-0.02, 1.02),
    )
    _save(fig, "human_scores_by_genre")


def run() -> None:
    _style()
    subsets, chunks = load_corpus()
    main = subsets["main"]
    roc_curves(main, chunks)
    sapling_by_version(chunks)
    auc_by_genre()
    detection_by_style()
    human_scores_by_genre(main)
