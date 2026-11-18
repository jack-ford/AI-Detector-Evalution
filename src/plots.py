import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from . import paths, registry, stats
from .evaluate import ALL_STYLES, DETECTORS, FLAG_THRESHOLD, GENRES, PROBABILITY, load_corpus

log = logging.getLogger(__name__)

INK = "#111111"
INK_SECONDARY = "#555555"
GRID = "#e4e3df"
FAMILY_COLOR = {registry.SUPERVISED: "#2a78d6", registry.ZERO_SHOT: "#eb6834"}
GENRE_COLOR = {"student": "#2a78d6", "forum": "#eb6834", "fiction": "#1baf7a"}
AUC_MAP = LinearSegmentedColormap.from_list(
    "auc", [(0.0, "#a52a2a"), (0.25, "#e8827f"), (0.5, "#f0efec"), (0.75, "#86b6ef"), (1.0, "#184f95")]
)
STYLE_NAMES = {
    "plain": "Plain",
    "student": "Student voice",
    "anti_ai": "Avoid AI tells",
    "casual": "Casual, sloppy",
    "paraphrase": "Paraphrased",
}
GENRE_NAMES = {"student": "Student essays", "forum": "Forum answers", "fiction": "Fiction"}
EVASIVE_STYLES = ("anti_ai", "casual", "paraphrase")
DPI = 300
FULL_WIDTH = 7.0


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.edgecolor": INK_SECONDARY, "axes.linewidth": 0.6,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        # STIX is bundled with matplotlib and matches the Times body text of the paper.
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, name: str) -> None:
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(paths.FIGURES / f"{name}.{extension}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", name)


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.TABLES / f"{name}.csv")


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def _detector_labels() -> list[str]:
    return [registry.label(name) for name in DETECTORS]


def _family_legend(ax: plt.Axes, **kwargs) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="", color=color, markersize=5, label=family)
        for family, color in FAMILY_COLOR.items()
    ]
    ax.legend(handles=handles, **kwargs)


def _dot_whisker(ax: plt.Axes, table: pd.DataFrame, value: str, lo: str, hi: str, reference: float | None) -> None:
    table = table.set_index("detector").loc[DETECTORS]
    y = np.arange(len(DETECTORS))
    for k, name in enumerate(DETECTORS):
        color = FAMILY_COLOR[registry.BY_NAME[name].family]
        ax.hlines(y[k], table.loc[name, lo], table.loc[name, hi], color=color, linewidth=1.4)
        ax.plot(table.loc[name, value], y[k], "o", color=color, markersize=4.5, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    if reference is not None:
        ax.axvline(reference, color=INK_SECONDARY, linewidth=0.7, zorder=1)
    ax.set_yticks(y, _detector_labels())
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def overview() -> None:
    overall = _table("overall")
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    _dot_whisker(left, overall, "auc", "auc_lo", "auc_hi", reference=0.5)
    left.set(xlim=(0.1, 0.8), xlabel="ROC-AUC (human vs AI)")
    left.text(0.505, 1.01, "coin flip", fontsize=7, color=INK_SECONDARY, ha="left", va="bottom", transform=left.get_xaxis_transform())
    _dot_whisker(right, overall, "tpr_at_5pct_fpr", "tpr_at_5pct_fpr_lo", "tpr_at_5pct_fpr_hi", reference=None)
    right.set(xlim=(-0.02, 0.5), xlabel="AI passages caught at 5% false positives")
    right.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    right.tick_params(axis="y", left=False)
    _family_legend(right, loc="upper right", title=None)
    _panel_letter(left, "a")
    _panel_letter(right, "b")
    _save(fig, "fig1_overview")


def _annotated_heatmap(ax: plt.Axes, values: np.ndarray, lo: np.ndarray, hi: np.ndarray, columns: list[str], rows: list[str] | None) -> None:
    ax.imshow(values, cmap=AUC_MAP, vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            significant = lo[i, j] > 0.5 or hi[i, j] < 0.5
            shade = AUC_MAP(values[i, j])
            luminance = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
            ax.text(
                j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                fontweight="bold" if significant else "normal", color="white" if luminance < 0.5 else INK,
            )
    ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right", rotation_mode="anchor")
    if rows is not None:
        ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)


def _matrix(table: pd.DataFrame, column: str, categories: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = {
        key: table[table[column] == key].set_index("detector").loc[DETECTORS] for key in categories
    }
    return tuple(
        np.column_stack([parts[key][field] for key in categories]) for field in ("auc", "auc_lo", "auc_hi")
    )


def _colorbar(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    mappable = matplotlib.cm.ScalarMappable(cmap=AUC_MAP, norm=matplotlib.colors.Normalize(0, 1))
    bar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.03)
    bar.set_label(label)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)


def style_matrix() -> None:
    overall = _table("overall").set_index("detector").loc[DETECTORS]
    by_style = _table("by_style")
    values, lo, hi = _matrix(by_style, "prompt_style", ALL_STYLES)
    values = np.column_stack([overall["auc"], values])
    lo = np.column_stack([overall["auc_lo"], lo])
    hi = np.column_stack([overall["auc_hi"], hi])
    columns = ["All AI", *(STYLE_NAMES[s] for s in ALL_STYLES)]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.8, 3.5))
    _annotated_heatmap(ax, values, lo, hi, columns, _detector_labels())
    ax.axvline(0.5, color="white", linewidth=4)
    _colorbar(fig, ax, "ROC-AUC (0.5 = coin flip)")
    _save(fig, "fig2_style")


def genre_matrix() -> None:
    all_ai = _table("by_genre")
    plain = _table("plain_by_genre")
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.4), sharey=True, gridspec_kw={"wspace": 0.06})
    columns = [GENRE_NAMES[g] for g in GENRES]
    _annotated_heatmap(left, *_matrix(all_ai, "genre", GENRES), columns, _detector_labels())
    _annotated_heatmap(right, *_matrix(plain, "genre", GENRES), columns, None)
    left.set_title("All AI passages (50 per genre)", loc="left")
    right.set_title("Plain-prompted AI only (18 per genre)", loc="left")
    right.tick_params(axis="y", left=False)
    _colorbar(fig, right, "ROC-AUC")
    _save(fig, "fig4_genre")


def human_percentile_boxes(main: pd.DataFrame) -> None:
    humans = main[main["label"] == "human"]
    ai = main[main["label"] == "ai"]
    fig, axes = plt.subplots(1, len(ALL_STYLES), figsize=(FULL_WIDTH, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, style in zip(axes, ALL_STYLES):
        part = ai[ai["prompt_style"] == style]
        data = []
        for detector in DETECTORS:
            reference = np.sort(humans[detector].to_numpy())
            data.append(np.searchsorted(reference, part[detector].to_numpy(), side="right") / len(reference))
        box = ax.boxplot(
            data, vert=False, widths=0.62, patch_artist=True, showfliers=False,
            medianprops={"color": INK, "linewidth": 1.1}, whiskerprops={"linewidth": 0.7, "color": INK_SECONDARY},
            capprops={"linewidth": 0.7, "color": INK_SECONDARY}, boxprops={"linewidth": 0.6, "edgecolor": INK_SECONDARY},
        )
        for patch, name in zip(box["boxes"], DETECTORS):
            patch.set_facecolor(FAMILY_COLOR[registry.BY_NAME[name].family])
            patch.set_alpha(0.55)
        ax.axvline(0.95, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (3, 2)))
        ax.axvline(0.5, color=GRID, linewidth=0.7, zorder=0)
        ax.set_title(STYLE_NAMES[style], fontsize=7.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0, 0.5, 0.95], ["0", ".5", ".95"])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(1, len(DETECTORS) + 1), _detector_labels())
    axes[0].invert_yaxis()
    fig.supxlabel("Share of human passages scoring below the AI passage (dashed: 5% false-positive cutoff)", fontsize=8, y=-0.02)
    handles = [
        Line2D([], [], marker="s", linestyle="", color=color, alpha=0.7, markersize=6, label=family)
        for family, color in FAMILY_COLOR.items()
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(1.0, -0.06), columnspacing=1.0)
    _save(fig, "fig3_human_percentile")


def threshold_transfer() -> None:
    table = _table("threshold_transfer")
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.2), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(DETECTORS))
    offsets = {"student": -0.22, "forum": 0.0, "fiction": 0.22}
    for ax, value in ((left, "fpr"), (right, "tpr_all_ai")):
        for genre in GENRES:
            part = table[table["held_out_genre"] == genre].set_index("detector").loc[DETECTORS]
            ax.hlines(y + offsets[genre], part[f"{value}_lo"], part[f"{value}_hi"], color=GENRE_COLOR[genre], linewidth=1.2, alpha=0.7)
            ax.plot(part[value], y + offsets[genre], "o", color=GENRE_COLOR[genre], markersize=3.8, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    left.axvline(0.05, color=INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)))
    left.set(xlabel="Human passages flagged in the held-out genre", xlim=(-0.02, 0.9))
    right.set(xlabel="AI passages caught in the held-out genre", xlim=(-0.02, 0.9))
    left.set_yticks(y, _detector_labels())
    left.invert_yaxis()
    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=5, label=GENRE_NAMES[g]) for g, c in GENRE_COLOR.items()]
    right.legend(handles=handles, title="Held-out genre", loc="lower right")
    _panel_letter(left, "a")
    _panel_letter(right, "b")
    _save(fig, "fig5_threshold_transfer")


def human_side() -> None:
    flagged = _table("false_positives_by_genre")
    flagged = flagged[flagged["detector"].isin(PROBABILITY)]
    ell = _table("ell_probe")
    quality = _table("quality_probe")
    fig = plt.figure(figsize=(FULL_WIDTH, 3.3))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.14)
    ax_flag = fig.add_subplot(grid[0])
    ax_ell = fig.add_subplot(grid[1], sharey=ax_flag)
    ax_quality = fig.add_subplot(grid[2], sharey=ax_flag)

    shares = np.full((len(DETECTORS), len(GENRES)), np.nan)
    for i, detector in enumerate(DETECTORS):
        for j, genre in enumerate(GENRES):
            match = flagged[(flagged["detector"] == detector) & (flagged["genre"] == genre)]
            if len(match):
                shares[i, j] = match["share_flagged_at_0.5"].iloc[0]
    ax_flag.imshow(shares, cmap=matplotlib.colormaps["Oranges"], vmin=0, vmax=0.6, aspect="auto")
    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            if np.isnan(shares[i, j]):
                ax_flag.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color=INK_SECONDARY)
            else:
                ax_flag.text(j, i, f"{shares[i, j]:.0%}", ha="center", va="center", fontsize=7, color="white" if shares[i, j] > 0.35 else INK)
    ax_flag.set_xticks(range(len(GENRES)), [GENRE_NAMES[g].split()[0] for g in GENRES])
    ax_flag.set_yticks(range(len(DETECTORS)), _detector_labels())
    ax_flag.tick_params(length=0)
    ax_flag.set_title("Called AI at the 0.5 cutoff", loc="left", fontsize=8)
    for side in ax_flag.spines.values():
        side.set_visible(False)

    _dot_whisker(ax_ell, ell, "p_ell_scores_higher", "lo", "hi", reference=0.5)
    ax_ell.set(xlim=(0.1, 0.7), xlabel="P(learner essay scores higher)")
    ax_ell.set_title("English learners", loc="left", fontsize=8)
    _family_legend(ax_ell, loc="lower right", fontsize=6.5, handletextpad=0.2)
    _dot_whisker(ax_quality, quality, "within_prompt_spearman", "lo", "hi", reference=0.0)
    ax_quality.set(xlim=(-0.2, 0.6), xlabel="Spearman: essay quality vs score")
    ax_quality.set_title("Essay quality", loc="left", fontsize=8)

    for ax in (ax_ell, ax_quality):
        ax.tick_params(axis="y", left=False, labelleft=False)
    _panel_letter(ax_flag, "a")
    _panel_letter(ax_ell, "b")
    _panel_letter(ax_quality, "c")
    _save(fig, "fig6_human_side")


def rewrites(chunks: pd.DataFrame) -> None:
    levels = [0, 25, 75, 100]
    fig, axes = plt.subplots(3, 3, figsize=(FULL_WIDTH, 5.6), sharex=True)
    for ax, detector in zip(axes.flat, DETECTORS):
        means = chunks.groupby(["essay", "percent_ai"])[detector].mean().unstack()
        color = FAMILY_COLOR[registry.BY_NAME[detector].family]
        for essay in means.index:
            ax.plot(levels, means.loc[essay, levels], color=color, alpha=0.35, linewidth=0.9)
        ax.plot(levels, means[levels].mean(), color=color, linewidth=2.2, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_title(registry.label(detector), fontsize=7.5, loc="left")
        ax.grid(axis="y")
        ax.set_xticks(levels)
        if detector in registry.PROBABILITY_NAMES:
            ax.set_ylim(-0.03, 1.03)
    for ax in axes[-1]:
        ax.set_xlabel("Intended % AI")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean score")
    fig.tight_layout()
    _save(fig, "fig7_rewrites")


def roc_curves(main: pd.DataFrame) -> None:
    humans = main[main["label"] == "human"]
    fig, axes = plt.subplots(3, 3, figsize=(FULL_WIDTH, 6.6))
    groups = {
        "Plain AI": ("plain",),
        "Evasive AI (avoid tells, casual, paraphrase)": EVASIVE_STYLES,
    }
    colors = {"Plain AI": "#2a78d6", "Evasive AI (avoid tells, casual, paraphrase)": "#eb6834"}
    for ax, detector in zip(axes.flat, DETECTORS):
        for name, styles in groups.items():
            part = pd.concat([humans, main[main["prompt_style"].isin(styles)]])
            is_ai = (part["label"] == "ai").to_numpy()
            fpr, tpr = stats.roc_curve(part[detector].to_numpy(), is_ai)
            ax.plot(fpr, tpr, color=colors[name], linewidth=1.4, label=name if detector == DETECTORS[0] else None)
            ax.text(0.97, 0.06 if name == "Plain AI" else 0.16, f"AUC {stats.roc_auc(part[detector].to_numpy(), is_ai):.2f}",
                    color=colors[name], fontsize=7, ha="right", transform=ax.transAxes)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=0.8, zorder=0)
        ax.set_title(registry.label(detector), fontsize=7.5, loc="left")
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    for ax in axes[-1]:
        ax.set_xlabel("False positive rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("True positive rate")
    handles = [Line2D([], [], color=c, linewidth=1.6, label=n) for n, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "figS1_roc")


def correlations() -> None:
    table = _table("detector_correlations").set_index("detector").loc[DETECTORS, DETECTORS]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    image = ax.imshow(table.to_numpy(), cmap=AUC_MAP, vmin=-1, vmax=1)
    for i in range(len(DETECTORS)):
        for j in range(len(DETECTORS)):
            value = table.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(value) > 0.6 else INK)
    ax.set_xticks(range(len(DETECTORS)), _detector_labels(), rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(DETECTORS)), _detector_labels())
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Spearman correlation")
    bar.outline.set_visible(False)
    _save(fig, "figS2_correlations")


def run() -> None:
    _style()
    subsets, chunks = load_corpus()
    main = subsets["main"]
    overview()
    style_matrix()
    genre_matrix()
    human_percentile_boxes(main)
    threshold_transfer()
    human_side()
    rewrites(chunks)
    roc_curves(main)
    correlations()
