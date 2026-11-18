import shutil
from pathlib import Path

import pandas as pd

from src import registry

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent
FIGURES = ["fig1_overview", "fig2_style", "fig3_human_percentile", "fig4_genre", "fig5_threshold_transfer", "fig6_human_side"]


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "tables" / f"{name}.csv")


def num(x: float, digits: int = 2, signed: bool = False) -> str:
    text = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return text.replace("-", "$-$").replace("+", "$+$")


def interval(value: float, lo: float, hi: float, digits: int = 2) -> str:
    return f"{num(value, digits)} [{num(lo, digits)}, {num(hi, digits)}]"


def ci(row: pd.Series, value: str, digits: int = 2) -> str:
    return interval(row[value], row[value + "_lo"], row[value + "_hi"], digits)


def bounds(row: pd.Series, value: str, digits: int = 2) -> str:
    return interval(row[value], row["lo"], row["hi"], digits)


def percent_ci(row: pd.Series, value: str) -> str:
    return rf"{row[value]:.0%} [{row[value + '_lo']:.0%}, {row[value + '_hi']:.0%}]".replace("%", r"\%")


def p_format(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name: str, header: str, body: list[str]) -> None:
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rows_by_family(render) -> list[str]:
    lines = []
    for family in (registry.SUPERVISED, registry.ZERO_SHOT):
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{family.capitalize()}}}}} \\")
        lines += [render(d.name) for d in registry.DETECTORS if d.family == family]
        if family == registry.SUPERVISED:
            lines.append(r"\addlinespace")
    return lines


def overall() -> None:
    data = table("overall").set_index("detector")
    best_auc = data["auc"].max()
    best_tpr = data["tpr_at_5pct_fpr"].max()

    def render(name: str) -> str:
        row = data.loc[name]
        auc = ci(row, "auc")
        tpr = percent_ci(row, "tpr_at_5pct_fpr")
        auc = rf"\textbf{{{auc}}}" if row["auc"] == best_auc else auc
        tpr = rf"\textbf{{{tpr}}}" if row["tpr_at_5pct_fpr"] == best_tpr else tpr
        return rf"{registry.label(name)} & {auc} & {tpr} & {row['tpr_at_1pct_fpr'] * 100:.0f}\% & {p_format(row['p_holm'])} \\"

    header = r"Detector & \auc\ [95\% CI] & \tpr\ at 5\% \fpr\ [95\% CI] & \tpr\ at 1\% \fpr & $p_{\text{Holm}}$ \\"
    write("overall", header, rows_by_family(render))


def rewrites() -> None:
    versions = table("rewrites_versus_human")
    trend = table("rewrites_by_version").set_index("detector")

    def render(name: str) -> str:
        cells = []
        for level in (25, 75, 100):
            row = versions[(versions["detector"] == name) & (versions["version"] == level)].iloc[0]
            cells.append(f"{row['auc']:.2f}")
        row = trend.loc[name]
        return rf"{registry.label(name)} & {cells[0]} & {cells[1]} & {cells[2]} & {num(row['trend_spearman'], signed=True)} \\"

    write("rewrites", r"Detector & 25\% & 75\% & 100\% & $\rho$ \\", rows_by_family(render))


def human_side() -> None:
    quality = table("quality_probe").set_index("detector")
    ell = table("ell_probe").set_index("detector")
    matched = table("ell_quality_matched").set_index("detector")

    def render(name: str) -> str:
        q, e, m = quality.loc[name], ell.loc[name], matched.loc[name]
        return (
            rf"{registry.label(name)} & {bounds(q, 'within_prompt_spearman')} & {bounds(e, 'p_ell_scores_higher')} "
            rf"& {num(m['rank_gap_matched'] * 100, 0, signed=True)} & {p_format(m['p_holm'])} \\"
        )

    header = r"Detector & Quality $\rho$ [95\% CI] & Learners [95\% CI] & Matched gap & $p_{\text{Holm}}$ \\"
    write("human_side", header, rows_by_family(render))


def copy_figures() -> None:
    (OUT / "figures").mkdir(exist_ok=True)
    for name in FIGURES:
        shutil.copy(RESULTS / "figures" / f"{name}.pdf", OUT / "figures" / f"{name}.pdf")


if __name__ == "__main__":
    overall()
    rewrites()
    human_side()
    copy_figures()
