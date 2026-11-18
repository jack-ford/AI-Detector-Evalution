import argparse
import logging


def _collect(args: argparse.Namespace) -> None:
    from .collect import collect_human
    collect_human()


def _build_rewrites(args: argparse.Namespace) -> None:
    from .corpus import build_rewrite_chunks
    build_rewrite_chunks()


def _score(args: argparse.Namespace) -> None:
    from .scoring import score_everything
    score_everything(args.only)


def _plan_ai(args: argparse.Namespace) -> None:
    from .assignments import plan_ai_passages
    plan_ai_passages()


def _register_ai(args: argparse.Namespace) -> None:
    from .assignments import register_ai_passages
    register_ai_passages()


def _evaluate(args: argparse.Namespace) -> None:
    from . import evaluate
    evaluate.run()


def _figures(args: argparse.Namespace) -> None:
    from . import plots
    plots.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="statproject",
        description="Can AI-text detectors tell human writing from AI writing?",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="download the human passages")
    collect.set_defaults(run=_collect)

    rewrites = commands.add_parser(
        "build-rewrites", help="cut the essay-rewrite set into 100-word chunks"
    )
    rewrites.set_defaults(run=_build_rewrites)

    plan = commands.add_parser("plan-ai", help="decide what each AI passage should be")
    plan.set_defaults(run=_plan_ai)

    register = commands.add_parser("register-ai", help="add the written AI passages to the metadata")
    register.set_defaults(run=_register_ai)

    score = commands.add_parser("score", help="run the local detectors over every text")
    score.add_argument("--only", nargs="+", metavar="DETECTOR", help="score with just these detectors")
    score.set_defaults(run=_score)

    evaluate = commands.add_parser("evaluate", help="compute the result tables")
    evaluate.set_defaults(run=_evaluate)

    figures = commands.add_parser("figures", help="draw the figures from the result tables")
    figures.set_defaults(run=_figures)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.run(args)
