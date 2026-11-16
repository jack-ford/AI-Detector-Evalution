import argparse
import logging


def _collect(args: argparse.Namespace) -> None:
    from .collect import collect_human
    collect_human()


def _convert_legacy(args: argparse.Namespace) -> None:
    from .corpus import convert_legacy_results
    convert_legacy_results()


def _score(args: argparse.Namespace) -> None:
    from .scoring import score_everything
    score_everything()


def _plan_ai(args: argparse.Namespace) -> None:
    from .assignments import plan_ai_passages
    plan_ai_passages()


def _register_ai(args: argparse.Namespace) -> None:
    from .assignments import register_ai_passages
    register_ai_passages()


def _evaluate(args: argparse.Namespace) -> None:
    from . import evaluate, plots
    evaluate.run()
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

    legacy = commands.add_parser(
        "convert-legacy", help="convert the original Sapling result files to csv"
    )
    legacy.set_defaults(run=_convert_legacy)

    plan = commands.add_parser("plan-ai", help="decide what each AI passage should be")
    plan.set_defaults(run=_plan_ai)

    register = commands.add_parser("register-ai", help="add the written AI passages to the metadata")
    register.set_defaults(run=_register_ai)

    score = commands.add_parser("score", help="run the local detectors over every text")
    score.set_defaults(run=_score)

    evaluate = commands.add_parser("evaluate", help="compute the result tables and figures")
    evaluate.set_defaults(run=_evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.run(args)
