"""Command-line entry points for generation, validation, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agent_loader import AgentProcessError, SubprocessAgent
from .catalog import Catalog
from .evaluator import evaluate_dataset
from .generator import GeneratorConfig, GenerationError, SCENARIOS, generate_samples
from .reporting import format_report, write_json
from .schema import Sample, read_jsonl, write_jsonl
from .validator import ValidationConfig, validate_dataset


def _samples(path: str | Path) -> list[Sample]:
    return [Sample.from_dict(row) for row in read_jsonl(str(path))]


def _validation_config(args: argparse.Namespace) -> ValidationConfig:
    return ValidationConfig(
        min_initial_candidates=int(args.min_initial_candidates),
        max_initial_candidates=int(args.max_initial_candidates),
        enforce_initial_bounds=not bool(getattr(args, "relaxed_initial_bounds", False)),
    )


def _print_validation(report: dict) -> None:
    print(f"validated {report['count']} samples: {report['valid_count']} valid, {report['invalid_count']} invalid")
    for result in report["results"]:
        if not result["valid"]:
            print(f"  {result['sample_id']}: {'; '.join(result['errors'])}")


def command_generate(args: argparse.Namespace) -> int:
    catalog = Catalog(args.catalog)
    config = GeneratorConfig(
        min_initial_candidates=args.min_initial_candidates,
        max_initial_candidates=args.max_initial_candidates,
        max_signature_facts=args.max_signature_facts,
    )
    scenarios = args.scenario or SCENARIOS
    try:
        samples = generate_samples(catalog, args.samples, seed=args.seed, config=config, scenarios=scenarios)
    except GenerationError as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 2
    validation = validate_dataset(catalog, samples, config=ValidationConfig(
        min_initial_candidates=args.min_initial_candidates,
        max_initial_candidates=args.max_initial_candidates,
        enforce_initial_bounds=True,
    ))
    if not validation["valid"]:
        _print_validation(validation)
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output), [sample.to_dict() for sample in samples])
    print(f"generated {len(samples)} samples at {output}")
    print(f"scenarios: {', '.join(sample.scenario_type for sample in samples)}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    catalog = Catalog(args.catalog)
    samples = _samples(args.dataset)
    report = validate_dataset(catalog, samples, config=_validation_config(args))
    _print_validation(report)
    if args.output:
        write_json(args.output, report)
    return 0 if report["valid"] else 1


def command_evaluate(args: argparse.Namespace) -> int:
    catalog = Catalog(args.catalog)
    samples = _samples(args.dataset)
    validation = validate_dataset(catalog, samples, config=_validation_config(args))
    if not validation["valid"] and not args.allow_invalid:
        _print_validation(validation)
        print("refusing to evaluate an invalid dataset; use --allow-invalid only for debugging", file=sys.stderr)
        return 2
    try:
        with SubprocessAgent(
            args.agent_repo,
            args.catalog,
            timeout=args.agent_timeout,
            force_intent_model=args.force_intent_model,
            intent_confidence=args.intent_confidence,
        ) as agent:
            report = evaluate_dataset(agent, catalog, samples, max_turns=args.max_turns, top_k=args.top_k)
    except AgentProcessError as exc:
        print(f"agent worker failed: {exc}", file=sys.stderr)
        return 2
    report["validation"] = validation
    write_json(args.output, report)
    print(format_report(report))
    print(f"wrote {args.output}")
    return 0 if not report["summary"].get("error_count") else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nl-benchmark", description="Target-exact TechJam natural-language benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic frozen JSONL dataset")
    generate.add_argument("--catalog", required=True)
    generate.add_argument("--samples", type=int, default=8)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", required=True)
    generate.add_argument("--scenario", action="append", choices=SCENARIOS, help="repeat to choose scenario order")
    generate.add_argument("--min-initial-candidates", type=int, default=2)
    generate.add_argument("--max-initial-candidates", type=int, default=200)
    generate.add_argument("--max-signature-facts", type=int, default=8)
    generate.set_defaults(handler=command_generate)

    validate = subparsers.add_parser("validate", help="validate a frozen JSONL dataset")
    validate.add_argument("--catalog", required=True)
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--output")
    validate.add_argument("--min-initial-candidates", type=int, default=2)
    validate.add_argument("--max-initial-candidates", type=int, default=200)
    validate.add_argument("--relaxed-initial-bounds", action="store_true")
    validate.set_defaults(handler=command_validate)

    evaluate = subparsers.add_parser("evaluate", help="run an external Agent on a frozen dataset")
    evaluate.add_argument("--agent-repo", required=True)
    evaluate.add_argument("--catalog", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--max-turns", type=int, default=10)
    evaluate.add_argument("--top-k", type=int, default=10)
    evaluate.add_argument("--agent-timeout", type=float, default=120.0)
    evaluate.add_argument(
        "--force-intent-model",
        action="store_true",
        help="benchmark ablation: call the Agent intent model on every turn",
    )
    evaluate.add_argument(
        "--intent-confidence",
        type=float,
        choices=(0.0, 1.0),
        help="benchmark ablation: override model and mutation confidence (use 1.0 for 100%%)",
    )
    evaluate.add_argument("--min-initial-candidates", type=int, default=2)
    evaluate.add_argument("--max-initial-candidates", type=int, default=200)
    evaluate.add_argument("--relaxed-initial-bounds", action="store_true")
    evaluate.add_argument("--allow-invalid", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
