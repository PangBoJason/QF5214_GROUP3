"""Pipeline entry point — runs all phases in sequence.

Usage examples:
    python main.py                              # full pipeline
    python main.py --from-step sde_simulation   # skip data/feature/model steps
    python main.py --only factor_backtest       # single step (no agent)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import uuid

STEPS: list[tuple[str, str]] = [
    ("get_stock_data",      "Phase 1b — stock prices + earnings dates (yfinance)"),
    ("edgar_supplement",    "Phase 1c — EDGAR EPS + 8-K dates + VADER sentiment scores"),
    ("feature_engineering", "Phase 2  — monthly/quarterly RPK + load-factor features"),
    ("sde_simulation",      "Phase 3  — log-OU Monte Carlo (LF momentum drift, 1 000 paths)"),
    ("factor_backtest",     "Phase 4  — four-factor IC / Sharpe: RPK + EPS surprise + 8-K tone + momentum"),
    ("data_quality",        "Phase 4b — data quality validation (hard gate)"),
    ("charts",              "Phase 5  — generate PNG visualisations + scorecard + summary table"),
]


def run_step(module: str, extra_args: list[str] | None = None) -> bool:
    print(f"\n{'='*60}\n[step] {module}.py\n{'='*60}")
    result = subprocess.run([sys.executable, f"{module}.py"] + (extra_args or []))
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Airline earnings quant pipeline.")
    parser.add_argument(
        "--from-step",
        choices=[s[0] for s in STEPS],
        default=None,
        help="Skip all steps before this one.",
    )
    parser.add_argument(
        "--only",
        choices=[s[0] for s in STEPS],
        default=None,
        help="Run exactly this one step (no agent afterward).",
    )
    parser.add_argument(
        "--question",
        default=(
            "Analyse all six airline carriers (DAL, UAL, AAL, LUV, JBLU, ALK). "
            "Retrieve the latest flight metrics, Monte Carlo simulation results, "
            "and four-factor backtest performance, then produce a structured investment report."
        ),
        help="Question passed to the agent after the pipeline finishes.",
    )
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:12]
    print(f"[pipeline] run_id={run_id}")

    steps = STEPS
    if args.only:
        steps = [s for s in STEPS if s[0] == args.only]
    elif args.from_step:
        idx = next(i for i, s in enumerate(STEPS) if s[0] == args.from_step)
        steps = STEPS[idx:]

    for module, description in steps:
        print(f"\n[run] {description}")
        extra: list[str] = ["--run-id", run_id]
        if not run_step(module, extra):
            print(f"\n[abort] {module}.py exited with an error. Fix it before continuing.")
            sys.exit(1)

    if not args.only:
        print(f"\n{'='*60}\n[agent] generating analysis report\n{'='*60}")
        run_step("agent", ["--question", args.question])


if __name__ == "__main__":
    main()