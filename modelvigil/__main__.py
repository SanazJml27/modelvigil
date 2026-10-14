"""Command-line interface.

  python -m modelvigil list
  python -m modelvigil run --scenario lab_unit_change
  python -m modelvigil run --all
  python -m modelvigil run --scenario concept_shift --llm anthropic --approval interactive
  python -m modelvigil benchmark --seeds 5
  python -m modelvigil verify-audit runs/lab_unit_change_s7/audit.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import verify_chain
from .config import RunConfig
from .simulator import SCENARIOS


def _run_one(args, scenario: str):
    from .engine import run
    from .report import pms_report, timeline_plot

    out = Path(args.out) / f"{scenario}_s{args.seed}"
    cfg = RunConfig(scenario=scenario, seed=args.seed, llm=args.llm, approval=args.approval,
                    n_weeks=args.weeks, n_per_week=args.per_week, quiet=args.quiet)
    res = run(cfg, out_dir=out)
    timeline_plot(res, out / "timeline.png")
    (out / "pms_report.md").write_text(pms_report(res, verify_chain(out / "audit.jsonl")))
    sc = res["scenario"]
    print(f"  ground truth: {sc['root_cause']} -> {sc['correct_action']}")
    print(f"  report: {out / 'pms_report.md'}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="modelvigil", description="Agentic post-market surveillance "
                                 "for clinical prediction models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list scenarios")

    r = sub.add_parser("run", help="run one surveillance simulation")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario", choices=list(SCENARIOS))
    g.add_argument("--all", action="store_true")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--llm", choices=["mock", "anthropic"], default="mock")
    r.add_argument("--approval", choices=["policy", "interactive", "auto"], default="policy")
    r.add_argument("--weeks", type=int, default=16)
    r.add_argument("--per-week", type=int, default=600)
    r.add_argument("--out", default="runs")
    r.add_argument("--quiet", action="store_true")

    b = sub.add_parser("benchmark", help="benchmark vs baselines across seeds")
    b.add_argument("--seeds", type=int, default=5)
    b.add_argument("--out", default="results")
    b.add_argument("--workers", type=int, default=None)

    v = sub.add_parser("verify-audit", help="verify a hash-chained audit log")
    v.add_argument("path")

    args = ap.parse_args(argv)
    if args.cmd == "list":
        for k, s in SCENARIOS.items():
            print(f"{k:18s} truth={s.root_cause:24s} action={s.correct_action:20s}\n{'':18s} {s.description}\n")
    elif args.cmd == "run":
        for s in (SCENARIOS if args.all else [args.scenario]):
            _run_one(args, s)
    elif args.cmd == "benchmark":
        from .benchmark import benchmark
        benchmark(seeds=args.seeds, out_dir=args.out, workers=args.workers)
    elif args.cmd == "verify-audit":
        ok, msg = verify_chain(args.path)
        print(("OK: " if ok else "TAMPERED: ") + msg)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
