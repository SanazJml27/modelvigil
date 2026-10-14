"""Benchmark: 7 scenarios x N seeds, ModelVigil vs two standard monitoring
baselines.

  naive_drift   alarm on any univariate KS test (Bonferroni) or classifier
                two-sample AUROC > 0.60 - typical 'data drift dashboard'
  perf_only     alarm when lagged realised AUROC/O-E leaves control limits -
                typical 'periodic performance audit'

Neither baseline produces a root cause or an action; ModelVigil is scored on
both, plus detection delay and false alarms.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import detectors as D
from .config import FEATURE_NAMES, LABEL_LAG_WEEKS, LABEL_WINDOW_WEEKS, THRESHOLDS, RunConfig
from .engine import run
from .model import fit_model
from .simulator import SCENARIOS, HospitalStream

NON_ACTIONABLE = {"benign_covariate_shift", "new_population_unverified", "insufficient_evidence"}


def _baselines(cfg: RunConfig) -> dict:
    s = HospitalStream(cfg.scenario, seed=cfg.seed, n_per_week=cfg.n_per_week,
                       n_weeks=cfg.n_weeks, shift_week=cfg.shift_week)
    ref_all = s.reference(cfg.n_ref)
    cut = int(len(ref_all) * 2 / 3)
    dev, ref = ref_all.iloc[:cut], ref_all.iloc[cut:].reset_index(drop=True)
    m, _ = fit_model(dev, seed=cfg.seed)
    rm = D.realised_metrics(ref["y"], m.predict_proba(ref))
    naive, perf, prev_bad = [], [], False
    for w in range(1, cfg.n_weeks + 1):
        cur = s.week(w)
        ks = min(stats.ks_2samp(ref[f], cur[f]).pvalue for f in FEATURE_NAMES)
        if ks < 0.01 / len(FEATURE_NAMES) or D.domain_classifier(ref, cur, seed=w)["auc"] > THRESHOLDS.domain_auc:
            naive.append(w)
        last = w - LABEL_LAG_WEEKS
        lw = [x for x in range(max(1, last - LABEL_WINDOW_WEEKS + 1), last + 1)]
        if len(lw) >= 2:
            d = pd.concat([s.week(x) for x in lw], ignore_index=True)
            r = D.realised_metrics(d["y"], m.predict_proba(d))
            lo, hi = r["oe_ci"]
            oe_bad = (r["oe"] < THRESHOLDS.oe_low or r["oe"] > THRESHOLDS.oe_high) and (lo > 1 or hi < 1)
            drop = rm["auroc"] - r["auroc"]
            bad = oe_bad or (drop > THRESHOLDS.auc_drop and drop > THRESHOLDS.z_overall * r["auroc_se"])
            if bad and prev_bad:          # same two-consecutive-looks run rule as ModelVigil
                perf.append(w)
            prev_bad = bad
    return {"naive_drift": naive, "perf_only": perf}


def _score_alarms(weeks: list[int], cfg: RunConfig, harmful: bool) -> dict:
    pre = any(w < cfg.shift_week for w in weeks)
    post = [w for w in weeks if w >= cfg.shift_week]
    if harmful:
        return {"false_alarm": pre, "detected": bool(post),
                "delay": (post[0] - cfg.shift_week) if post else None, "rca_correct": None, "action_correct": None}
    return {"false_alarm": bool(weeks), "detected": None, "delay": None, "rca_correct": None, "action_correct": None}


def _score_vigil(res: dict, cfg: RunConfig) -> dict:
    sc = SCENARIOS[cfg.scenario]
    incs = res["incidents"]
    pre = any(i["opened_week"] < cfg.shift_week for i in incs)
    actionable = [i for i in incs if i["diagnosis"]["root_cause"] not in NON_ACTIONABLE]
    if not sc.harmful:
        rca = not actionable
        return {"false_alarm": bool(actionable) or pre, "detected": None, "delay": None,
                "rca_correct": rca, "action_correct": rca and not res["pipeline_corrections"]
                and res["final_model"]["version"] == "1.0.0", "n_incidents": len(incs)}
    first = actionable[0] if actionable else None
    return {"false_alarm": pre, "detected": first is not None,
            "delay": (first["opened_week"] - cfg.shift_week) if first else None,
            "rca_correct": bool(first and first["diagnosis"]["root_cause"] == sc.root_cause),
            "action_correct": bool(first and first["plan"]["category"] == sc.correct_action),
            "n_incidents": len(incs)}


def _one(args) -> list[dict]:
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):          # avoid OpenMP oversubscription across worker processes
        return _one_inner(args)


def _one_inner(args) -> list[dict]:
    scenario, seed = args
    cfg = RunConfig(scenario=scenario, seed=seed, quiet=True)
    res = run(cfg)
    rows = []
    harmful = SCENARIOS[scenario].harmful
    v = _score_vigil(res, cfg) | {"method": "ModelVigil"} | res["impact"]
    rows.append(v)
    for name, weeks in _baselines(cfg).items():
        rows.append(_score_alarms(weeks, cfg, harmful) | {"method": name})
    for r in rows:
        r |= {"scenario": scenario, "seed": seed, "harmful": harmful}
    return rows


def benchmark(seeds: int = 5, out_dir: str = "results", workers: int | None = None) -> pd.DataFrame:
    jobs = [(s, seed) for s in SCENARIOS for seed in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for rs in ex.map(_one, jobs) for r in rs]
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark_runs.csv", index=False)
    summary = summarise(df)
    (out / "benchmark_summary.md").write_text(summary)
    _plot(df, out / "benchmark.png")
    print(summary)
    return df


def summarise(df: pd.DataFrame) -> str:
    def pct(x):
        x = x.dropna()
        return f"{100 * x.astype(float).mean():.0f}%" if len(x) else "n/a"

    lines = ["## Overall (all scenarios x seeds)", "",
             "| Method | False-alarm rate (stable + benign) | Detection rate (harmful) | Median delay, weeks | Root-cause accuracy | Correct action |",
             "|---|---|---|---|---|---|"]
    for m in ["ModelVigil", "naive_drift", "perf_only"]:
        d = df[df.method == m]
        nh, h = d[~d.harmful], d[d.harmful]
        delay = h["delay"].dropna()
        lines.append(f"| {m} | {pct(nh['false_alarm'])} | {pct(h['detected'])} | "
                     f"{delay.median():.1f} |" if len(delay) else f"| {m} | {pct(nh['false_alarm'])} | {pct(h['detected'])} | n/a |")
        lines[-1] += (f" {pct(d['rca_correct'])} | {pct(d['action_correct'])} |" if m == "ModelVigil"
                      else " — | — |")
    lines += ["", "## ModelVigil by scenario", "",
              "| Scenario | Truth | Detected | Median delay | Root cause correct | Action correct | False alarm | Mean incidents |",
              "|---|---|---|---|---|---|---|---|"]
    v = df[df.method == "ModelVigil"]
    for s, g in v.groupby("scenario", sort=False):
        sc = SCENARIOS[s]
        dl = g["delay"].dropna()
        lines.append(f"| {s} | {sc.root_cause} | {pct(g['detected'])} | "
                     f"{dl.median():.1f} | " if len(dl) else f"| {s} | {sc.root_cause} | {pct(g['detected'])} | — | ")
        lines[-1] += f"{pct(g['rca_correct'])} | {pct(g['action_correct'])} | {pct(g['false_alarm'])} | {g['n_incidents'].mean():.1f} |"
    lines += ["", "## Patient-level impact after the shift (harmful scenarios, totals across seeds; simulation oracle)", "",
              "| Scenario | Events | Missed: no monitoring | Missed: ModelVigil | False alerts: no monitoring | "
              "False alerts: ModelVigil | Net benefit /1000: no monitoring | Net benefit /1000: ModelVigil |",
              "|---|---|---|---|---|---|---|---|"]
    for s, g in v[v.harmful].groupby("scenario", sort=False):
        lines.append(f"| {s} | {g.post_shift_events.sum():.0f} | {g.missed_events_no_monitoring.sum():.0f} | "
                     f"{g.missed_events_live.sum():.0f} | {g.false_alerts_no_monitoring.sum():.0f} | {g.false_alerts_live.sum():.0f} | "
                     f"{g.net_benefit_per_1000_no_monitoring.mean():.1f} | {g.net_benefit_per_1000_live.mean():.1f} |")
    lines += ["", "Net benefit = TP/n - FP/n x pt/(1-pt) at the deployed 20% threshold (decision-curve analysis): "
              "it prices a false alert at 1/4 of a caught deterioration, the trade-off implied by the threshold."]
    return "\n".join(lines) + "\n"


def _plot(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["ModelVigil", "naive_drift", "perf_only"]
    colors = ["#1b6e8a", "#c98a2b", "#8a8a8a"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    nh = [df[(df.method == m) & ~df.harmful]["false_alarm"].astype(float).mean() * 100 for m in methods]
    det = [df[(df.method == m) & df.harmful]["detected"].astype(float).mean() * 100 for m in methods]
    dl = [df[(df.method == m) & df.harmful]["delay"].dropna().median() for m in methods]
    for ax, vals, title in zip(axes, [nh, det, dl], ["False alarms on stable/benign (%)",
                                                     "Harmful shifts detected (%)", "Median detection delay (weeks)"]):
        ax.bar(methods, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}" if title.endswith("(%)") else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("ModelVigil vs standard monitoring (root cause + action only available in ModelVigil)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
