#!/usr/bin/env python3
"""
Ranges for the pension fund study (full model: shared state, longevity, 10 years).

Part 1, for the decision maker: decision rules
  - risk appetite: market-consistent limit x statutory limit (3 x 3), budget 15%
  - budget: reallocation of 10 / 15 / 20% at the agreed limits
Part 2, for the practitioner: state calibration (one parameter at a time, low / high)
  - fast market state: crash share, persistence, equity loss in a crash, rate fall in a crash
  - slow longevity state: drift persistence (phi), drift size

Uses pension_study.py (same folder) for the model and the criterion.
Usage: python pension_sensitivity.py [--paths 100000]
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ps", HERE / "pension_study.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

FULL = ("multivariate", "total", "multi")

# appetite and budget grid
MARKET_LIMITS = [0.10, 0.15, 0.20]
STAT_LIMITS = [0.05, 0.10, 0.15]
BUDGETS = [0.10, 0.15, 0.20]

# state parameters: (label, setter, low, high, format)
STATE_PARAMS = [
    ("Crash share", lambda c, v: c["market_regime"].__setitem__("adverse_share", v), 0.10, 0.18, "{:.0%}"),
    ("Crash persistence", lambda c, v: c["market_regime"].__setitem__("adverse_stay", v), 0.25, 0.65, "{:.0%}"),
    ("Equity loss in a crash", lambda c, v: c["equity"]["adverse"].__setitem__("mu", v), -15.0, -35.0, "{:.0f}%"),
    ("Rate fall in a crash", lambda c, v: c["rates"]["adverse"].__setitem__("mu", v), -0.30, -0.90, "{:+.2f} pp"),
    ("Longevity drift persistence", lambda c, v: c["longevity"].__setitem__("phi", v), 0.85, 0.98, "{:.2f}"),
    ("Longevity drift size", lambda c, v: c["longevity"].__setitem__("drift_sd", v), 0.30, 0.70, "{:.1f}%"),
]
STATE_BASE = {"Crash share": 0.14, "Crash persistence": 0.50, "Equity loss in a crash": -25.0,
              "Rate fall in a crash": -0.60, "Longevity drift persistence": 0.95, "Longevity drift size": 0.50}


def optimum_under(r, lm, ls):
    ok = (r["b_mc"] <= lm) & (r["b_st"] <= ls)
    if not ok.any():
        return None
    g = np.where(ok, r["g_mc"], -np.inf)
    i, j = np.unravel_index(np.argmax(g), g.shape)
    return {"hedge": float(r["hrs"][i]), "ils": float(r["fracs"][j]), "b_mc": float(r["b_mc"][i, j]),
            "b_st": float(r["b_st"][i, j]), "growth": float(r["g_mc"][i, j]), "n_ok": int(ok.sum())}


def run_full(cfg, n):
    h = cfg["simulation"]["horizon_years"]
    w = ps.simulate_state_world(cfg, np.random.default_rng(cfg["simulation"]["seed"]), n, h)
    return ps.evaluate_joint(cfg, {"multivariate": w}, *FULL)


def fmt_opt(o):
    if o is None:
        return "no admissible\ncombination"
    return (f"hedge {o['hedge']:.0%}\n{o['ils']:.0%} ILS / {1 - o['ils']:.0%} ETF\n"
            f"growth {100 * o['growth']:.2f}%\nP {o['b_mc']:.1%} / {o['b_st']:.1%}")


def draw_appetite(cfg, appetite, budget, path):
    br = cfg["branding"]
    ps.setup_font(br)
    fig = plt.figure(figsize=(br["figsize"][0], 7.2), dpi=br["dpi"], facecolor=br["background"])
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.25], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis("off")
    base = (0.15, 0.10)
    for a, ls in enumerate(STAT_LIMITS):
        for b, lm in enumerate(MARKET_LIMITS):
            o = appetite[(lm, ls)]
            x, y = b, 2 - a
            is_base = (lm, ls) == base
            face = br["accent"] if is_base else ("#F1F2F6" if o else "#FFFFFF")
            ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, facecolor=face, alpha=0.25 if is_base else 1,
                                       edgecolor=br["primary"] if is_base else br["neutral"], lw=1.5 if is_base else 0.8))
            ax.text(x + 0.5, y + 0.5, fmt_opt(o), ha="center", va="center", fontsize=8.5,
                    color=br["text"] if o else br["muted"], fontweight="bold" if is_base else "normal")
    for b, lm in enumerate(MARKET_LIMITS):
        ax.text(b + 0.5, 3.08, f"market-consistent limit {lm:.0%}", ha="center", fontsize=9, color=br["primary"])
    for a, ls in enumerate(STAT_LIMITS):
        ax.text(-0.05, 2 - a + 0.5, f"statutory\nlimit {ls:.0%}", ha="right", va="center", fontsize=9,
                color=br["primary"])
    ax.set_title("Risk appetite (budget 15% of assets)", fontsize=10.5, loc="left", pad=22)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    for b, bud in enumerate(BUDGETS):
        o = budget[bud]
        is_base = bud == 0.15
        ax2.add_patch(plt.Rectangle((b + 0.03, 0.03), 0.94, 0.94,
                                    facecolor=br["accent"] if is_base else "#F1F2F6", alpha=0.25 if is_base else 1,
                                    edgecolor=br["primary"] if is_base else br["neutral"], lw=1.5 if is_base else 0.8))
        ax2.text(b + 0.5, 0.5, fmt_opt(o), ha="center", va="center", fontsize=8.5, color=br["text"],
                 fontweight="bold" if is_base else "normal")
        ax2.text(b + 0.5, 1.08, f"budget {bud:.0%} of assets", ha="center", fontsize=9, color=br["primary"])
    ax2.set_title("Budget (agreed limits: market 15%, statutory 10%)", fontsize=10.5, loc="left", pad=22)

    fig.suptitle("Decision rules: what the board's choices change", x=0.02, ha="left", fontsize=12.5,
                 fontweight="bold", color=br["text"])
    fig.text(0.02, 0.012, "Full model (shared state, longevity, 10 years). Cell: optimal levers, growth of the "
             "market-consistent funding ratio, P(underfunded) market / statutory. Highlighted: agreed rules. "
             + br["footer"], fontsize=6.6, color=br["muted"])
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.05)
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


def draw_state(cfg, base_opt, rows, path):
    br = cfg["branding"]
    ps.setup_font(br)
    fig, ax = plt.subplots(figsize=(br["figsize"][0], 0.55 * len(rows) + 1.9), dpi=br["dpi"],
                           facecolor=br["background"])
    ax.axis("off")
    cols = ["Parameter", "Low", "Optimum (low)", "High", "Optimum (high)"]
    xs = [0.0, 0.25, 0.37, 0.62, 0.74]
    yt = 1.0
    for x, c in zip(xs, cols):
        ax.text(x, yt, c, fontsize=9, fontweight="bold", color=br["primary"], transform=ax.transAxes)

    def short(o):
        if o is None:
            return "none admissible"
        return f"hedge {o['hedge']:.0%}, {o['ils']:.0%}/{1 - o['ils']:.0%}, g {100 * o['growth']:.2f}%"

    step = 0.9 / (len(rows) + 1)
    y = yt - step
    ax.text(0.0, y, "Base calibration", fontsize=8.8, color=br["text"], fontweight="bold", transform=ax.transAxes)
    ax.text(0.37, y, short(base_opt), fontsize=8.8, color=br["text"], fontweight="bold", transform=ax.transAxes)
    for lab, fmt, lo, o_lo, hi, o_hi in rows:
        y -= step
        ax.text(xs[0], y, f"{lab} (base {fmt.format(STATE_BASE[lab])})", fontsize=8.5, color=br["text"],
                transform=ax.transAxes)
        ax.text(xs[1], y, fmt.format(lo), fontsize=8.5, color=br["text"], transform=ax.transAxes)
        ax.text(xs[2], y, short(o_lo), fontsize=8.5, color=br["text"], transform=ax.transAxes)
        ax.text(xs[3], y, fmt.format(hi), fontsize=8.5, color=br["text"], transform=ax.transAxes)
        ax.text(xs[4], y, short(o_hi), fontsize=8.5, color=br["text"], transform=ax.transAxes)
    fig.suptitle("The hard part: how the optimum moves with the state calibration", x=0.02, ha="left",
                 fontsize=12.5, fontweight="bold", color=br["text"])
    fig.text(0.02, 0.02, "Full model, agreed limits (market 15%, statutory 10%), budget 15%. One parameter "
             "changed at a time. Split shown as ILS/ETF of the sleeve. " + br["footer"],
             fontsize=6.6, color=br["muted"])
    fig.subplots_adjust(left=0.02, right=0.99, top=0.84, bottom=0.08)
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100_000)
    args = ap.parse_args()
    base_cfg = copy.deepcopy(ps.CONFIG)
    n = args.paths

    # ---------- Part 1: decision rules ----------
    r15 = run_full(base_cfg, n)
    appetite = {(lm, ls): optimum_under(r15, lm, ls) for lm in MARKET_LIMITS for ls in STAT_LIMITS}
    budget = {}
    for bud in BUDGETS:
        if bud == 0.15:
            budget[bud] = appetite[(0.15, 0.10)]
            continue
        c = copy.deepcopy(base_cfg)
        c["decision"]["share"] = bud
        budget[bud] = optimum_under(run_full(c, n), 0.15, 0.10)
    draw_appetite(base_cfg, appetite, budget, "pension_ranges_decision.png")

    # ---------- Part 2: state calibration ----------
    base_opt = appetite[(0.15, 0.10)]
    rows = []
    for lab, setter, lo, hi, fmt in STATE_PARAMS:
        res = []
        for v in (lo, hi):
            c = copy.deepcopy(base_cfg)
            setter(c, v)
            res.append(optimum_under(run_full(c, n), 0.15, 0.10))
        rows.append((lab, fmt, lo, res[0], hi, res[1]))
        print(f"  done: {lab}")
    draw_state(base_cfg, base_opt, rows, "pension_ranges_state.png")

    # ---------- CSV + console ----------
    with open("pension_ranges_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "setting", "hedge", "ils_fraction", "p_market", "p_statutory", "growth", "n_admissible"])

        def row(part, setting, o):
            if o is None:
                w.writerow([part, setting, "", "", "", "", "", 0])
            else:
                w.writerow([part, setting, o["hedge"], o["ils"], f"{o['b_mc']:.4f}", f"{o['b_st']:.4f}",
                            f"{o['growth']:.5f}", o["n_ok"]])
        for (lm, ls), o in appetite.items():
            row("appetite", f"market {lm:.0%} / statutory {ls:.0%}", o)
        for bud, o in budget.items():
            row("budget", f"{bud:.0%}", o)
        for lab, fmt, lo, o_lo, hi, o_hi in rows:
            row("state", f"{lab} = {fmt.format(lo)}", o_lo)
            row("state", f"{lab} = {fmt.format(hi)}", o_hi)

    def s(o):
        return "none" if o is None else (f"hedge {o['hedge']:.0%} | ILS {o['ils']:.0%} / ETF {1 - o['ils']:.0%} | "
                                         f"P {o['b_mc']:.1%} / {o['b_st']:.1%} | g {100 * o['growth']:.2f}% | n_ok {o['n_ok']}")
    print("Risk appetite (budget 15%):")
    for (lm, ls), o in appetite.items():
        print(f"  market {lm:.0%}, statutory {ls:.0%}: {s(o)}")
    print("Budget (limits 15% / 10%):")
    for bud, o in budget.items():
        print(f"  {bud:.0%}: {s(o)}")
    print("State calibration (limits 15% / 10%, budget 15%):")
    print(f"  base: {s(base_opt)}")
    for lab, fmt, lo, o_lo, hi, o_hi in rows:
        print(f"  {lab}: low {fmt.format(lo)} -> {s(o_lo)}")
        print(f"  {'':{len(lab)}}  high {fmt.format(hi)} -> {s(o_hi)}")
    print("wrote pension_ranges_decision.png, pension_ranges_state.png, pension_ranges_results.csv")


if __name__ == "__main__":
    main()
