#!/usr/bin/env python3
"""
Decision matrix: marginal vs aggregate  x  one-year vs multi-year.

Two candidate exposures are added to an existing book that loads on a shared
Markov regime ("adverse state"):
  A  : fat-tailed, idiosyncratic (independent of the shared state)
  B' : benefits in the shared adverse state (e.g. health care / services /
       insurers for a pension fund exposed to longevity)

A no-state benchmark (multivariate normal with the same means, vols and
correlations, iid across years) shows what a conventional model would decide.

The script simulates annual returns, evaluates the views, writes a results CSV
and draws a 2x2 figure.  All parameters and branding live in CONFIG below and
can be overridden with a JSON file:  python decision_matrix.py --config my.json

Requires: numpy, matplotlib  (pip install numpy matplotlib)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# =============================================================================
# CONFIG  -- edit here or override via --config JSON (same nested keys)
# =============================================================================
CONFIG = {
    "simulation": {
        "n_paths": 600_000,
        "horizon_years": 3,
        "seed": 11,
    },
    # Shared Markov chain: long-run share of adverse state and its persistence
    "regime": {
        "adverse_share": 0.12 / 0.87,     # stationary probability of adverse state
        "adverse_stay": 0.50,             # P(adverse -> adverse); 0.25 = base, 0.50 = sticky
    },
    # Existing book: normal returns per state (% p.a.)
    "book": {
        "calm":    {"mu": 5.0,   "sigma": 15.0},
        "adverse": {"mu": -32.5, "sigma": 8.0},   # = 5% - 2.5 x 15%
    },
    # Candidate A: idiosyncratic Student-t, rescaled to target mean / sd
    "candidate_A": {
        "label": "A: idiosyncratic, fat tail",
        "df": 3,
        "mu": 5.0,
        "sigma": 15.0,
    },
    # Candidate B': loading on the shared state (positive = gains in adverse state)
    "candidate_B": {
        "label": "B': benefits in shared state",
        "df": 4,                  # fat own noise
        "state_loading": 1.16,    # shift (in noise sd) between adverse and calm state
        "noise_adverse": 0.8,     # relative noise sd in adverse state
        "mu": 4.0,
        "sigma": 18.0,
    },
    "decision": {
        "weight": 0.25,           # candidate size relative to the book
        "quantile": 0.995,        # one-year loss quantile
        "loss_threshold": -35.0,  # multi-year cumulative loss threshold in %
        "limit_prob": 0.10,       # admissibility limit for P(cum loss < threshold)
    },
    # Kelly sizing: overlay of w x candidate on top of the (fixed) book,
    # funded at rf; maximise expected log growth of the total
    "kelly": {
        "w_max": 3.0,
        "w_step": 0.05,
        "rf": 0.0,                # funding rate of the overlay in % p.a.
        "loss_floor": -99.9,      # total annual return floored here (limited liability)
        "figure": "kelly_sizing.png",
    },
    # Branding -- replace with your exact brand values
    "branding": {
        "font_family": "Carlito",            # falls back to DejaVu Sans if missing
        "font_file": None,                   # optional path to a .ttf to register
        "primary": "#3B3B98",                # indigo
        "accent": "#1ABC9C",                 # turquoise
        "neutral": "#B8BCC8",
        "text": "#1F2333",
        "muted": "#6B7080",
        "background": "#FFFFFF",
        "title": "Same exposures, different decisions",
        "subtitle": "Which candidate to add depends on the view: marginal vs aggregate, with or without a shared state, one year vs several",
        "footer": "Illustrative, hand-set parameters. The Augmented Actuary.",
        "dpi": 200,
        "figsize": [9.0, 9.0],
    },
    "output": {
        "figure": "decision_matrix.png",
        "csv": "decision_matrix_results.csv",
    },
}


# =============================================================================
# Helpers
# =============================================================================
def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def standardise(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (x - x.mean()) / x.std() * sigma + mu


def unit_t(rng, df, size):
    """Student-t draws rescaled to unit variance (df > 2)."""
    return rng.standard_t(df, size) * np.sqrt((df - 2) / df)


# =============================================================================
# Simulation
# =============================================================================
def simulate(cfg: dict) -> dict:
    sim, reg = cfg["simulation"], cfg["regime"]
    rng = np.random.default_rng(sim["seed"])
    n, h = sim["n_paths"], sim["horizon_years"]

    pi = reg["adverse_share"]
    p_aa = reg["adverse_stay"]
    p_ca = pi * (1 - p_aa) / (1 - pi)       # keeps stationary share fixed

    states = np.empty((n, h), dtype=int)
    s = (rng.random(n) < pi).astype(int)
    for t in range(h):
        states[:, t] = s
        u = rng.random(n)
        s = np.where(s == 0, (u < p_ca).astype(int), (u < p_aa).astype(int))

    b = cfg["book"]
    book = np.where(
        states == 1,
        rng.normal(b["adverse"]["mu"], b["adverse"]["sigma"], (n, h)),
        rng.normal(b["calm"]["mu"], b["calm"]["sigma"], (n, h)),
    )

    ca = cfg["candidate_A"]
    cand_a = standardise(unit_t(rng, ca["df"], (n, h)), ca["mu"], ca["sigma"])

    cb = cfg["candidate_B"]
    z = unit_t(rng, cb["df"], (n, h))
    shift = cb["state_loading"]
    raw_b = np.where(states == 1, shift + cb["noise_adverse"] * z, z)
    cand_b = standardise(raw_b, cb["mu"], cb["sigma"])

    # No-state benchmark: multivariate normal with the same means, vols and
    # correlation as the regime model (year 1), drawn iid across years.
    stack = np.stack([book[:, 0], cand_a[:, 0], cand_b[:, 0]])
    mean, cov = stack.mean(axis=1), np.cov(stack)
    ns = rng.multivariate_normal(mean, cov, size=(n, h))      # shape (n, h, 3)

    return {"states": states, "book": book, "A": cand_a, "B": cand_b,
            "ns_book": ns[..., 0], "ns_A": ns[..., 1], "ns_B": ns[..., 2],
            "p_ca": p_ca, "p_aa": p_aa}


def cumulative(r: np.ndarray) -> np.ndarray:
    return 100 * (np.prod(1 + r / 100, axis=1) - 1)


def evaluate(cfg: dict, sim: dict) -> dict:
    d = cfg["decision"]
    q = 100 * (1 - d["quantile"])
    w = d["weight"]
    out = {}
    for key in ("A", "B"):
        x = sim[key]
        port = (sim["book"] + w * x) / (1 + w)
        port_ns = (sim["ns_book"] + w * sim["ns_" + key]) / (1 + w)
        out[key] = {
            "nostate_1y_loss": -np.percentile(port_ns[:, 0], q),
            "nostate_Hy_prob": np.mean(cumulative(port_ns) < d["loss_threshold"]),
            "marginal_1y_loss": -np.percentile(x[:, 0], q),
            "marginal_Hy_prob": np.mean(cumulative(x) < d["loss_threshold"]),
            "aggregate_1y_loss": -np.percentile(port[:, 0], q),
            "aggregate_Hy_prob": np.mean(cumulative(port) < d["loss_threshold"]),
            "mean": x.mean(),
            "sd": x[:, 0].std(),
            "corr_book": np.corrcoef(x[:, 0], sim["book"][:, 0])[0, 1],
        }
    return out


# =============================================================================
# Figure
# =============================================================================
def setup_font(br: dict) -> None:
    if br.get("font_file"):
        font_manager.fontManager.addfont(br["font_file"])
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = br["font_family"] if br["font_family"] in names else "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "text.color": br["text"],
                         "axes.labelcolor": br["text"], "xtick.color": br["muted"]})


def draw(cfg: dict, res: dict, path: str) -> None:
    br, d, h = cfg["branding"], cfg["decision"], cfg["simulation"]["horizon_years"]
    setup_font(br)
    la, lb = cfg["candidate_A"]["label"], cfg["candidate_B"]["label"]
    qlab = f"{d['quantile']*100:.1f}%"
    thr = f"{abs(d['loss_threshold']):.0f}%"

    cells = [
        # (row, col, metric key, title, value format, lower_is_better, use_limit)
        (0, 0, "marginal_1y_loss",  f"1-year loss at {qlab}, standalone", "{:.1f}%", False),
        (0, 1, "marginal_Hy_prob",  f"P({h}-year loss > {thr}), standalone", "{:.1%}", True),
        (1, 0, "nostate_1y_loss",   f"1-year loss at {qlab}, in the book", "{:.1f}%", False),
        (1, 1, "nostate_Hy_prob",   f"P({h}-year loss > {thr}), in the book", "{:.1%}", True),
        (2, 0, "aggregate_1y_loss", f"1-year loss at {qlab}, in the book", "{:.1f}%", False),
        (2, 1, "aggregate_Hy_prob", f"P({h}-year loss > {thr}), in the book", "{:.1%}", True),
    ]

    fig, axs = plt.subplots(3, 2, figsize=br["figsize"], dpi=br["dpi"],
                            facecolor=br["background"])
    # common x-scale per column so rows are directly comparable
    col_max = {}
    for r, c, key, *_ in cells:
        col_max[c] = max(col_max.get(c, 0), res["A"][key], res["B"][key])
    col_max[1] = max(col_max[1], d["limit_prob"])

    for r, c, key, title, fmt, use_limit in cells:
        ax = axs[r, c]
        va, vb = res["A"][key], res["B"][key]
        if use_limit and r >= 1:
            ok = [va <= d["limit_prob"], vb <= d["limit_prob"]]
            if all(ok):
                choice = "both admissible"
            elif any(ok):
                choice = "only " + ("A" if ok[0] else "B'") + " admissible"
            else:
                choice = "neither admissible"
            winner = None if all(ok) or not any(ok) else (0 if ok[0] else 1)
        else:
            winner = 0 if va < vb else 1
            choice = "choose " + ("A" if winner == 0 else "B'")

        vals = [va, vb]
        colors = [br["neutral"], br["neutral"]]
        if winner is not None:
            colors[winner] = br["accent"] if winner == 1 else br["primary"]
        ax.barh([1, 0], vals, color=colors, height=0.55)
        for yy, v in zip([1, 0], vals):
            ax.text(v, yy, "  " + fmt.format(v), va="center", fontsize=9, color=br["text"])
        if use_limit and r >= 1:
            ax.axvline(d["limit_prob"], color=br["muted"], lw=1, ls="--")
            ax.text(d["limit_prob"], 1.45, f"limit {d['limit_prob']:.0%}", ha="center",
                    fontsize=7.5, color=br["muted"])
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["A", "B'"], fontsize=10)
        ax.set_xticks([])
        ax.set_xlim(0, col_max[c] * 1.3)
        ax.set_ylim(-0.6, 1.6)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(br["muted"])
        ax.set_title(title, fontsize=9.5, color=br["text"], loc="left")
        ax.text(0.98, 0.04, choice, transform=ax.transAxes, ha="right", fontsize=10,
                fontweight="bold", color=br["primary"])

    # axis labels for the two dimensions
    fig.text(0.29, 0.905, "One year", ha="center", fontsize=11, fontweight="bold", color=br["primary"])
    fig.text(0.74, 0.905, f"{h} years (time axis)", ha="center", fontsize=11, fontweight="bold", color=br["primary"])
    for yy, lab in ((0.735, "Marginal"), (0.45, "Aggregate,\nno state"), (0.165, "Aggregate,\nshared state")):
        fig.text(0.02, yy, lab, rotation=90, va="center", ha="center", fontsize=10.5,
                 fontweight="bold", color=br["primary"])

    fig.suptitle(br["title"], x=0.05, ha="left", y=0.985, fontsize=14, fontweight="bold", color=br["text"])
    fig.text(0.05, 0.945, br["subtitle"], fontsize=9, color=br["muted"])
    reg = cfg["regime"]
    foot = (f"{la} vs {lb}; candidate weight {d['weight']:.0%} of book; adverse share "
            f"{reg['adverse_share']:.0%}, P(stay adverse) {reg['adverse_stay']:.0%}. {br['footer']}")
    fig.text(0.05, 0.015, foot, fontsize=6.8, color=br["muted"])
    plt.tight_layout(rect=(0.05, 0.03, 1, 0.9))
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


# =============================================================================
# Kelly
# =============================================================================
def log_growth(book: np.ndarray, x: np.ndarray, w: float, k: dict, mask=None) -> float:
    r = book + w * (x - k["rf"])
    if mask is not None:
        r = r[mask]
    return float(np.mean(np.log1p(np.maximum(r, k["loss_floor"]) / 100)))


def kelly(cfg: dict, sim: dict) -> dict:
    k = cfg["kelly"]
    grid = np.round(np.arange(0, k["w_max"] + 1e-9, k["w_step"]), 4)
    prev = sim["states"][:, 0]          # state observed today
    out = {"grid": grid}
    for key in ("A", "B"):
        # year-2 returns: unconditional, and conditional on today's state
        curves = {
            "state": [log_growth(sim["book"][:, 1], sim[key][:, 1], w, k) for w in grid],
            "nostate": [log_growth(sim["ns_book"][:, 1], sim["ns_" + key][:, 1], w, k) for w in grid],
            "calm": [log_growth(sim["book"][:, 1], sim[key][:, 1], w, k, prev == 0) for w in grid],
            "adverse": [log_growth(sim["book"][:, 1], sim[key][:, 1], w, k, prev == 1) for w in grid],
        }
        out[key] = {name: {"curve": np.array(c), "w_opt": float(grid[int(np.argmax(c))]),
                           "g_opt": float(np.max(c))} for name, c in curves.items()}
        # what the no-state optimum really earns under the state model
        w_ns = out[key]["nostate"]["w_opt"]
        out[key]["g_at_nostate_w"] = float(np.array(curves["state"])[int(np.argmax(curves["nostate"]))])
        out[key]["w_nostate"] = w_ns
    return out


def draw_kelly(cfg: dict, kr: dict, path: str) -> None:
    br = cfg["branding"]
    setup_font(br)
    grid = kr["grid"]
    fig, axs = plt.subplots(1, 2, figsize=(br["figsize"][0], 4.6), dpi=br["dpi"],
                            facecolor=br["background"], gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    for key, col, lab in (("A", br["primary"], "A"), ("B", br["accent"], "B'")):
        for model, ls, mlab in (("state", "-", "shared state"), ("nostate", "--", "no state")):
            c = kr[key][model]["curve"] * 100
            ax.plot(grid, c, color=col, ls=ls, lw=2 if ls == "-" else 1.4, label=f"{lab}, {mlab}")
            wo = kr[key][model]["w_opt"]
            ax.plot(wo, kr[key][model]["g_opt"] * 100, "o", color=col, ms=5,
                    mfc=col if ls == "-" else br["background"])
    ax.axhline(0, color=br["muted"], lw=0.6)
    ax.set_xlabel("Overlay size w (x book)")
    ax.set_ylabel("Expected log growth, % p.a.")
    ax.set_title("Kelly curves: the no-state model overstates achievable growth", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = axs[1]
    labels, vals, cols = [], [], []
    for key, col, lab in (("A", br["primary"], "A"), ("B", br["accent"], "B'")):
        for st in ("calm", "adverse"):
            labels.append(f"{lab}\n{st}")
            vals.append(kr[key][st]["w_opt"])
            cols.append(col if st == "adverse" else br["neutral"])
    ax.bar(range(4), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks([])
    ax.set_title("Kelly size given today's state", fontsize=10, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    reg = cfg["regime"]
    fig.text(0.01, 0.01, f"Overlay funded at {cfg['kelly']['rf']}%; total return floored at "
             f"{cfg['kelly']['loss_floor']}%. P(stay adverse) {reg['adverse_stay']:.0%}. {br['footer']}",
             fontsize=6.8, color=br["muted"])
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


def write_csv(res: dict, path: str) -> None:
    keys = list(res["A"].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "A", "B_prime"])
        for k in keys:
            w.writerow([k, f"{res['A'][k]:.6f}", f"{res['B'][k]:.6f}"])


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", help="JSON file with overrides (nested keys as in CONFIG)")
    ap.add_argument("--dump-config", action="store_true", help="print the effective config and exit")
    args = ap.parse_args()

    cfg = copy.deepcopy(CONFIG)
    if args.config:
        deep_update(cfg, json.loads(Path(args.config).read_text()))
    if args.dump_config:
        print(json.dumps(cfg, indent=2))
        return

    sim = simulate(cfg)
    res = evaluate(cfg, sim)
    write_csv(res, cfg["output"]["csv"])
    draw(cfg, res, cfg["output"]["figure"])
    kr = kelly(cfg, sim)
    draw_kelly(cfg, kr, cfg["kelly"]["figure"])

    print(f"P(calm->adverse)={sim['p_ca']:.3f}  P(adverse->adverse)={sim['p_aa']:.3f}")
    for k, lab in (("A", "A"), ("B", "B'")):
        r = res[k]
        print(f"{lab:2s} mean {r['mean']:5.1f}  sd {r['sd']:5.1f}  corr(book) {r['corr_book']:5.2f}  "
              f"| marg 1y {r['marginal_1y_loss']:5.1f}  marg Hy {r['marginal_Hy_prob']:6.2%}  "
              f"| no-state 1y {r['nostate_1y_loss']:5.1f}  no-state Hy {r['nostate_Hy_prob']:6.2%}  "
              f"| agg 1y {r['aggregate_1y_loss']:5.1f}  agg Hy {r['aggregate_Hy_prob']:6.2%}")
    print("Kelly (overlay size w, log growth % p.a.):")
    for k, lab in (("A", "A"), ("B", "B'")):
        r = kr[k]
        print(f"  {lab:2s} state w*={r['state']['w_opt']:.2f} g*={100*r['state']['g_opt']:.2f}% | "
              f"no-state w*={r['nostate']['w_opt']:.2f} g*={100*r['nostate']['g_opt']:.2f}% "
              f"(earns {100*r['g_at_nostate_w']:.2f}% under state) | "
              f"given calm w*={r['calm']['w_opt']:.2f}, given adverse w*={r['adverse']['w_opt']:.2f}")
    print(f"  book alone g={100*kr['A']['state']['curve'][0]:.2f}%")
    print(f"wrote {cfg['output']['figure']}, {cfg['kelly']['figure']} and {cfg['output']['csv']}")


if __name__ == "__main__":
    main()
