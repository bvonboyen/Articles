#!/usr/bin/env python3
"""
Pension fund decision study: ILS fund vs longevity-benefiting equity ETF.

A pension fund with a given strategic allocation adds a share s of its assets
(taken from bonds) to one of two candidates:

  ILS : insurance-linked securities fund. Carry, fat left tail from cat events,
        independent of markets and longevity (optional collateral link).
  ETF : equity ETF benefiting from longevity (health care, services, ageing).
        Loads on the market regime (falls in crashes) AND on the slow
        longevity drift (gains when people live longer).

Target metric: funding ratio FR = assets / liabilities, with annual pension
payments paid out of assets (path dependence when underfunded).

Five switches (valuation first), each can be toggled in CONFIG["switches"]:
  V valuation : statutory (fixed technical rate) vs market-consistent funding ratio
  0 criterion : constraint (P(FR < floor within horizon) <= limit) vs Kelly
                (maximise expected log growth of FR)
  1 horizon   : one year vs multi-year
  2 tails     : normal (jointly Gaussian, iid years)
                vs univariate heavy tails (fat-tailed marginals, Gaussian copula, iid years)
                vs multivariate (shared fast market regime + slow longevity state)
  3 scope     : partial (market/credit only, deterministic longevity)
                vs total (stochastic longevity on the liability side)

The script runs a "decision ladder" from the conventional set-up
(partial, univariate, one year) to the full set-up (total, shared state,
multi-year), writes a CSV with all 8 combinations, and draws one figure.

All parameters are illustrative and hand-set.
Usage:  python pension_study.py [--config overrides.json] [--dump-config]
Requires numpy, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "simulation": {"n_paths": 200_000, "horizon_years": 10, "seed": 42},   # observation period: 1 year or this

    # --- fast market regime (shared Markov chain) ---------------------------
    "market_regime": {
        "adverse_share": 0.14,     # stationary probability of the crash state
        "adverse_stay": 0.50,      # persistence: P(crash -> crash)
    },
    "equity": {"calm": {"mu": 8.0, "sigma": 14.0}, "adverse": {"mu": -25.0, "sigma": 10.0}},
    # change in discount rate, %-points per year (falls in crashes: flight to quality)
    "rates": {"calm": {"mu": 0.05, "sigma": 0.35}, "adverse": {"mu": -0.60, "sigma": 0.35},
              "bond_yield": 2.0, "bond_duration": 13.0,
              "neutral_drift": True,
              # second-order price effect: +0.5 * C * dr^2, with C = factor * duration^2
              # (bonds and liabilities); swap overlay convexity = factor * D_liab per unit duration
              "convexity_factor": 1.0},   # set calm mean so that average rate change is zero

    # --- liabilities ---------------------------------------------------------
    "liabilities": {
        "technical_rate": 2.0,     # accrual in % p.a.
        "duration": 14.0,
        "benefit_rate": 4.0,       # pensions paid per year, % of liabilities
    },
    # slow longevity state: persistent drift in liability value (% p.a.)
    "longevity": {"phi": 0.95, "drift_sd": 0.50, "noise_sd": 0.30},

    # --- strategic allocation (before the candidate) ------------------------
    "allocation": {"equity": 0.30, "bonds": 0.65, "cash": 0.05, "cash_rate": 1.0,
                   "funding_ratio_0": 1.18,
                   "funded_from": "equity"},   # candidate replaces "equity" or "bonds"

    # --- interest-rate hedge (third lever) -----------------------------------
    # ratio = target funding-ratio hedge ratio (asset duration / liability duration).
    # None = physical bonds only. The gap to the physical bond duration is closed
    # (or reduced) with a swap overlay that costs `overlay_cost_bp` per unit of
    # overlay duration per year (in bp of assets).
    "hedge": {"ratio": None, "overlay_cost_bp": 1.0,
              "grid": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},

    # --- candidates ----------------------------------------------------------
    "ils": {"label": "ILS fund", "carry": 7.0, "noise_sd": 2.0,
            "event_prob": 0.05, "loss_min": 25.0, "loss_max": 70.0,
            "collateral_loss_in_crash": 0.0},   # e.g. 5.0 to mimic 2008 collateral hits
    "etf": {"label": "Longevity ETF", "mu": 6.0, "equity_beta": 0.6,
            "longevity_loading": 8.0,    # % ETF return per 1%-pt longevity drift
            "idio_sd": 6.0},

    # --- decision ------------------------------------------------------------
    "decision": {
        "share": 0.15,                 # candidate share of assets for the comparison
        "share_grid_max": 0.30, "share_grid_step": 0.02,
        "fr_floor": 1.00,              # underfunding threshold
        "limit_prob": 0.15,            # max P(FR < floor at any year-end within horizon)
        "tail_df": 4,                  # marginal t df for the univariate model
        "split_step": 0.05,            # grid for the ILS fraction of the sleeve
    },

    # --- overarching criterion (agreed 21 Nov 2026) -------------------------
    # maximise expected annual log growth of the MARKET-CONSISTENT funding ratio
    # over the observation period (net of hedging costs), subject to
    #   P(market-consistent FR < floor at any year-end) <= limit_market
    #   P(statutory FR < floor at any year-end)         <= limit_statutory
    "criterion": {"limit_market": 0.15, "limit_statutory": 0.10,
                  "joint_split_step": 0.10,
                  "joint_hedge_grid": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},

    # --- switches: the "full" set-up; the ladder walks from conventional to full
    "switches": {"criterion": "both", "valuation": "market", "horizon": "multi",
                 "tails": "multivariate", "scope": "total"},
    # valuation: "statutory" = fixed technical rate (liabilities insensitive to rates,
    #            mortality tables not updated within the horizon -> no longevity shocks)
    #            "market"    = market-consistent (liabilities move with rates and longevity)

    "branding": {
        "font_family": "Carlito", "font_file": None,
        "primary": "#3B3B98", "accent": "#1ABC9C", "neutral": "#B8BCC8",
        "text": "#1F2333", "muted": "#6B7080", "background": "#FFFFFF",
        "title": "Same pension fund, same two candidates, different decisions",
        "subtitle": "Each modelling step measured in both funding ratios: risk of falling below 100% funding",
        "footer": "Illustrative, hand-set parameters. The Augmented Actuary.",
        "dpi": 200, "figsize": [9.0, 7.4],
    },
    "output": {"figure": "pension_decision_ladder.png", "csv": "pension_study_results.csv",
               "split_figure": "pension_split.png", "split_csv": "pension_split_results.csv",
               "hedge_figure": "pension_hedge.png", "hedge_csv": "pension_hedge_results.csv",
               "joint_figure": "pension_optimum_ladder.png", "joint_map": "pension_optimum_map.png",
               "joint_csv": "pension_optimum_results.csv", "when_figure": "pension_when.png"},
}


# =============================================================================
def deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


# =============================================================================
# Scenario generation
# =============================================================================
def simulate_state_world(cfg, rng, n, h, start_state=None):
    """Multivariate world: shared fast market regime + slow longevity drift."""
    mr = cfg["market_regime"]
    pi, p_aa = mr["adverse_share"], mr["adverse_stay"]
    p_ca = pi * (1 - p_aa) / (1 - pi)
    if start_state is None:
        s = (rng.random(n) < pi).astype(int)
    else:
        # today's (observed) state; year 1 follows from the transition probabilities
        s0 = np.full(n, int(start_state))
        u = rng.random(n)
        s = np.where(s0 == 0, (u < p_ca).astype(int), (u < p_aa).astype(int))
    states = np.empty((n, h), int)
    for t in range(h):
        states[:, t] = s
        u = rng.random(n)
        s = np.where(s == 0, (u < p_ca).astype(int), (u < p_aa).astype(int))

    eq_c, eq_a = cfg["equity"]["calm"], cfg["equity"]["adverse"]
    equity = np.where(states == 1, rng.normal(eq_a["mu"], eq_a["sigma"], (n, h)),
                      rng.normal(eq_c["mu"], eq_c["sigma"], (n, h)))
    rc, ra = dict(cfg["rates"]["calm"]), cfg["rates"]["adverse"]
    if cfg["rates"].get("neutral_drift"):
        rc["mu"] = -pi * ra["mu"] / (1 - pi)
    d_rate = np.where(states == 1, rng.normal(ra["mu"], ra["sigma"], (n, h)),
                      rng.normal(rc["mu"], rc["sigma"], (n, h)))

    lg = cfg["longevity"]
    stat_sd = lg["drift_sd"] / np.sqrt(1 - lg["phi"] ** 2)
    drift = np.empty((n, h))
    d = rng.normal(0, stat_sd, n)
    for t in range(h):
        d = lg["phi"] * d + rng.normal(0, lg["drift_sd"], n) if t > 0 else d
        drift[:, t] = d
    longevity = drift + rng.normal(0, lg["noise_sd"], (n, h))

    ils = ils_returns(cfg, rng, (n, h), states)
    etf = etf_returns(cfg, rng, equity, drift)
    return {"equity": equity, "d_rate": d_rate, "longevity": longevity, "ils": ils, "etf": etf}


def ils_returns(cfg, rng, shape, states=None):
    c = cfg["ils"]
    hit = rng.random(shape) < c["event_prob"]
    loss = np.where(hit, rng.uniform(c["loss_min"], c["loss_max"], shape), 0.0)
    r = c["carry"] + rng.normal(0, c["noise_sd"], shape) - loss
    if states is not None and c["collateral_loss_in_crash"]:
        r = r - c["collateral_loss_in_crash"] * (states == 1)
    return r


def etf_returns(cfg, rng, equity, drift):
    e = cfg["etf"]
    eq_mean = equity.mean()
    return (e["mu"] + e["equity_beta"] * (equity - eq_mean)
            + e["longevity_loading"] * drift + rng.normal(0, e["idio_sd"], equity.shape))


def simulate_normal_world(cfg, rng, n, h, ref):
    """Normal world: all risks (incl. ILS) jointly normal with the same means, vols and
    correlations as `ref`, iid across years. No fat tails, no state, no persistence."""
    keys = ["equity", "d_rate", "longevity", "etf", "ils"]
    X = np.stack([ref[k][:, 0] for k in keys], axis=1)
    z = rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), size=(n, h))
    return {k: z[..., i] for i, k in enumerate(keys)}


def simulate_univariate_world(cfg, rng, n, h, ref):
    """Univariate world: same means, vols and correlations as `ref`, but no state:
    iid years, Gaussian copula, fat-tailed (t) marginals rescaled to match moments.
    ILS keeps its own (already fat) marginal by resampling."""
    keys = ["equity", "d_rate", "longevity", "etf"]
    X = np.stack([ref[k][:, 0] for k in keys], axis=1)
    mu, sd = X.mean(0), X.std(0)
    corr = np.corrcoef(X, rowvar=False)
    z = rng.multivariate_normal(np.zeros(len(keys)), corr, size=(n, h))
    df = cfg["decision"]["tail_df"]
    t = stats.t.ppf(stats.norm.cdf(z), df) * np.sqrt((df - 2) / df)
    out = {k: mu[i] + sd[i] * t[..., i] for i, k in enumerate(keys)}
    out["ils"] = ils_returns(cfg, rng, (n, h))
    return out


# =============================================================================
# Balance-sheet projection
# =============================================================================
def project(cfg, world, cand, share, scope, h, valuation="market"):
    """Funding-ratio paths. Returns FR at each year-end, shape (n, h)."""
    a, l = cfg["allocation"], cfg["liabilities"]
    rt = cfg["rates"]
    cf = rt.get("convexity_factor", 0.0)
    dr2 = world["d_rate"] ** 2 / 100          # (dr in %-pts)^2 -> % return units
    bond = rt["bond_yield"] - rt["bond_duration"] * world["d_rate"] + 0.5 * cf * rt["bond_duration"] ** 2 * dr2
    if a.get("funded_from", "bonds") == "equity":
        w_eq, w_bd = a["equity"] - share, a["bonds"]
    else:
        w_eq, w_bd = a["equity"], a["bonds"] - share
    w_cs = a["cash"]
    cand_r = world[cand] if share > 0 else 0.0
    r_assets = (w_eq * world["equity"] + w_bd * bond + w_cs * a["cash_rate"]
                + share * cand_r) / 100
    hg = cfg.get("hedge", {})
    if hg.get("ratio") is not None:
        d_overlay = hg["ratio"] * l["duration"] - w_bd * rt["bond_duration"]
        r_assets = r_assets + (-d_overlay * world["d_rate"]
                               + 0.5 * cf * l["duration"] * d_overlay * dr2
                               - abs(d_overlay) * hg["overlay_cost_bp"] / 100) / 100
    if valuation == "statutory":
        r_liab = np.full_like(world["d_rate"], l["technical_rate"])
    else:
        r_liab = l["technical_rate"] - l["duration"] * world["d_rate"] + 0.5 * cf * l["duration"] ** 2 * dr2
    if scope == "total" and valuation == "market":
        r_liab = r_liab + world["longevity"]
    r_liab = np.maximum(r_liab / 100, -0.95)
    r_assets = np.maximum(r_assets, -0.95)

    n = r_assets.shape[0]
    A = np.full(n, a["funding_ratio_0"])
    L = np.ones(n)
    b = l["benefit_rate"] / 100
    fr = np.empty((n, h))
    for t in range(h):
        A = A * (1 + r_assets[:, t])
        L = L * (1 + r_liab[:, t])
        pay = b * L
        A, L = A - pay, L - pay
        A = np.maximum(A, 1e-6)
        fr[:, t] = A / L
    return fr


def limit_for(cfg, valuation):
    cr = cfg.get("criterion")
    if cr:
        return cr["limit_market"] if valuation == "market" else cr["limit_statutory"]
    return cfg["decision"]["limit_prob"]


def metrics(cfg, fr, h):
    d = cfg["decision"]
    fr_h = fr[:, :h]
    breach = np.mean((fr_h < d["fr_floor"]).any(axis=1))
    growth = np.mean(np.log(fr_h[:, -1] / cfg["allocation"]["funding_ratio_0"])) / h
    return breach, growth


def hedge_ratios(cfg, share):
    a, l, rt = cfg["allocation"], cfg["liabilities"], cfg["rates"]
    w_bd = a["bonds"] if a.get("funded_from", "bonds") == "equity" else a["bonds"] - share
    d_assets = w_bd * rt["bond_duration"]
    if cfg.get("hedge", {}).get("ratio") is not None:
        d_assets = cfg["hedge"]["ratio"] * l["duration"]
    return {"asset_duration": d_assets, "liability_duration": l["duration"],
            "fr_hedge": d_assets / l["duration"],
            "surplus_hedge": a["funding_ratio_0"] * d_assets / l["duration"],
            "fr_sensitivity_per_100bp": -(d_assets - l["duration"])}


# =============================================================================
# One configuration
# =============================================================================
def evaluate(cfg, worlds, tails, scope, horizon, valuation="market"):
    d = cfg["decision"]
    h = 1 if horizon == "one" else cfg["simulation"]["horizon_years"]
    world = worlds[tails]
    res = {}
    grid = np.round(np.arange(0, d["share_grid_max"] + 1e-9, d["share_grid_step"]), 4)
    for cand in ("ils", "etf"):
        fr = project(cfg, world, cand, d["share"], scope, h, valuation)
        br, g = metrics(cfg, fr, h)
        curve = []
        for s in grid:
            b_s, g_s = metrics(cfg, project(cfg, world, cand, s, scope, h, valuation), h)
            curve.append((s, b_s, g_s))
        kelly = max(curve, key=lambda x: x[2])
        adm = [c for c in curve if c[1] <= limit_for(cfg, valuation)]
        kelly_adm = max(adm, key=lambda x: x[2]) if adm else None
        res[cand] = {"breach": br, "growth": g, "admissible": br <= limit_for(cfg, valuation),
                     "kelly_share": kelly[0], "kelly_growth": kelly[2],
                     "kelly_adm_share": kelly_adm[0] if kelly_adm else None,
                     "kelly_adm_growth": kelly_adm[2] if kelly_adm else None}
    # decisions
    i, e = res["ils"], res["etf"]
    if i["admissible"] and e["admissible"]:
        res["constraint_choice"] = "both"
    elif i["admissible"] or e["admissible"]:
        res["constraint_choice"] = "ILS" if i["admissible"] else "ETF"
    else:
        res["constraint_choice"] = "neither"
    res["kelly_choice"] = "ILS" if i["kelly_growth"] >= e["kelly_growth"] else "ETF"
    adm = [(k, res[k]["growth"]) for k in ("ils", "etf") if res[k]["admissible"]]
    res["constrained_kelly"] = (max(adm, key=lambda x: x[1])[0].upper() + f" {d['share']:.0%}") if adm else "none"
    return res


# =============================================================================
# Split study: fixed sleeve, optimal ILS / ETF mix
# =============================================================================
def evaluate_split(cfg, worlds, tails, scope, horizon, valuation):
    d = cfg["decision"]
    h = 1 if horizon == "one" else cfg["simulation"]["horizon_years"]
    w = dict(worlds[tails])
    fracs = np.round(np.arange(0, 1 + 1e-9, d["split_step"]), 4)   # ILS fraction of the sleeve
    rows = []
    for f in fracs:
        w["mix"] = f * w["ils"] + (1 - f) * w["etf"]
        b, g = metrics(cfg, project(cfg, w, "mix", d["share"], scope, h, valuation), h)
        rows.append((f, b, g))
    rows = np.array(rows)
    adm = rows[rows[:, 1] <= limit_for(cfg, valuation)]
    return {"fracs": rows[:, 0], "breach": rows[:, 1], "growth": rows[:, 2],
            "min_risk": float(rows[np.argmin(rows[:, 1]), 0]),
            "growth_opt": float(adm[np.argmax(adm[:, 2]), 0]) if len(adm) else None,
            "adm_range": (float(adm[:, 0].min()), float(adm[:, 0].max())) if len(adm) else None}


def draw_split(cfg, split, labels, path):
    br, d = cfg["branding"], cfg["decision"]
    setup_font(br)
    fig, axs = plt.subplots(1, 2, figsize=(br["figsize"][0], 5.0), dpi=br["dpi"],
                            facecolor=br["background"], sharey=True)
    shades = ["#D5D8E0", br["neutral"], "#8A90A6", br["primary"], br["accent"]][-len(labels):]
    ymax = max(split[v][i]["breach"].max() for v in split for i in range(len(labels)))
    ymax = max(ymax, limit_for(cfg, "market")) * 1.15
    for ax, val, title in ((axs[0], "statutory", "Statutory funding ratio"),
                           (axs[1], "market", "Market-consistent funding ratio")):
        for i, (lab, col) in enumerate(zip(labels, shades)):
            r = split[val][i]
            x = r["fracs"] * 100
            ax.plot(x, r["breach"] * 100, color=col, lw=2.2 if i >= len(labels) - 2 else 1.6,
                    label=lab.replace("\n", " "))
            k = int(np.argmin(r["breach"]))
            ax.plot(x[k], r["breach"][k] * 100, "o", color=col, ms=5)
        lim = limit_for(cfg, val)
        ax.axhline(lim * 100, color=br["muted"], ls="--", lw=1)
        ax.text(1, lim * 100 + 0.3, f"limit {lim:.0%}", fontsize=7.5, color=br["muted"])
        ax.set_ylim(0, ymax * 100)
        ax.set_xlabel(f"ILS share of the {d['share']:.0%} sleeve (rest: longevity ETF), %")
        ax.set_title(title, fontsize=10, loc="left", color=br["text"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel(f"P(funding ratio < {d['fr_floor']:.0%}), %")
    h_, l_ = axs[1].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.015, 0.93), ncol=3)
    fig.suptitle(f"Reallocating {d['share']:.0%}: how to split it between ILS and the longevity ETF",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=br["text"])
    fig.text(0.02, 0.012, f"Dots: risk-minimising split. Statutory: the longevity line coincides with the shared-state line. {br['footer']}", fontsize=6.8, color=br["muted"])
    plt.tight_layout(rect=(0, 0.03, 1, 0.86))
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


# =============================================================================
# Hedge-ratio study (full model): risk-minimising split for each hedge ratio
# =============================================================================
def evaluate_hedge(cfg, worlds, valuation, key=("multivariate", "total", "multi")):
    out = []
    for hr in cfg["hedge"]["grid"]:
        c = copy.deepcopy(cfg)
        c["hedge"]["ratio"] = hr
        r = evaluate_split(c, worlds, *key, valuation)
        k = int(np.argmin(r["breach"]))
        out.append({"ratio": hr, "breach": float(r["breach"][k]), "ils_fraction": float(r["fracs"][k]),
                    "growth": float(r["growth"][k]),
                    "breach_all_ils": float(r["breach"][-1]), "breach_all_etf": float(r["breach"][0])})
    return out


def draw_hedge(cfg, hedge, path):
    br, d = cfg["branding"], cfg["decision"]
    setup_font(br)
    fig, axs = plt.subplots(1, 2, figsize=(br["figsize"][0], 4.6), dpi=br["dpi"],
                            facecolor=br["background"], gridspec_kw={"width_ratios": [1.4, 1]})
    cols = {"statutory": br["neutral"], "market": br["primary"]}
    labs = {"statutory": "Statutory funding ratio", "market": "Market-consistent funding ratio"}
    ax = axs[0]
    for v in ("statutory", "market"):
        x = [r["ratio"] * 100 for r in hedge[v]]
        y = [r["breach"] * 100 for r in hedge[v]]
        ax.plot(x, y, "-o", color=cols[v], lw=2.2, ms=4, label=labs[v])
        k = int(np.argmin(y))
        ax.plot(x[k], y[k], "o", ms=9, mfc="none", mec=cols[v], mew=1.5)
    cur = hedge_ratios({**cfg, "hedge": {**cfg["hedge"], "ratio": None}}, d["share"])["fr_hedge"] * 100
    ax.axvline(cur, color=br["muted"], ls=":", lw=1)
    ax.text(cur, ax.get_ylim()[1] * 0.88, f" today {cur:.0f}%", fontsize=7.5, color=br["muted"], va="top")
    ax.axhline(limit_for(cfg, "market") * 100, color=br["primary"], ls="--", lw=1)
    ax.axhline(limit_for(cfg, "statutory") * 100, color=br["muted"], ls=":", lw=1)
    ax.set_xlabel("Interest-rate hedge ratio (funding-ratio terms), %")
    ax.set_ylabel(f"P(funding ratio < {d['fr_floor']:.0%}), %")
    ax.set_title("Risk at the safest ILS/ETF split", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = axs[1]
    for v in ("statutory", "market"):
        x = [r["ratio"] * 100 for r in hedge[v]]
        y = [(1 - r["ils_fraction"]) * 100 for r in hedge[v]]
        ax.plot(x, y, "-o", color=cols[v] if v == "statutory" else br["accent"], lw=2.2, ms=4,
                label="Statutory" if v == "statutory" else "Market-consistent")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("Interest-rate hedge ratio, %")
    ax.set_ylabel(f"Longevity ETF share of the {d['share']:.0%} sleeve, %")
    ax.set_title("Hedging rates raises the value\nof the longevity hedge", fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("Third lever: the interest-rate hedge ratio", x=0.02, ha="left", fontsize=12.5,
                 fontweight="bold", color=br["text"])
    fig.text(0.02, 0.012, f"Full model (shared state, longevity, {cfg['simulation']['horizon_years']} years). "
             f"Swap overlay cost {cfg['hedge']['overlay_cost_bp']} bp per unit of duration p.a. "
             f"Circles: lowest risk. Right: 5% grid, small wiggles are noise. {br['footer']}", fontsize=6.8, color=br["muted"])
    plt.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


# =============================================================================
# Joint optimisation over split x hedge ratio with the overarching criterion
# =============================================================================
def evaluate_joint(cfg, worlds, tails, scope, horizon):
    cr, d = cfg["criterion"], cfg["decision"]
    h = 1 if horizon == "one" else cfg["simulation"]["horizon_years"]
    w = dict(worlds[tails])
    fracs = np.round(np.arange(0, 1 + 1e-9, cr["joint_split_step"]), 4)
    hrs = list(cr["joint_hedge_grid"])
    shape = (len(hrs), len(fracs))
    b_mc, b_st, g_mc = np.empty(shape), np.empty(shape), np.empty(shape)
    e_mc, e_st = np.empty(shape), np.empty(shape)     # complementary: underfunded at the final year-end
    for i, hr in enumerate(hrs):
        c = copy.deepcopy(cfg)
        c["hedge"]["ratio"] = hr
        for j, f in enumerate(fracs):
            w["mix"] = f * w["ils"] + (1 - f) * w["etf"]
            fr_m = project(c, w, "mix", d["share"], scope, h, "market")
            fr_s = project(c, w, "mix", d["share"], scope, h, "statutory")
            b_mc[i, j], g_mc[i, j] = metrics(c, fr_m, h)
            b_st[i, j], _ = metrics(c, fr_s, h)
            e_mc[i, j] = np.mean(fr_m[:, h - 1] < d["fr_floor"])
            e_st[i, j] = np.mean(fr_s[:, h - 1] < d["fr_floor"])
    ok = (b_mc <= cr["limit_market"]) & (b_st <= cr["limit_statutory"])
    res = {"hrs": np.array(hrs), "fracs": fracs, "b_mc": b_mc, "b_st": b_st, "g_mc": g_mc, "ok": ok,
           "e_mc": e_mc, "e_st": e_st}
    if ok.any():
        g = np.where(ok, g_mc, -np.inf)
        i, j = np.unravel_index(np.argmax(g), g.shape)
        res["opt"] = {"hedge": hrs[i], "ils": float(fracs[j]), "b_mc": float(b_mc[i, j]),
                      "e_mc": float(e_mc[i, j]), "e_st": float(e_st[i, j]),
                      "b_st": float(b_st[i, j]), "growth": float(g_mc[i, j])}
        # which limit binds: check neighbours with higher growth
        better = g_mc > g_mc[i, j] + 1e-12
        res["binding"] = sorted({lab for lab, mask in (("market", b_mc > cr["limit_market"]),
                                                       ("statutory", b_st > cr["limit_statutory"]))
                                 if (better & mask).any()})
    else:
        res["opt"], res["binding"] = None, []
    # least-risk combination (fallback when nothing is admissible): minimise the larger
    # of the two breach probabilities relative to its limit
    worst = np.maximum(b_mc / cr["limit_market"], b_st / cr["limit_statutory"])
    i, j = np.unravel_index(np.argmin(worst), worst.shape)
    res["least_risk"] = {"hedge": hrs[i], "ils": float(fracs[j]), "b_mc": float(b_mc[i, j]),
                         "e_mc": float(e_mc[i, j]), "e_st": float(e_st[i, j]),
                         "b_st": float(b_st[i, j]), "growth": float(g_mc[i, j])}
    # starting point: no reallocation, today's hedge ratio (physical bonds only)
    c0 = copy.deepcopy(cfg)
    c0["hedge"]["ratio"] = None
    base = {}
    for val in ("market", "statutory"):
        fr0 = project(c0, worlds[tails], "ils", 0.0, scope, h, val)
        b_, g_ = metrics(c0, fr0, h)
        base[val] = (b_, g_, float(np.mean(fr0[:, h - 1] < cfg["decision"]["fr_floor"])))
    res["base"] = {"b_mc": base["market"][0], "b_st": base["statutory"][0], "growth": base["market"][1],
                   "e_mc": base["market"][2], "e_st": base["statutory"][2],
                   "hedge": hedge_ratios(c0, 0.0)["fr_hedge"],
                   "ok": base["market"][0] <= cr["limit_market"] and base["statutory"][0] <= cr["limit_statutory"]}
    return res


def draw_joint_ladder(cfg, steps, joint, path):
    br, cr = cfg["branding"], cfg["criterion"]
    setup_font(br)
    n = len(steps)
    fig, axs = plt.subplots(1, 2, figsize=(br["figsize"][0], 1.5 * n + 1.8), dpi=br["dpi"], facecolor=br["background"],
                            sharey=True, gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(n)[::-1]
    ax = axs[0]
    for off, key, col, lab in ((0.18, "b_mc", br["primary"], "market-consistent"),
                               (-0.18, "b_st", br["neutral"], "statutory")):
        vals = [joint[i]["opt"][key] if joint[i]["opt"] else np.nan for i in range(n)]
        base = [joint[i]["base"][key] for i in range(n)]
        ax.barh(y + off, base, height=0.34, fill=False, edgecolor=col, lw=1.2, ls="--")
        ax.barh(y + off, vals, height=0.34, color=col, label=f"P(FR < 100%), {lab}")
        for yy, v, b0 in zip(y + off, vals, base):
            ax.text(max(v, b0), yy, f" {v:.1%} (start {b0:.1%})", va="center", fontsize=7.5, color=br["text"])
    ax.axvline(cr["limit_market"], color=br["primary"], ls="--", lw=1)
    ax.axvline(cr["limit_statutory"], color=br["muted"], ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in steps], fontsize=9)
    ax.set_xticks([])
    ax.set_title(f"Risk at the optimum; outline = starting point\n(limits: market "
                 f"{cr['limit_market']:.0%} dashed, statutory {cr['limit_statutory']:.0%} dotted)", fontsize=8.5, loc="left")
    h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.015, 0.93), ncol=2)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)

    ax = axs[1]
    ax.axis("off")
    ax.set_title("Optimal levers", fontsize=10, loc="left")
    for yy, r in zip(y, joint):
        o = r["opt"]
        if o is None:
            txt = "no admissible combination"
        else:
            bind = ", ".join(r["binding"]) if r["binding"] else "none"
            b0 = r["base"]
            txt = (f"hedge ratio {o['hedge']:.0%}\n"
                   f"sleeve: {o['ils']:.0%} ILS / {1 - o['ils']:.0%} ETF\n"
                   f"growth {100 * o['growth']:.2f}% p.a. (start {100 * b0['growth']:.2f}%)\n"
                   f"underfunded at final year: {o['e_mc']:.1%} / {o['e_st']:.1%} "
                   f"(start {b0['e_mc']:.1%} / {b0['e_st']:.1%})\n"
                   f"binding: {bind}{'' if b0['ok'] else ' | start breaches a limit'}")
        ax.text(0.02, yy, txt, va="center", fontsize=7.8, color=br["primary"], fontweight="bold",
                transform=ax.get_yaxis_transform(), linespacing=1.25)
    fig.suptitle(f"One criterion, {n} modelling steps: optimal split and hedge ratio", x=0.02, ha="left",
                 fontsize=12.5, fontweight="bold", color=br["text"])
    plt.tight_layout(rect=(0, 0.03, 1, 0.86))
    fig.text(0.02, 0.012, "Criterion: max growth of the market-consistent funding ratio s.t. both underfunding "
             f"limits. Reallocation {cfg['decision']['share']:.0%} of assets. {br['footer']}",
             fontsize=6.8, color=br["muted"])
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


def draw_joint_map(cfg, r, path, title):
    br, cr = cfg["branding"], cfg["criterion"]
    setup_font(br)
    fig, ax = plt.subplots(figsize=(br["figsize"][0] * 0.8, 5.0), dpi=br["dpi"], facecolor=br["background"])
    X, Y = np.meshgrid(r["fracs"] * 100, r["hrs"] * 100)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("brand", ["#FFFFFF", br["accent"]])
    im = ax.pcolormesh(X, Y, r["g_mc"] * 100, cmap=cmap, shading="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("Growth of market-consistent funding ratio, % p.a.", fontsize=8)
    c1 = ax.contour(X, Y, r["b_mc"], levels=[cr["limit_market"]], colors=[br["primary"]], linewidths=2)
    c2 = ax.contour(X, Y, r["b_st"], levels=[cr["limit_statutory"]], colors=[br["muted"]],
                    linewidths=2, linestyles="dotted")
    ax.clabel(c1, fmt={cr["limit_market"]: f"market {cr['limit_market']:.0%}"}, fontsize=7.5)
    ax.clabel(c2, fmt={cr["limit_statutory"]: f"statutory {cr['limit_statutory']:.0%}"}, fontsize=7.5)
    bad = ~r["ok"]
    ax.scatter(X[bad], Y[bad], marker="x", s=10, color=br["muted"], alpha=0.5, lw=0.8)
    if r["opt"]:
        ax.plot(r["opt"]["ils"] * 100, r["opt"]["hedge"] * 100, "*", ms=16, color=br["primary"],
                mec="white", mew=1)
    ax.set_xlabel(f"ILS share of the {cfg['decision']['share']:.0%} sleeve (rest: longevity ETF), %")
    ax.set_ylabel("Interest-rate hedge ratio, %")
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    fig.text(0.02, 0.012, f"x = breaches a limit. Star = optimum. {br['footer']}", fontsize=6.8, color=br["muted"])
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


# =============================================================================
# "When": optimum conditional on today's market state (full model)
# =============================================================================
def draw_when(cfg, when, path):
    br, cr = cfg["branding"], cfg["criterion"]
    setup_font(br)
    labs = list(when.keys())
    fig, ax = plt.subplots(figsize=(br["figsize"][0], 3.6), dpi=br["dpi"], facecolor=br["background"])
    ax.axis("off")
    cols = [br["neutral"], br["primary"], br["accent"]]
    for k, (lab, col) in enumerate(zip(labs, cols)):
        r = when[lab]
        o, b0 = r["opt"], r["base"]
        x = 0.02 + k * 0.33
        ax.text(x, 0.92, lab, fontsize=11, fontweight="bold", color=col, transform=ax.transAxes)
        if o is None:
            lr = r["least_risk"]
            body = (f"no combination meets both limits\n\nleast-risk levers:\n"
                    f"hedge ratio {lr['hedge']:.0%}\n"
                    f"sleeve {lr['ils']:.0%} ILS / {1 - lr['ils']:.0%} ETF\n"
                    f"P(<100%): market {lr['b_mc']:.1%}, statutory {lr['b_st']:.1%}\n"
                    f"growth {100 * lr['growth']:.2f}% p.a.\n\n"
                    f"start (no reallocation):\nmarket {b0['b_mc']:.1%}, statutory {b0['b_st']:.1%}")
        else:
            body = (f"hedge ratio {o['hedge']:.0%}\n"
                    f"sleeve {o['ils']:.0%} ILS / {1 - o['ils']:.0%} ETF\n"
                    f"P(<100%): market {o['b_mc']:.1%}, statutory {o['b_st']:.1%}\n"
                    f"at year 10: market {o['e_mc']:.1%}, statutory {o['e_st']:.1%}\n"
                    f"growth {100 * o['growth']:.2f}% p.a.\n"
                    f"binding: {', '.join(r['binding']) or 'none'}\n\n"
                    f"start (no reallocation):\nmarket {b0['b_mc']:.1%}, statutory {b0['b_st']:.1%}")
        ax.text(x, 0.80, body, fontsize=9, va="top", color=br["text"], transform=ax.transAxes, linespacing=1.4)
    fig.suptitle("When: the optimal levers depend on the state the fund is in today", x=0.02, ha="left",
                 fontsize=12.5, fontweight="bold", color=br["text"])
    fig.text(0.02, 0.02, f"Full model (shared state, longevity, {cfg['simulation']['horizon_years']} years). "
             f"Today's market state fixed; persistence {cfg['market_regime']['adverse_stay']:.0%}. {br['footer']}",
             fontsize=6.8, color=br["muted"])
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


# =============================================================================
# Figure: decision ladder
# =============================================================================
def setup_font(br):
    if br.get("font_file"):
        font_manager.fontManager.addfont(br["font_file"])
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = br["font_family"] if br["font_family"] in names else "DejaVu Sans"


def draw_ladder(cfg, ladder, path):
    """Ladder of modelling steps; each step measured in both funding ratios."""
    br, d = cfg["branding"], cfg["decision"]
    setup_font(br)
    n = len(ladder)
    fig, axs = plt.subplots(1, 2, figsize=br["figsize"], dpi=br["dpi"], facecolor=br["background"],
                            sharey=True)
    y = np.arange(n)[::-1]
    xmax = max(step[v][c]["breach"] for step in ladder for v in ("statutory", "market")
               for c in ("ils", "etf"))
    xmax = max(xmax, limit_for(cfg, "market")) * 1.9

    def short(r):
        adm = {"both": "both", "ILS": "ILS only", "ETF": "ETF only", "neither": "neither"}[r["constraint_choice"]]
        return f"admissible: {adm}\n\u2192 {r['constrained_kelly']}"

    for ax, val, title in ((axs[0], "statutory", "Statutory funding ratio\n(fixed technical rate)"),
                           (axs[1], "market", "Market-consistent funding ratio\n(rates and longevity priced)")):
        for off, cand, col in ((0.18, "ils", br["primary"]), (-0.18, "etf", br["accent"])):
            vals = [step[val][cand]["breach"] for step in ladder]
            ax.barh(y + off, vals, height=0.34, color=col, label=cfg[cand]["label"])
            for yy, v in zip(y + off, vals):
                ax.text(v, yy, f" {v:.1%}", va="center", fontsize=8, color=br["text"])
        lim = limit_for(cfg, val)
        ax.axvline(lim, color=br["muted"], ls="--", lw=1)
        ax.text(lim, n - 0.45, f"limit {lim:.0%}", ha="center", fontsize=7.5, color=br["muted"])
        ax.set_xlim(0, xmax)
        ax.set_xticks([])
        ax.set_title(title, fontsize=10, loc="left", color=br["text"])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        for yy, step in zip(y, ladder):
            ax.text(0.98, yy, short(step[val]), transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=8, color=br["primary"], fontweight="bold")

    axs[0].set_yticks(y)
    axs[0].set_yticklabels([step["label"] for step in ladder], fontsize=9)
    axs[1].spines["left"].set_visible(False)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.015, 0.905), ncol=2)

    fig.suptitle(br["title"], x=0.02, ha="left", fontsize=13, fontweight="bold", color=br["text"])
    fig.text(0.02, 0.925, br["subtitle"], fontsize=9, color=br["muted"])
    foot = (f"P(funding ratio < {d['fr_floor']:.0%} at any year-end within the horizon), candidate "
            f"{d['share']:.0%} of assets. Decision = highest Kelly growth among admissible shares "
            f"(grid 0-{d['share_grid_max']:.0%}). {br['footer']}")
    fig.text(0.02, 0.012, foot, fontsize=6.8, color=br["muted"])
    plt.tight_layout(rect=(0, 0.03, 1, 0.86))
    fig.subplots_adjust(top=0.80)
    fig.savefig(path, facecolor=br["background"])
    plt.close(fig)


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--dump-config", action="store_true")
    args = ap.parse_args()
    cfg = copy.deepcopy(CONFIG)
    if args.config:
        deep_update(cfg, json.loads(Path(args.config).read_text()))
    if args.dump_config:
        print(json.dumps(cfg, indent=2))
        return

    sim = cfg["simulation"]
    rng = np.random.default_rng(sim["seed"])
    n, h = sim["n_paths"], sim["horizon_years"]
    state_world = simulate_state_world(cfg, rng, n, h)
    worlds = {"multivariate": state_world,
              "univariate": simulate_univariate_world(cfg, rng, n, h, state_world),
              "normal": simulate_normal_world(cfg, rng, n, h, state_world)}

    # all 8 combinations -> CSV
    rows = []
    for val, tails, scope, horizon in itertools.product(("statutory", "market"),
                                                        ("normal", "univariate", "multivariate"),
                                                        ("partial", "total"), ("one", "multi")):
        r = evaluate(cfg, worlds, tails, scope, horizon, val)
        rows.append((val, tails, scope, horizon, r))
    with open(cfg["output"]["csv"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["valuation", "tails", "scope", "horizon", "ils_breach", "etf_breach", "ils_kelly_share",
                    "ils_kelly_growth", "etf_kelly_share", "etf_kelly_growth",
                    "constraint_choice", "kelly_choice", "constraint_first_kelly"])
        for val, tails, scope, horizon, r in rows:
            w.writerow([val, tails, scope, horizon, f"{r['ils']['breach']:.4f}", f"{r['etf']['breach']:.4f}",
                        r["ils"]["kelly_share"], f"{r['ils']['kelly_growth']:.5f}",
                        r["etf"]["kelly_share"], f"{r['etf']['kelly_growth']:.5f}",
                        r["constraint_choice"], r["kelly_choice"], r["constrained_kelly"]])

    lookup = {(v, t, s, hz): r for v, t, s, hz, r in rows}
    steps = [
        ("Conventional:\nnormal tails, market\nrisk only, 1 year", ("normal", "partial", "one")),
        (f"+ {h} years", ("normal", "partial", "multi")),
        ("+ heavy tails\n(univariate)", ("univariate", "partial", "multi")),
        ("+ shared state", ("multivariate", "partial", "multi")),
        ("+ longevity\n(total risk)", ("multivariate", "total", "multi")),
    ]
    ladder = [{"label": lab, "statutory": lookup[("statutory",) + key], "market": lookup[("market",) + key]}
              for lab, key in steps]
    draw_ladder(cfg, ladder, cfg["output"]["figure"])

    # split study across the same steps, both valuations
    split = {v: [evaluate_split(cfg, worlds, *key, v) for _, key in steps] for v in ("statutory", "market")}
    draw_split(cfg, split, [lab for lab, _ in steps], cfg["output"]["split_figure"])
    with open(cfg["output"]["split_csv"], "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["valuation", "step", "ils_fraction", "breach", "growth"])
        for v in split:
            for (lab, _), r in zip(steps, split[v]):
                for fr_, b_, g_ in zip(r["fracs"], r["breach"], r["growth"]):
                    wr.writerow([v, lab.replace(chr(10), " "), fr_, f"{b_:.4f}", f"{g_:.5f}"])
    # overarching criterion through the agreed steps
    joint = [evaluate_joint(cfg, worlds, *key) for _, key in steps]
    draw_joint_ladder(cfg, steps, joint, cfg["output"]["joint_figure"])
    draw_joint_map(cfg, joint[-1], cfg["output"]["joint_map"],
                   f"Full model: where the optimum sits ({h} years)")
    with open(cfg["output"]["joint_csv"], "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "hedge_ratio", "ils_fraction", "p_market", "p_statutory",
                     "p_market_final_year", "p_statutory_final_year", "growth_market", "admissible"])
        for (lab, _), r in zip(steps, joint):
            for i, hr in enumerate(r["hrs"]):
                for j, fr_ in enumerate(r["fracs"]):
                    wr.writerow([lab.replace(chr(10), " "), hr, fr_, f"{r['b_mc'][i, j]:.4f}",
                                 f"{r['b_st'][i, j]:.4f}", f"{r['e_mc'][i, j]:.4f}", f"{r['e_st'][i, j]:.4f}",
                                 f"{r['g_mc'][i, j]:.5f}", bool(r["ok"][i, j])])
    print("Optimum under the overarching criterion:")
    for (lab, _), r in zip(steps, joint):
        o = r["opt"]
        print(f"  {lab.replace(chr(10), ' '):45s} " + ("none" if o is None else
              f"hedge {o['hedge']:.0%} | ILS {o['ils']:.0%} / ETF {1 - o['ils']:.0%} | "
              f"P market {o['b_mc']:.1%}, statutory {o['b_st']:.1%} | growth {100 * o['growth']:.2f}% | "
              f"binding {r['binding']} | final year {o['e_mc']:.1%} / {o['e_st']:.1%}")
              + f"\n  {'':45s} start (no reallocation, hedge {r['base']['hedge']:.0%}): P market {r['base']['b_mc']:.1%}, "
                f"statutory {r['base']['b_st']:.1%} | final year {r['base']['e_mc']:.1%} / {r['base']['e_st']:.1%} | "
                f"growth {100 * r['base']['growth']:.2f}%"
                + ("" if r['base']['ok'] else " | breaches a limit"))

    # "when": same full model, but today's market state fixed (calm vs crash)
    when = {"Unconditional": joint[-1]}
    for lab, st in (("Today calm", 0), ("Today in crash", 1)):
        w_st = simulate_state_world(cfg, np.random.default_rng(sim["seed"] + 1 + st), n, h, start_state=st)
        when[lab] = evaluate_joint(cfg, {"multivariate": w_st}, "multivariate", "total", "multi")
    draw_when(cfg, when, cfg["output"]["when_figure"])
    print("When (full model, optimum by today's market state):")
    for lab, r in when.items():
        o = r["opt"]
        lr = r["least_risk"]
        print(f"  {lab:15s} " + (f"none admissible; least-risk: hedge {lr['hedge']:.0%} | ILS {lr['ils']:.0%} / "
                                 f"ETF {1 - lr['ils']:.0%} | P market {lr['b_mc']:.1%}, statutory {lr['b_st']:.1%} | "
                                 f"growth {100 * lr['growth']:.2f}%" if o is None else
              f"hedge {o['hedge']:.0%} | ILS {o['ils']:.0%} / ETF {1 - o['ils']:.0%} | P market {o['b_mc']:.1%}, "
              f"statutory {o['b_st']:.1%} | final year {o['e_mc']:.1%} / {o['e_st']:.1%} | "
              f"growth {100 * o['growth']:.2f}% | binding {r['binding']}")
              + f" | start: {r['base']['b_mc']:.1%} / {r['base']['b_st']:.1%}")

    hedge = {v: evaluate_hedge(cfg, worlds, v) for v in ("statutory", "market")}
    draw_hedge(cfg, hedge, cfg["output"]["hedge_figure"])
    with open(cfg["output"]["hedge_csv"], "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["valuation", "hedge_ratio", "breach_at_safest_split", "safest_ils_fraction",
                     "growth", "breach_all_ils", "breach_all_etf"])
        for v in hedge:
            for r in hedge[v]:
                wr.writerow([v, r["ratio"], f"{r['breach']:.4f}", r["ils_fraction"], f"{r['growth']:.5f}",
                             f"{r['breach_all_ils']:.4f}", f"{r['breach_all_etf']:.4f}"])
    print("Hedge ratio (full model, safest split):")
    for v in hedge:
        print("  " + v + ": " + "  ".join(f"{r['ratio']:.0%}:{r['breach']:.1%}/ILS{r['ils_fraction']:.0%}"
                                         for r in hedge[v]))
    print("Split of the sleeve (ILS fraction):")
    for v in split:
        for (lab, _), r in zip(steps, split[v]):
            print(f"  {v:9s} {lab.replace(chr(10), ' '):45s} min-risk {r['min_risk']:.0%} "
                  f"(P {r['breach'].min():.1%}) | growth-optimal within limit "
                  f"{'-' if r['growth_opt'] is None else format(r['growth_opt'], '.0%')} | "
                  f"admissible ILS range {r['adm_range']}")

    for step in ladder:
        line = step["label"].replace(chr(10), " ")
        for val in ("statutory", "market"):
            r = step[val]
            line += (f" | {val[:4]}: ILS {r['ils']['breach']:5.1%} ETF {r['etf']['breach']:5.1%}"
                     f" -> {r['constrained_kelly']}")
        print(line)
    hr = hedge_ratios(cfg, cfg["decision"]["share"])
    print(f"  hedge ratio: asset duration {hr['asset_duration']:.2f} vs liabilities {hr['liability_duration']:.1f} | "
          f"funding-ratio hedge {hr['fr_hedge']:.0%} | surplus hedge {hr['surplus_hedge']:.0%} | "
          f"FR change per -100bp {-hr['fr_sensitivity_per_100bp']:+.2f}% (market-consistent); "
          f"statutory: liabilities duration 0, bonds are unhedged FR risk")
    print(f"  average rate change p.a.: {worlds['multivariate']['d_rate'].mean():+.3f} %-pts")
    w0 = worlds["multivariate"]
    for k in ("equity", "ils", "etf"):
        x = w0[k][:, 0]
        print(f"  marginal {k:6s} mean {x.mean():5.1f}  sd {x.std():5.1f}  1y 0.5% {np.percentile(x, 0.5):6.1f}")
    print(f"wrote {cfg['output']['figure']} and {cfg['output']['csv']}")


if __name__ == "__main__":
    main()
