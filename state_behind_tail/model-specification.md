---
title: "Pension ALM study: model specification"
author: "Bernhard von Boyen"
date: "22 November 2026"
geometry: margin=2.2cm
fontsize: 10pt
mainfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
header-includes:
  - \usepackage{amsmath,amssymb}
  - \usepackage{booktabs}
  - \usepackage{longtable}
---

Mathematical specification of `pension_study.py` and `pension_sensitivity.py`, the model behind the article *The State Behind the Tail*.

## Scope and files

The model simulates annual paths of a closed pension fund's balance sheet over a horizon $T$ of 1 or 10 years and chooses two levers under one criterion. All returns and rate changes below are in percent (rate changes in percentage points) unless stated otherwise.

| File | What it computes | Outputs |
|:--|:--|:--|
| `pension_study.py` | Scenario worlds, projection, ladder of modelling steps, joint optimisation, conditioning on today's state, single-candidate, split and hedge-ratio views | Optimum ladder, map, when-panel, backup figures, CSV files |
| `pension_sensitivity.py` | Ranges: risk appetite, budget, state calibration (full model) | Decision-rule and state-calibration figures, CSV |
| `decision_matrix.py` | Earlier stylised two-asset model (decision matrix, Kelly curves) | Two figures |

All parameters sit in the `CONFIG` dictionary at the top of `pension_study.py` and can be overridden with `--config overrides.json`. Simulation: $N = 200{,}000$ paths, seed 42.

## Fast market regime

A two-state Markov chain $S_t \in \{0 = \text{calm},\, 1 = \text{crash}\}$ drives equities and rates. The stationary crash share is $\pi$ and the persistence is $p_{aa} = P(S_{t+1}=1 \mid S_t=1)$. The calm-to-crash probability is set so that $\pi$ stays the stationary share:

$$
p_{ca} = P(S_{t+1}=1 \mid S_t=0) = \frac{\pi\,(1-p_{aa})}{1-\pi}
$$

The first year's state is drawn from the stationary distribution (unconditional runs) or from the transition row of a fixed state today (conditional runs).

Equity returns are normal within each state:

$$
R^{E}_t \mid S_t = s \;\sim\; \mathcal{N}\big(\mu^{E}_s,\, (\sigma^{E}_s)^2\big)
$$

The annual change of the single discount rate (parallel shift) is normal within each state, with the calm mean set for zero average drift:

$$
\Delta r_t \mid S_t = s \;\sim\; \mathcal{N}\big(\mu^{r}_s,\, (\sigma^{r})^2\big), \qquad \mu^{r}_0 = -\frac{\pi\,\mu^{r}_1}{1-\pi}
$$

Rates rise slowly in calm years and fall sharply in crashes (flight to quality).

## Slow longevity state

Longevity is a persistent drift $d_t$ in the liability value (percent per year), an AR(1) process started in its stationary distribution:

$$
d_1 \sim \mathcal{N}\!\left(0,\ \frac{\sigma_d^2}{1-\phi^2}\right), \qquad d_t = \phi\, d_{t-1} + \eta_t,\quad \eta_t \sim \mathcal{N}(0,\sigma_d^2)
$$

The annual longevity effect on liabilities adds independent noise:

$$
L^{\mathrm{lon}}_t = d_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\sigma_\varepsilon^2)
$$

The drift is independent of the market regime. With $\phi = 0.95$ and $\sigma_d = 0.5$, the stationary drift has a standard deviation of 1.6 percentage points a year, and it persists for many years.

## Assets

**Bonds** carry duration $D_B$ and convexity $C_B = c\,D_B^2$, with $c$ the convexity factor (1 in the base case). With $\Delta r$ in percentage points, the convexity term is scaled by $1/100$:

$$
R^{B}_t = y - D_B\,\Delta r_t + \tfrac{1}{2}\, c\, D_B^2\, \frac{\Delta r_t^2}{100}
$$

**Cash** earns a fixed rate $r_{\text{cash}}$.

**ILS fund**: carry minus catastrophe losses, independent of markets and longevity:

$$
R^{ILS}_t = \kappa + \nu_t - \mathbb{1}\{U_t < p_{\text{cat}}\}\, X_t, \qquad \nu_t \sim \mathcal{N}(0,\sigma_\nu^2),\quad X_t \sim \mathcal{U}(\ell_{\min}, \ell_{\max})
$$

An optional collateral loss in crash years (2008 type) is available and switched off by default.

**Longevity ETF**: loads on the equity deviation and on the longevity drift, plus its own noise:

$$
R^{ETF}_t = \mu_{ETF} + \beta\,\big(R^{E}_t - \overline{R^{E}}\big) + \lambda\, d_t + \zeta_t, \qquad \zeta_t \sim \mathcal{N}(0,\sigma_\zeta^2)
$$

where $\overline{R^{E}}$ is the sample mean of equity returns. With $\beta = 0.6$, $\lambda = 8$ and $\sigma_\zeta = 6$, the ETF's volatility (17.7%) equals that of equities; its correlation with equities is 0.6.

**Portfolio return** with candidate share $s$ funded from equities, weights $w_E = 0.30 - s$, $w_B = 0.65$, $w_C = 0.05$, and ILS fraction $f$ of the sleeve:

$$
R^{A}_t = w_E R^{E}_t + w_B R^{B}_t + w_C\, r_{\text{cash}} + s\,\big(f\,R^{ILS}_t + (1-f)\,R^{ETF}_t\big)
$$

**Swap overlay.** For a target hedge ratio $h$ (funding-ratio terms), the overlay closes the gap between the target asset duration $h\,D_L$ and the physical bond duration $w_B D_B$. It carries swap convexity $c\,D_L$ per unit of overlay duration and a cost of $k$ basis points per unit of duration a year:

$$
D_{ov} = h\, D_L - w_B D_B
$$

$$
R^{A}_t \;\leftarrow\; R^{A}_t - D_{ov}\,\Delta r_t + \tfrac{1}{2}\, c\, D_L\, D_{ov}\, \frac{\Delta r_t^2}{100} - |D_{ov}|\,\frac{k}{100}
$$

Without an overlay the hedge ratio is $w_B D_B / D_L = 0.65 \times 13 / 14 = 60\%$. In surplus terms it is $FR_0 \times 60\% = 71\%$.

## Liabilities

Liabilities accrue at the technical rate $r_{\text{tech}}$. Their valuation follows one of two views.

**Market-consistent**: duration $D_L$ and convexity $c\,D_L^2$ on the rate change; in the total-risk scope the longevity effect is added:

$$
R^{L,mc}_t = r_{\text{tech}} - D_L\,\Delta r_t + \tfrac{1}{2}\, c\, D_L^2\, \frac{\Delta r_t^2}{100} + \mathbb{1}_{\text{total}}\, L^{\mathrm{lon}}_t
$$

**Statutory**: discounted at the fixed technical rate, with mortality tables not updated within the horizon, so neither rates nor longevity move the liability value:

$$
R^{L,st}_t = r_{\text{tech}}
$$

In the market-risk-only scope (partial), the longevity term is dropped in both views. The ETF still carries its longevity loading there, where it acts as unrewarded noise.

## Balance-sheet projection

Start with liabilities $L_0 = 1$ and assets $A_0 = FR_0$. Each year both grow at their returns (each floored at $-95\%$), then pensions of $b$ times the grown liability value are paid out of assets:

$$
\tilde A_t = A_{t-1}\,(1 + R^{A}_t/100), \qquad \tilde L_t = L_{t-1}\,(1 + R^{L}_t/100)
$$

$$
A_t = \max\!\big(\tilde A_t - b\,\tilde L_t,\ 10^{-6}\big), \qquad L_t = (1-b)\,\tilde L_t, \qquad FR_t = A_t / L_t
$$

The payment makes the funding ratio path-dependent. With $x_t$ the funding ratio before payment, it becomes $(x_t - b)/(1 - b)$: it rises when the fund is over 100% and falls further when it is under.

## Risk and growth measures

All measures are averages over the $N$ simulated paths, in each valuation view $v$ (market-consistent or statutory), with floor $F = 100\%$.

**Underfunding probability (path-based, the limit's measure)**:

$$
P^{v} = P\Big(\min_{1 \le t \le T} FR^{v}_t < F\Big)
$$

**Final-year underfunding (complementary view, reported only)**:

$$
P^{v}_{T} = P\big(FR^{v}_T < F\big)
$$

**Growth (objective, Kelly-type)**: expected log growth of the market-consistent funding ratio per year:

$$
g = \frac{1}{T}\, \mathbb{E}\big[\ln(FR^{mc}_T / FR_0)\big]
$$

The path-based measure is stricter than a share of year-end states: at the full-model optimum, 14.1% of paths dip below 100% at least once, 6.1% of all year-end states are below 100%, and 9.2% of paths are below 100% at year 10.

## Levers, criterion and search

The reallocation budget $s = 15\%$ of assets is fixed. Two levers are chosen jointly: the ILS fraction $f$ of the sleeve (0 to 1, steps of 0.1) and the hedge ratio $h$ (40% to 120%, steps of 10 points): $11 \times 9 = 99$ combinations.

**Criterion.** Maximise growth subject to both underfunding limits:

$$
\max_{f,\,h}\ g(f,h) \quad \text{s.t.} \quad P^{mc}(f,h) \le \alpha_{mc} = 15\%, \qquad P^{st}(f,h) \le \alpha_{st} = 10\%
$$

Economics steers (growth on the market-consistent funding ratio); both views bound survival.

**Binding limit.** A limit is reported as binding at the optimum if some combination with higher growth breaches it.

**No admissible combination.** The least-risk combination is reported instead:

$$
(f,h)^{\ast} = \arg\min_{f,\,h}\ \max\!\left(\frac{P^{mc}}{\alpha_{mc}},\ \frac{P^{st}}{\alpha_{st}}\right)
$$

The same simulated paths are used for every combination, so differences between combinations are not blurred by sampling noise.

## Modelling worlds and the ladder

Three scenario worlds are generated. The state world comes first; the other two are matched to its year-1 means, volatilities and correlations, so each step changes one thing only.

| World | Dependence and tails | Time structure |
|:--|:--|:--|
| Normal | Equity, rate change, longevity effect, ETF and ILS jointly normal (means and covariance of the state world) | Independent years |
| Heavy tails (univariate) | Gaussian copula with the state world's correlations; Student-t marginals (4 degrees of freedom) rescaled to the same mean and volatility; ILS keeps its catastrophe tail and stays independent | Independent years |
| Shared state (multivariate) | Market regime and longevity drift as specified above | Persistent states |

The ladder evaluates the criterion step by step, in both valuation views:

1. Conventional: normal world, market risk only (partial), $T = 1$.
2. Plus 10 years: normal world, partial, $T = 10$.
3. Plus heavy tails: heavy-tail world, partial, $T = 10$.
4. Plus shared state: state world, partial, $T = 10$.
5. Plus longevity: state world, total risk, $T = 10$ (the full model).

## Conditioning, starting point and ranges

**Starting point.** For each step, the same measures are computed with no reallocation ($s = 0$) and no overlay (hedge ratio 60% from physical bonds).

**Conditioning on today's state ("when").** In the full model, today's market state $s_0$ is fixed (calm or crash); the first simulated year's state is drawn from its transition row, so a crash today continues with probability $p_{aa}$. The joint optimisation is repeated on these paths (fresh seeds per state). The funding ratio starts at $FR_0$ in both cases.

**Ranges** (`pension_sensitivity.py`, full model, 100,000 paths):

- Risk appetite: the criterion is re-applied on one evaluated grid for $\alpha_{mc} \in \{10\%, 15\%, 20\%\}$ and $\alpha_{st} \in \{5\%, 10\%, 15\%\}$, budget 15%.
- Budget: $s \in \{10\%, 15\%, 20\%\}$ at the agreed limits; each budget is re-evaluated.
- State calibration, one parameter at a time, at the agreed limits: crash share 10% / 18%; crash persistence 25% / 65%; equity crash mean $-15\%$ / $-35\%$; rate change in a crash $-0.30$ / $-0.90$ points; longevity persistence 0.85 / 0.98; longevity drift volatility 0.3 / 0.7. Each variation re-simulates the state world and re-optimises.

## Base calibration

All values hand-set and illustrative; `CONFIG` key in brackets.

| Block | Parameter | Value |
|:--|:--|:--|
| Simulation | Paths, horizon, seed (`simulation`) | 200,000; 1 or 10 years; 42 |
| Market regime | Crash share, persistence (`market_regime`) | 14%; 50% (calm-to-crash 8.1%) |
| Equities | Calm mean / vol; crash mean / vol (`equity`) | 8% / 14%; $-25\%$ / 10% (overall mean 3.4%, vol 17.7%) |
| Rates | Rate change calm mean / crash mean; vol (`rates`) | $+0.10$ / $-0.60$ points; 0.35 |
| Bonds | Yield, duration, convexity factor (`rates`) | 2%; 13; 1.0 |
| Liabilities | Technical rate, duration, pensions paid (`liabilities`) | 2%; 14; 4% of liabilities a year |
| Longevity | $\phi$, drift vol, noise vol (`longevity`) | 0.95; 0.5; 0.3 |
| Allocation | Equities / bonds / cash, cash rate, $FR_0$ (`allocation`) | 30% / 65% / 5%; 1%; 118%; candidate funded from equities |
| ILS | Carry, noise vol, event probability, loss range (`ils`) | 7%; 2%; 5%; 25% to 70% (mean 4.6%, vol 10.9%) |
| Longevity ETF | Mean, equity beta, longevity loading, own vol (`etf`) | 6%; 0.6; 8; 6% (vol 17.7%, correlation with equities 0.6) |
| Overlay | Cost per unit of duration, hedge grid (`hedge`) | 1 bp a year; 40% to 120% |
| Decision | Budget, floor, heavy-tail degrees of freedom (`decision`) | 15% of assets; 100%; 4 |
| Criterion | Limits market / statutory, split step (`criterion`) | 15% / 10%; 10% |

## Simplifications and numerical notes

- One discount rate with parallel shifts; second-order (convexity) price effects, no curve shape, no rate floor, no mean reversion of the rate level.
- No term premium; the bond yield equals the technical rate, so duration earns nothing beyond convexity.
- Closed fund: no active members, contributions or management actions (recovery measures, rebalancing rules); a constant payout rate instead of a cash-flow schedule.
- Statutory liabilities ignore longevity within the horizon (in practice mortality tables are updated every few years).
- Collateral and liquidity of the swap overlay are not modelled; the overlay cost is a flat fee.
- The shared state and its persistence enter the ladder in one step; they are not separated.
- Annual time step: risk between year-ends is not measured.
- Returns are floored at $-95\%$ a year and assets at $10^{-6}$ to keep the funding ratio and its logarithm defined.
- The same seed is used across combinations within a run (common random numbers); the conditional runs use separate seeds per starting state.
- Runtime: `pension_study.py` about 4 minutes at 200,000 paths; `pension_sensitivity.py` about 2 minutes at 100,000 paths.
