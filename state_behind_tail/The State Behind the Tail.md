# The State Behind the Tail

Nov 22, 2026 · @B

## One factor, two sides of the balance sheet

The CRO Forum's new paper, *Balancing the Balance Sheet*, describes ALM as balance-sheet steering across several objectives at once: solvency, economic value, liquidity, earnings. It is a careful survey of practice, and it argues from the one-year view.

Eurovita fits that view well. One factor, rates, hit both sides of the balance sheet. Bonds fell, policyholders surrendered, and the redemptions could only be met by selling assets that had themselves lost value.

The factor was known. Its regime was not. Rates move in regimes, from a decade of near-zero yields to the sharp turn of 2022, yet most ALM models treat them as one stable process: a one-year shock, or a mean-reverting generator with fixed parameters.

This article asks what happens when the link between assets and liabilities is not a single shock, but a state that persists. In a one-year view, the pension fund I study below looks safe: its chance of becoming underfunded is 0.4%. Over ten years it is 22% to 28%.

## A missing state

It started with a chart on LinkedIn: the distribution of S&P 500 yearly returns, a normal curve, a t-distribution and the empirical density. The empirical curve had a sharp peak and a second bump far in the left tail.

My first thought was fatter tails. So I simulated monthly returns with t-distributed shocks and compounded them to a year. The shape barely moved: the peak rose by 6%, the quantiles stayed where they were. Twelve fat-tailed months average out into a nearly normal year.

What produced the bump was a second state: a calm regime and a crash regime, with a Markov chain switching between them. Place the crash mean about 2.5 standard deviations below the calm mean and give it a low volatility, and the second mode appears. I went looking for fatter tails and found a missing state.

&#91;image: Annual return density of a two-state model as the crash state moves further into the tail\]

Moving the crash mean from 1.96 to 3.5 calm standard deviations below the calm mean: the shoulder becomes a second mode, and the 1-in-200 loss falls from -40% to -62% while the mean barely moves.

## Why I never bought fatter marginal tails

Multi-year ALM studies are often sold on fatter tails for each risk, with 2008 as the motivation. I never bought it. In my experience the result is decided in the aggregate tail, and fattening each marginal while keeping the first two moments mostly diversifies away.

A simple test makes the point. Standardise every risk to the same mean and variance, give each a Pareto tail, and add them up:

| Loss at 99.5%, in standard deviations | One risk | Sum of 5 | Sum of 20 |
| --- | --- | --- | --- |
| Normal marginals | 2.57 | 2.58 | 2.57 |
| Pareto marginals (alpha 3.5) | 3.77 | 3.32 | 2.98 |
| Near-normal marginals, shared state | 2.70 | 4.92 |  |

The fat marginal tail is 47% above normal on its own, 29% above for five risks and 16% for twenty. Below the far tail, at 95%, the fat-tailed sum is even thinner than normal. A shared state does the opposite: near-normal risks, an aggregate tail 1.9 times the normal one.

## Where the effort goes, and where the result is decided

Over the years I have built multi-year ALM studies and seen many more applied. The pattern is consistent: the modelling effort goes into making individual risks more extreme, while the result is decided elsewhere.

| What usually gets the attention | What decides the result in my experience |
| --- | --- |
| Fatter tails for individual risks | The states those risks share |
| One-year shocks | How long a state persists |
| Market and credit risk | Assets and liabilities in one model, longevity included |
| Pure constraints: does the allocation pass the limits? | The optimal allocation within the limits |
| One funding ratio | Where the economic and statutory views diverge |
| One optimal allocation | Levers that depend on where the fund is today |

Each row has a number behind it in the study below. None of them needs a more complex model. They need the right state, written down, and one criterion that puts survival first and growth second.

## Shared states on two timescales

Two states matter for a pension fund, and they run at very different speeds.

**The fast state is the market regime.** In a crash, equities fall, rates fall as money flees to safety, and the crash tends to last: in the study, a crash year is followed by another one half the time. Equities and the liability value then move against the fund in the same years.

**The slow state is longevity.** When mortality improves, more people survive into old age, and the older cohorts grow. Cohort sizes drive demand: DellaVigna and Pollet showed that one percentage point of demographic demand growth predicts 5 to 10 percentage points of abnormal annual returns for the industries serving that demand, and that only demand five to ten years ahead is priced in too slowly. The same slow state that lengthens pension payments shifts demand towards health, care and services. One driver, two channels.

I have seen the same pattern in a very different place. In my soccer simulation lab, long-term predictions for individual teams only became realistic once they allowed for idiosyncratic regime shifts, such as a key player's injury. A state changes the path, not just one result.

## Building on my earlier posts

In [ALM as a System](https://www.augmentedactuary.com/2025/12/08/alm-system.html) I described ALM as one engine that joins the economic, liquidity, capital and accounting views. In [Risk Capacity, Exposure, and Order](https://www.augmentedactuary.com/2026/01/15/Exposure-ALM.html) I argued that risk capacity must be set before allocation, along two axes: aggregation across exposures and time.

At a former employer I built a multi-objective exposure framework that answered the first half of that: where risk should sit. What it could not answer was how much risk the balance sheet can carry over time, and when that answer changes. Both questions need a state. The study below adds it.

## The study: a pension fund, three decisions, one criterion

The answer the full model gives is specific: hedge 70% of the interest-rate risk and split the reallocation 40% ILS, 60% longevity ETF. A conventional set-up would have said: no hedge change, everything into the ETF.

**The fund.** A closed pension fund, pensioners only, funded at 118%, paying 4% of its liabilities a year. It holds 30% equities, 65% bonds and 5% cash, which hedges 60% of its interest-rate risk in funding-ratio terms.

**The decisions.** Move 15% of assets out of equities and split it between an ILS fund (independent of markets, with a catastrophe tail) and an equity ETF that benefits from longevity (health, care, services; same volatility as equities). Set the interest-rate hedge ratio between 40% and 120%.

**The criterion.** Maximise the growth of the market-consistent funding ratio, subject to two limits over ten years: the chance of falling below 100% at any year-end stays at or below 15% in the market-consistent view and at or below 10% in the statutory view. Survival first, growth second.

**The ladder.** The same fund and the same decisions, with one modelling step added at a time:

| Step | Hedge ratio | Sleeve ILS / ETF | Underfunding risk, market / statutory | Growth p.a. |
| --- | --- | --- | --- | --- |
| Conventional: normal tails, market risk only, 1 year | 60% | 0 / 100 | 0.2% / 0.0% | 1.38% |
| + 10 years | 60% | 0 / 100 | 13.8% / 7.5% | 1.55% |
| + heavy tails | 60% | 0 / 100 | 12.3% / 7.1% | 1.55% |
| + shared market state | 70% | 50 / 50 | 14.5% / 8.4% | 1.47% |
| + longevity (total risk) | 70% | 40 / 60 | 14.1% / 9.7% | 1.51% |

Five things stand out.

1. **The starting point fails.** Without a reallocation, the fund breaches both limits in every ten-year step: 22% to 28% market-consistent, 12% to 14% statutory. Each modelling step makes it look worse. The fund has to act.
2. **Heavy tails change nothing.** Risk even falls slightly and the decision stays the same. That is the aggregation table above, inside a real decision.
3. **The shared state changes the decision.** Crashes that last move the hedge ratio from 60% to 70% and half the sleeve into ILS, because falling rates and falling equities now arrive together.
4. **Longevity creates room, and the statutory view limits it.** Counted as a liability risk, longevity turns the ETF into a partial hedge and the split moves back towards it. The statutory limit then binds at 9.7%, because statutory liabilities recognise longevity only with a lag.
5. **Constraints alone do not decide.** In the full model, 62 combinations pass the market-consistent limit and 11 pass both. The growth criterion picks one, and names the limit that holds the fund back.

&#91;image: Full model: growth of the funding ratio over split and hedge ratio, with both limits\]

In the full model, the combinations that meet both limits form a narrow corridor: a 70% to 80% hedge ratio with 40% to 100% of the sleeve in ILS. The star marks the optimum.

After ten years, 9.2% of paths are still underfunded at the optimum, against 20.6% at the starting point. The reallocation more than halves it, and adds about half a point of growth a year.

**What the board's choices change.** The levers depend on the rules more than the result does. Across risk appetites from 10% to 20% (market-consistent) and 5% to 15% (statutory), growth stays between 1.46% and 1.56%, while the hedge ratio moves between 60% and 90%. Tighter limits than agreed leave no admissible combination. A budget of 10% is not enough to meet the limits; 20% raises growth to 1.65%.

## When: the answer depends on today's state

The right levers in a calm year are not the right levers in a crash. Run the same optimisation, once with the fund starting in a calm market and once starting in a crash, and the answers diverge:

| Today | Hedge ratio | Sleeve ILS / ETF | Underfunding risk, market / statutory | Growth p.a. |
| --- | --- | --- | --- | --- |
| Calm | 60% | 30 / 70 | 14.7% / 9.6% | 1.64% |
| Unconditional | 70% | 40 / 60 | 14.1% / 9.7% | 1.51% |
| Crash (least risk; no combination meets both limits) | 90% | 70 / 30 | 16.7% / 9.5% | 0.93% |

In a calm market the fund can hedge less and hold more of the ETF. In a crash it should hedge much more and move towards ILS, and even then it cannot meet both limits: without a reallocation it would sit at 45.5%. The unconditional answer is a compromise that fits neither state.

This only matters because states persist. If a crash said nothing about next year, today's state would say nothing about the levers.

## From the middle ground to dynamic hedging

The CRO Forum lists dynamic hedging among the future applications of AI. I have seen what that looks like up close. [AAACCELL](https://www.aaaccell.ch/), a friend's company I represented in Barcelona, built HedgePilot, an AI co-pilot for dynamic FX hedging. Underneath it sits a risk-factor model and a decision model, and a neural network is trained on both. It recommends when and how much to hedge, within the policy a company sets.

That architecture is the one this study follows: risk factors and states, then a decision rule, then an action. HedgePilot learns its decision model from data, within the policy it is given. This study writes the state and the rule down explicitly, across the whole balance sheet: the exposures, the liabilities, the longevity drift and the limits a board sets.

The two meet in the middle. The explicit layer defines what the learned layer optimises: which exposures exist, which states matter, and what "better" means within which limits. The learned layer does what no investment committee can: watch the state continuously and act on it consistently. The table above is what it would be steering.

**For the practitioner: the hard part is the state.** States are not observed; they are estimated, and persistence is the hardest parameter to calibrate. A modestly harsher calibration of the crash share (18% instead of 14%), the equity loss in a crash (-35% instead of -25%), the rate fall in a crash (-0.9 instead of -0.6 points) or the persistence of the longevity drift leaves no admissible combination at all. A milder one gives noticeably more room. Higher crash persistence moves the optimum to an 80% hedge and a 60/40 split. That is why the state has to be written down as judgement, and tested in ranges.

## Judgment

The CRO Forum is right that there is no universally optimal ALM framework. Once a board has set its risk appetite and budget, though, and once the state is written down, the answer is specific.

- **The state, not the tail.** Fatter tails for individual risks did not change the decision. A shared, persistent state did.
- **Economics steers, the statutory view bounds.** Growth is measured on the market-consistent funding ratio; the statutory limit is respected because underfunding there triggers recovery measures.
- **Persistence is a risk driver.** No one-year number sees it, and it moves the hedge ratio more than any other calibration choice.
- **Admissible paths become computable.** The corridor in the full model is narrow, 11 of 99 combinations, but it can be found.

The natural next step is a separate rate regime: rising rates with collateral calls on the swap overlay, the pattern behind Eurovita and the 2022 LDI crisis. That is where the liquidity objective of the CRO Forum paper and the time axis of this one meet.

The pension fund is only the example. Anyone who steers a balance sheet over several years, insurer, reinsurer or pension fund, can build on the CRO Forum's picture of today's practice. The next step is to add time, add the state, and look past the fatter tails.

## Caveats and sources

- All model parameters are hand-set and illustrative. The optimum depends on the limits and on the ETF's assumed return premium over broad equities (6% against 3.4%); without the premium, about half of the ETF's advantage over equities disappears.
- The longevity link is supported in mechanism, not in size: longevity changes cohort sizes, and cohort sizes drive sector demand. How much liability risk sector equities offset is an assumption, with basis risk between the fund's members, the population and global sectors.
- The statutory view is modelled as not recognising longevity within ten years; in practice mortality tables are updated every few years, so it sees longevity with a lag.
- The shared state and its persistence enter the study in one step.
- The Eurovita mechanism is taken from the CRO Forum paper; the HedgePilot architecture is described from my own experience with the company.

Sources: [CRO Forum, Balancing the Balance Sheet](https://thecroforum.org/balancing-the-balance-sheet/) (November 2026) · [DellaVigna and Pollet, Demographics and Industry Returns](https://econpapers.repec.org/RePEc:aea:aecrev:v:97:y:2007:i:5:p:1667-1702) (AER 2007) · [Friedberg and Webb, Life Is Cheap](https://econpapers.repec.org/RePEc:bpj:bejeap:v:7:y:2007:i:1:n:31) (2007) · [OECD, Health at a Glance 2025: health spending projections](https://www.oecd.org/en/publications/2025/11/health-at-a-glance-2025_a894f72e/full-report/health-spending-projections_d3702f1a.html) · [AAACCELL, HedgePilot](https://www.aaaccell.ch/) · [ALM as a System](https://www.augmentedactuary.com/2025/12/08/alm-system.html) · [Risk Capacity, Exposure, and Order](https://www.augmentedactuary.com/2026/01/15/Exposure-ALM.html)
