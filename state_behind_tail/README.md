# The State Behind the Tail: delivery package (22 Nov 2026)

## Contents
- `scripts/pension_study.py`: pension fund study (scenario worlds, five-step ladder, joint optimisation of split and hedge ratio, "when", single-candidate, split and hedge-ratio views). All parameters and branding in `CONFIG`; override with `--config overrides.json`; `--dump-config` prints the effective settings. About 4 minutes at 200,000 paths.
- `scripts/pension_sensitivity.py`: ranges for the full model (risk appetite 3x3, budget 10/15/20%, state calibration low/high). Imports `pension_study.py` from the same folder. About 2 minutes at 100,000 paths.
- `scripts/decision_matrix.py`: earlier stylised two-asset model (decision matrix, Kelly curves).
- `results/`: figures, CSV files and run logs from the run of 22 Nov 2026 (seed 42).

## Requirements
Python 3 with numpy, scipy, matplotlib.

## Run
```
cd scripts
python pension_study.py
python pension_sensitivity.py
```

## Main results (full model: shared state, longevity, 10 years)
- Optimum: hedge ratio 70%, sleeve 40% ILS / 60% longevity ETF.
- Underfunding risk at any year-end: 14.1% market-consistent, 9.7% statutory (limits 15% / 10%).
- Underfunded at year 10: 9.2% (starting point 20.6%). Growth 1.51% a year (starting point 1.04%).
- Starting point breaches both limits in every 10-year step.
- Calm today: hedge 60%, 30/70. Crash today: no admissible combination; least risk hedge 90%, 70/30.

## Figures
- `pension_optimum_ladder.png` (main), `pension_optimum_map.png`, `pension_when.png`
- `pension_ranges_decision.png` (board), `pension_ranges_state.png` (practitioner)
- Backup: `pension_decision_ladder.png`, `pension_split.png`, `pension_hedge.png`

## Documents
- Draft article: https://claude.ai/artifact/FYX1aXUrzCxNTFNXkpu9RV
- Model specification: https://claude.ai/artifact/NgTuqdYWCPDytQ7fRgyJyu

All parameters are hand-set and illustrative.
