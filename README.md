# Leakage-aware LSTM forecasting of influenza positivity, Tianjin, China, 2015–2025

Analysis code for the manuscript *"A leakage-aware comparison of static and
online LSTM forecasting of influenza positivity under distribution shift in
Tianjin, China"* (Scientific Reports, submission
476f10d0-e255-4f6a-93e9-711e7176fdb7).

This repository contains the complete analysis pipeline: data construction,
model training (14 configurations × 5 seeds), leakage-free walk-forward
evaluation, statistical testing, and manuscript figure generation. The two
public datasets (81,421 de-identified specimen records and the 574-week
aggregated modelling series) are distributed in the companion Zenodo deposit:

**https://doi.org/10.5281/zenodo.23590335**

## Contents

```
code/                                     analysis pipeline (Python 3)
  01_build_weekly_data.py                 original weekly construction (reads the raw
                                          LIS export with direct identifiers - NOT released)
  01b_build_weekly_from_deidentified.py   rebuild the weekly series from the released
                                          specimen file (validates n_samples/n_any exactly)
  02b_descriptive_stats_iso.py            descriptive tables S1/S2 (ISO-year aggregation)
  04_train_models.py                      core training: 14 configurations x 5 seeds,
                                          boundary-corrected split (388 training windows)
  07_replay_variant.py                    experience-replay variant
  08_final_extras.py                      leaky comparator + replay re-runs
  08b_posthoc_fixed.py                    post-hoc tables 1-5
  10_revision_extras.py                   reviewer-requested analyses S3-S10,
                                          restructured DM Table 5, revision_analyses.json
  11_generate_figures_rev.py              manuscript figures (corrected seed/smoothing/scaling)
  12_combine_figures_rev.py               composite Figures 2-4
  13_boundary_timeline.py                 Supplementary Figure S1 (boundary audit timeline)
  14_r2_table5_reconcile.py               [Revision 2] reconciled Table 5: MAE differences
                                          and bootstrap CIs computed as the mean over seeds
                                          of per-seed paired differences (now identical by
                                          construction to Table 1); DM test on the
                                          seed-averaged error differential; Holm correction
  15_r2_figure2a_fix.py                   [Revision 2] Figure 2 layout fix: legends moved
                                          outside the plotting areas in both panels
  genshin_style.py                        figure style helpers
  run_all.py / run_resume.py              one-command runners
  requirements.txt
```

## Reproducing the analysis

```bash
# 1. download the data from the Zenodo deposit and place data/ next to code/
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
cd code
python 01b_build_weekly_from_deidentified.py        # optional self-check: rebuilds the
                                                    # weekly series from the public specimen
                                                    # file (n_samples/n_any exact)
python run_all.py                                   # full pipeline: 02b -> 04 -> 08 -> 08b
                                                    # -> 10 -> 11 -> 12 -> 14 -> 15
```

Outputs (tables, figures, logs) are written to `tables/`, `figures/`, `logs/`
next to `code/`. The full pipeline runs on a modern desktop CPU in roughly
2–3 hours (model training dominates; no GPU is required). Fixed random seeds
make results reproducible; the manuscript reports mean ± SD across the five
seeds.

Note: `01_build_weekly_data.py` documents the original construction from the
laboratory information system export, which contains direct identifiers and
therefore cannot be released. `01b_build_weekly_from_deidentified.py`
rebuilds the weekly series from the public specimen file: weekly sample
counts and any-positive counts match the released series exactly (574 weeks,
81,421 specimens, 15,406 any-positive); subtype-specific counts differ only
where multi-subtype (mixed) infections occur, because the public file
records them as a single token while the raw data counts each RT-PCR channel
independently (the difference is quantified when 01b runs and affects well
under 0.5% of subtype detections). The released `tianjin_weekly_influenza.csv`
is the authoritative modelling input and is included in the Zenodo deposit.

## Revision 2 changes

Two scripts were added in Revision 2 (both included in the Zenodo deposit
version of this pipeline):

- **`14_r2_table5_reconcile.py`** — In the first revised version, the MAE
  differences in Table 5 were computed on seed-averaged *predictions* (an
  implicit 5-seed ensemble), so they did not equal the differences between
  the seed-averaged MAEs in Table 1. All seed-level quantities are now
  averaged across seeds *first* and summarised afterwards: the MAE difference
  is the mean over seeds of per-seed paired MAE differences (identical by
  construction to the difference between the Table 1 means), the bootstrap
  confidence interval is computed on the seed-averaged per-window
  absolute-error differences, and the aggregate Diebold–Mariano test is
  applied to the seed-averaged error differential. The per-seed DM tests are
  unchanged and reproduce the previous output exactly.
- **`15_r2_figure2a_fix.py`** — Legends in Figure 2 panels (a) and (b) are
  moved outside the plotting areas, eliminating text/legend overlap with the
  curves and bars.

## De-identification

All direct identifiers in the source laboratory information system (patient
name, medical record number, home address, telephone number, sending
physician, hospital department) were removed before export. No free-text
fields are included. See the Zenodo deposit (`data/README_data.md`) for the
full statement, inclusion criteria and ethics approval information.

## License

- Code: MIT
- Data (Zenodo deposit): Creative Commons Attribution 4.0 International
  (CC-BY 4.0)

## How to cite

Yu T, Wu X, Han N, Feng S, Fan J, Li J, Guo C, Lv Z, Fang Z, Li X, Yu L.
Analysis code and de-identified influenza sentinel surveillance data for
leakage-aware LSTM forecasting, Tianjin, China, 2015-2025. Zenodo.
https://doi.org/10.5281/zenodo.23590335
