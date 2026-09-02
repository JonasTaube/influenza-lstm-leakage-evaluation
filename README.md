# Leakage-aware evaluation of LSTM adaptation strategies for influenza forecasting

Companion code for the manuscript:

> **Forecasting influenza positivity rates under distribution shift: a leakage-aware evaluation of LSTM adaptation strategies in city-wide sentinel surveillance, Tianjin, China, 2015–2025** (submitted to *International Journal of Infectious Diseases*)

This repository contains the full analysis pipeline used to evaluate static, online (walk-forward fine-tuned) and periodically retrained LSTM models against classical and machine-learning baselines for multi-week (t+1–t+4) forecasting of the combined weekly influenza positivity rate, using city-wide sentinel RT-PCR surveillance data from Tianjin, China (81 421 specimens, 574 weeks, 2015–2025).

## Key findings

- The best model was the **simplest**: a static LSTM receiving only the aggregated weekly positivity rate (MAE 0.0665, MASE 0.39, R 0.842), statistically indistinguishable from rolling-update ARIMA (Diebold–Mariano p = 0.53) and persistence.
- **Aggregate input significantly outperformed subtype-resolved input** (DM p = 0.023, significant in 4/5 seeds) — the only significant top-line comparison.
- Under **leakage-free evaluation, walk-forward online fine-tuning provided no benefit**: single-step update (p = 0.56), delayed multi-step update (harmful, R 0.512), experience replay (at best parity).
- **Periodic retraining was the only adaptation strategy that helped** (R 0.728 → 0.814, p = 0.020) but did not surpass the aggregate-input static model.
- A subtle **information-leakage flaw** (fine-tuning on the full 4-step target immediately after prediction) inflated R by ≈0.05 and reduced apparent MAE by ≈16%, manufacturing an artefactual "online advantage". The `leaky` comparator is retained to quantify this artefact.

## Data

The weekly surveillance data (`tianjin_weekly_influenza.csv`) are **not** included in this repository. The aggregated weekly data are available from the corresponding author on reasonable request, consistent with the manuscript's data availability statement. The analysis expects the data file at `data/tianjin_weekly_influenza.csv`.

`01_build_weekly_data.py` builds the weekly series from raw specimen records at `data/raw_specimen_records.csv`, which expects the following columns (identifying fields are excluded for privacy):

| Column | Description |
|---|---|
| `specimen_id` | Anonymised specimen identifier |
| `collect_date` | Collection date (`YYYY-MM-DD`) |
| `source` | Sentinel source (national ILI sentinel hospital) |
| `specimen_type` | Specimen type (throat swab) |
| `rt_a_universal`, `rt_a_h3`, `rt_a_h1` | Influenza A RT-PCR results (universal / H3N2 / H1N1pdm09) |
| `rt_b`, `rt_b_yamagata`, `rt_b_victoria` | Influenza B RT-PCR results (universal / Yamagata / Victoria) |
| `rt_covid19` | SARS-CoV-2 RT-PCR result |

Result values use `positive` / `negative` tokens.

## Dependencies

- Python ≥ 3.10
- PyTorch (CPU is sufficient; all models were trained on CPU)
- numpy, pandas, scipy
- statsmodels (ARIMA), scikit-learn (XGBoost), xgboost
- matplotlib (figures), python-docx (optional, for `09_md_to_docx.py`)

## Pipeline

| Script | Purpose |
|---|---|
| `01_build_weekly_data.py` | Build weekly subtype positivity-rate series from raw specimen records |
| `02_descriptive_stats.py` | Descriptive epidemiology tables (Supplementary Tables S1–S2) |
| `03_make_figures_1_3.py` | Descriptive time-series figures |
| `04_train_models.py` | **Core experiment**: train all 14 model configurations × 5 seeds; saves `seed_predictions.npz` |
| `05_make_figure1.py` | Architecture / walk-forward evaluation workflow diagram |
| `06_check_manuscript.py`, `06b/c/d_*.py` | Consistency checks (dates, abstract word count, placeholders) |
| `07_replay_variant.py` | Experience-replay online update variant |
| `08_final_extras.py` | Leakage-quantification experiments (leaky & replay variants); saves `extra_predictions.npz` |
| `08b_posthoc_fixed.py` | **Post-hoc analysis without retraining**: loads saved predictions, computes all metrics, DM tests, rewrites Tables 1–5 and Figures 2/4/6/7 |
| `09_md_to_docx.py` | Convert the manuscript markdown to submission-format .docx |

### Reproducing the results

```bash
# 1. place data at data/tianjin_weekly_influenza.csv
# 2. run the core experiment (CPU, ~30–60 min for 5 seeds × 14 configurations)
python scripts/04_train_models.py
# 3. leakage-quantification experiments
python scripts/07_replay_variant.py
python scripts/08_final_extras.py
# 4. tables + figures (loads saved predictions; fast)
python scripts/08b_posthoc_fixed.py
```

## Evaluation protocol notes

- Chronological 70/30 split: 391 training / 168 test sliding windows (12-week input, 4-week target). Test period: September 2022 – December 2025 (includes the post-pandemic resurgence).
- All metrics are seed-averaged over five random seeds (42, 123, 777, 2024, 8888); seed-42 used for visualisation only.
- **Leakage-free update schedules** (single-step, delayed multi-step, experience replay) never use target values that would be unobservable at update time in operational deployment. The deliberately flawed `leaky` schedule is marked with an asterisk in all tables.
- Diebold–Mariano tests on squared-error differentials with Newey–West HAC variance (lag 4), computed per seed and summarised across seeds.

## License

No license is specified; all rights reserved. Contact the corresponding author for reuse inquiries.
