# Telling Cause from Correlation on Corr2Cause

A neuro-symbolic pipeline that decides whether a causal claim follows from a set of correlation and independence
facts written in text. It uses the [Corr2Cause](https://huggingface.co/datasets/causal-nlp/corr2cause) dataset
(ICLR 2024).

**How it works:** a regex parser reads each premise, and pgmpy's PC algorithm turns it into a causal graph.
Classical models (Logistic Regression, Random Forest, Gradient Boosting) then predict which claims are valid. The
predicted cause arrows are built into a DAG and a pgmpy Bayesian network, which answers intervention ("what if")
questions.

**Phase 2** adds a fine-tuned **DistilBERT** that reads the claims, so the pipeline still works when the text is
reworded or the variables are renamed. It compares this with a TF-IDF reader and with an end-to-end DistilBERT
that reads everything and uses no graph.

## Phase 2 results (deep learning)

Test F1 on the original test set and on the authors' two harder versions of it (same 21 premises):

| Method | Original | Reworded claims | Renamed variables |
|---|---|---|---|
| Regex reader + PC (Phase 1) | 0.952 | 0.000 | 0.948 |
| TF-IDF + Logistic Regression reader + PC | 0.952 | 0.536 | 0.948 |
| **DistilBERT reader + PC** | **0.952** | **0.948** | **0.948** |
| End-to-end DistilBERT (no graph) | 0.528 | 0.388 | 0.294 |

* **Reading the claim:** DistilBERT reads 100% of the reworded claims (83–91% with other random seeds). The
  regex reads none, and TF-IDF swaps every collider and confounder, because those two wordings use the same words
  in a different order.
* **No fluent shortcut:** the end-to-end DistilBERT calls about 40% of claims valid when only 15% are. Its 95% CI
  never overlaps the graph pipeline's (e.g. reworded: 0.27–0.47 vs 0.91–0.98).
* **Graphs from reworded text:** edge precision 0.89, recall 1.00, and 28 / 32 real intervention effects correct.
  The Phase 1 regex finds no edges there, so it gets 0 / 32.

## Phase 1 results (test set)

| What we measured | Result |
|---|---|
| Symbolic rule (parser + PC, no learning): share of labels matched | **98.5%** (F1 0.952) |
| Random Forest, chosen on validation: precision / recall / F1 | 0.927 / 0.989 / **0.957** |
| 95% bootstrap confidence interval for F1 (resampling premises) | 0.921 – 0.982 |
| Validation F1: TF-IDF only vs TF-IDF + structured features | 0.47 vs 0.88 |
| Edge precision of the predicted DAGs (against the PC graph) | 0.89 (1.00) |
| Intervention queries correct on the 5 fully known test graphs | **92 / 96**, including 28 / 32 real effects |

For comparison, an empty graph or a reversed graph gets **0 of the 32** real effects right.

## Pipeline

| Step | Notebook | Owner | What it does |
|---|---|---|---|
| 1 | `01_preprocessing_eda` | Chaitanya Sai Meka ([@ChaitanyaSai-Meka](https://github.com/ChaitanyaSai-Meka)) | Cleaning, leakage removal, subsampling train, EDA |
| 2 | `02_parser_pc` | Rakshita ([@Rakshita-0023](https://github.com/Rakshita-0023)) | Regex parser, schema checks, PC algorithm, equivalence class, `mec_support` |
| 3 | `03_feature_engineering` | Dhanvin Vadlamudi ([@Dhanvin1520](https://github.com/Dhanvin1520)) | TF-IDF, 15 structured features, chi-squared / mutual information, ablation |
| 4 | `04_models_evaluation` | Kasula Lalithendra ([@Lalith0024](https://github.com/Lalith0024)) | GridSearchCV with GroupKFold, model selection, confusion matrix, bootstrap CI, error analysis |
| 5 | `05_dag_bayesian_network` | Jagruthi Pulumati ([@Jag2007](https://github.com/Jag2007)) | DAG building, edge precision, Bayesian network, intervention queries |

**Phase 2**

| Step | Notebook | Owner | What it does |
|---|---|---|---|
| 6 | `06_robustness_test_sets` | Chaitanya Sai Meka | Load the reworded and renamed test sets, rename variables to A–F, leakage check, EDA |
| 7 | `07_claim_reader` | Rakshita | Regex vs TF-IDF vs fine-tuned DistilBERT reader, P/R/F1, review of unsure reads, seed check |
| 8 | `08_end_to_end_transformer` | Dhanvin Vadlamudi | End-to-end DistilBERT on premise + hypothesis, no graph |
| 9 | `09_comparison` | Kasula Lalithendra | Every method on all three test sets, bootstrap CIs, error analysis |
| 10 | `10_dag_on_reworded_text` | Jagruthi Pulumati | DAGs and intervention queries from reworded text, final results table |

## Worked example: ice cream and drowning

Ice cream sales and drownings rise together, but hot weather causes both. Test premise 2360 has exactly this shape
(A = hot weather, B = ice cream, C = drowning):

> A correlates with B. A correlates with C. B correlates with C. However, B and C are independent given A.

| Step | What happens |
|---|---|
| 01 Clean | The premise gets an ID and stays only in the test set. It comes with 18 claims. |
| 02 Parse and PC | PC links heat to ice cream and heat to drowning, with **no link between ice cream and drowning**. Three DAGs fit these facts. |
| 03 Features | "B directly causes C" (ice cream causes drowning) becomes 256 numbers: no arrow either way, independent given A, `mec_support` = 0. |
| 04 Predict | The Random Forest says **not valid** (2.8% probability of valid). Correct. |
| 05 What-if | With no path from ice cream to drowning, forcing ice cream sales up does not change drowning. |

The model gets all 18 claims about this premise right. Even "A is a common cause of B and C" is correctly marked not
valid: it holds in only 1 of the 3 fitting DAGs (`mec_support` = 0.33), so correlations alone cannot prove it.

## Features and models

Notebook 02 passes on each claim's text plus its parsed variables, claim type and PC graph. Notebook 03 turns these
into **256 features**: 241 TF-IDF terms from the text (135 premise + 106 hypothesis) and 15 structured features from
the parser and the graph, including `mec_support`, the share of fitting DAGs in which the claim is true.

| Features (logistic regression, validation) | F1 |
|---|---|
| TF-IDF only | 0.469 |
| Structured only | 0.846 |
| TF-IDF + structured | 0.884 |
| Symbolic rule: valid if `mec_support` = 1 (no learning) | 0.906 |

We compare one model from each classic family, and a no-learning rule as the baseline:

| Model | Why | Validation F1 | Test F1 |
|---|---|---|---|
| Logistic Regression | Linear baseline; its weights show which features matter | 0.878 | 0.952 |
| **Random Forest** (final) | Bagged trees; learns sharp cutoffs and feature interactions | **0.929** | **0.957** |
| Gradient Boosting | Boosted trees; often strongest on tabular data | 0.879 | 0.957 |
| Symbolic rule | Measures what learning adds on top of the graph step | 0.906 | 0.952 |

The models predict the **label** (1 = the claim is valid, 0 = it is not). `mec_support` is an input feature, not the
output. The Random Forest beats the rule by only 0.005, inside the confidence interval, so the graph step does most of
the causal reasoning.

## Getting started

Requires Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks in `notebooks/` in order (01 → 10). Notebooks 01 and 06 download the data from Hugging Face.
Each notebook writes its output as CSV files to `data/processed/` for the next notebook, and saves its figures to
`report/figures/`. Notebooks 01–05 take about 4 minutes and 06, 07, 09 and 10 a few minutes each.

> **Notebook 08 needs a GPU.** It fine-tunes DistilBERT on 35,340 premise and hypothesis pairs, which took 31
> minutes on a free Google Colab T4 GPU (our results). On an 8 GB laptop it runs out of memory.

> **Note:** the processed CSVs (about 140 MB) and the raw data are not stored in git. Running the notebooks in order
> rebuilds them.

Run the unit tests:

```bash
python -m pytest
```

## Repository layout

```
notebooks/        01 → 05 (Phase 1) and 06 → 10 (Phase 2), one per team member in each phase
src/c2c/          shared code used by the notebooks
  parsing.py      regex parser and schema checks        (notebook 02)
  graphs.py       PC, equivalence class, claim rules    (notebook 02)
  features.py     structured features                   (notebook 03)
  bayes.py        DAG building, intervention effects    (notebook 05)
  reader.py       rename variables, claim wordings      (notebooks 06, 07)
  transformer.py  fine-tune DistilBERT and predict      (notebooks 07, 08)
tests/            pytest unit tests for every module
report/           IEEE reports (report = Phase 1, report_phase2 = Phase 2), figures, phase 1 and phase 2 slides
data/             raw/ and processed/ CSVs, rebuilt by the notebooks (not in git)
```

## Evaluation protocol

* **Train:** fits everything (TF-IDF, scaler, models). It is subsampled to 400 whole premises (seed 42).
* **Validation:** used to choose the final model.
* **Test:** used once, only to report the final results.
* **Phase 2 readers** learn only from sentence wordings (`src/c2c/reader.py`), never from labels. None of our
  wordings use the words of the authors' reworded test, and the last wording of each claim type is held out for
  validation.

Premises shared between splits are removed, so there is no data leakage. Cross-validation (GroupKFold) and the
bootstrap both group rows by premise.

## Report

The write-ups are in [`report/report.pdf`](report/report.pdf) (Phase 1) and
[`report/report_phase2.pdf`](report/report_phase2.pdf) (Phase 2). To rebuild them from the LaTeX source, run
`tectonic report.tex` and `tectonic report_phase2.tex` inside `report/`. The slides are in
[`report/phase1_presentation.pptx`](report/phase1_presentation.pptx) and
[`report/phase2_presentation.pptx`](report/phase2_presentation.pptx).
