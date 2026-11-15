# PLAN

Dated log of plans and decisions. Newest entry first.

---

## 2026-11-14 (evening) — Journal manuscript; two input errors found in E0

**Manuscript.** Target: Springer journal 41062 (believed to be *Innovative Infrastructure Solutions*; guidelines page
not reachable, Springer Nature defaults used). LaTeX, sn-jnl template. Content rewritten from the ICSRI-2026 draft
with the current results (56 RACFST records; ungated E0; ratio study; pre-registered E1 on FRP and rectangular
CFST; circular CFST exploratory). Only the abstract was published at ICSRI-2026; the journal version must cite it.

**Errors found while writing the methods (affect E0 and all earlier RACFST runs, not E1):**
1. Ductility (mu) and damping (xi), both measured outputs of the cyclic test, were used as inputs when predicting
   Pu. A designer would not know them before testing.
2. The reconstruction formula `w_c=12/13` indexes the post-drop column list, where 12/13 is fy/n, not
   Water/Cement (10/5). Every run added a mislabelled "w/c" column equal to fy/n.

**E0-fixed (descriptive, not pre-registered):** E0 rerun with mu and xi dropped and w_c = Water/Cement, at 1:1 and
11:1 (`runs/2026-11-14_E0fixed_*`). These runs replace the ungated E0 numbers in the paper and deck.

**Superseded the same evening (user's request):** the interactive prompts and `input.txt` (column indices, index-based
formulas) were a shortcut. Replaced by a one-time preparation step (`scripts/prepare_racfst.py` ->
`datasets/racfst_cyclic.xlsx`, named columns; mu, xi, the source ratio columns and the all-zero RFA column removed)
and a JSON run configuration (`configs/racfst.json`) that names the target, class column and derived ratios
(w_c = water/cement, D_t = D/t, L_D = (2L/2)/D, i.e. L = half the reported 2L, at the user's request). Formulas and physical-bound constraints now refer to columns by name.
In fold-local CV, derived ratios are now rebuilt after generation (before, they were computed first, fed to the
generators and bounded like measured columns), which matches the global path. Imputation and generation stay inside
each fold. The partial E0-fixed runs were stopped and deleted; E0 is rerun as `runs/2026-11-14_E0clean_*`.
Source-data note: the A3 "L/D" column mixes 2L/D and L/D conventions (one specimen matches neither), so it is not used.

**E0-clean result (2026-11-15, `runs/2026-11-14_E0clean_*`):** real-only MAE 4.97-6.60 kN (R2 0.72-0.88; lower than
before, since mu and xi are no longer inputs). 1:1 lowered mean MAE in 4 of 30 settings (all XGBoost; CART with KNN
and MI imputation, TVAE with iterative and KNN), by 1-9% (max 0.55 kN), but the augmented model won only 5-8 of 15
folds (smallest two-sided paired Wilcoxon p = 0.39, uncorrected, descriptive). The other 26 rose by 1-60%. 11:1 worse
in 30 of 30. Paper and deck (v4) rebuilt from these runs; wording changed from "never lowered" to "not consistently".

**Series structure (found 2026-11-14 night; user chose: report as a limitation only).** The 56 RACFST tests form 7
series (section, D, t). Quick real-only check: random 5x3 CV MAE about 5 kN, leave-one-series-out 33-39 kN. E0
therefore measures prediction within known series. Reported in the paper's limitations with numbers from
`scripts/racfst_series_check.py`; no grouped-CV augmentation analysis. The L/D definition changes real-only MAE by
at most about 0.3 kN (2L/D and (2L/2)/D give identical tree-model predictions).

**Open question for E1 (not yet decided):** the FRP inputs include ecu (ultimate axial strain) and eh,rup (hoop
rupture strain), which are also measured at failure. The registered E1 used them; both arms (real-only and
augmented) had the same inputs, so the verdicts answer the registered question, but the task is not a pre-test
design prediction. A sensitivity run without these two columns would be a new analysis and should be registered
(public commit) before it runs.

---

## 2026-11-14 — Remove the statistical gate; training-size experiment on new datasets

### Decision: no statistical gate (made 2026-11-14)
Batch-level metric thresholds (Marginal, corr, cov, propensity) are removed from synthetic data generation for every
generator. Reasons: the gated metrics were also the reported metrics (circular), thresholds differed by generator
(unfair comparison), and an unreachable threshold could stall a generator indefinitely. Physical validity is still
enforced row by row (training-data min/max per numeric column plus user constraints) for every generator, and
rows that violate it are discarded and regenerated.

### E0 — Rerun of the original RACFST analysis without the gate (descriptive, not pre-registered)
Same pipeline and settings as the 2026-11-14 gated runs (56 records, 3 imputations, 5 generators, repeated 5-fold
CV x 3, XGBoost and Random Forest, real-only baseline), at 1:1 and 11:1 augmentation. Purpose: make every figure in
the deck come from the same ungated method. No decision rule; results are reported as they come out.

### E1 — Does augmentation help at small training sizes? (pre-registered; see rule below)

**Datasets**
- D1: FRP-confined concrete, Ozbakkaloglu & Lim (2013), 832 specimens (`datasets/frp_confined_concrete.xlsx`).
  Target: f'cc (MPa). Categorical input: confinement type.
- D2: Rectangular CFST, fy >= 460 MPa, concentric, from Thai et al. (2021) database, 255 tests
  (`datasets/cfst_rectangular_hss.xlsx`). Target: N_Test (kN). Categorical input: fc type.
- Not used in E1: circular HSS CFST (93 tests); too few rows for a fixed test set plus several training sizes.

**Design**
- 20 draws per dataset (draw r uses seed 2026 + r). In each draw, a random 25% of rows is the test set; the rest is
  the training pool, shuffled once. Training sets are nested prefixes of the shuffled pool.
- Training sizes n: D1 {25, 50, 100, 200, 400}; D2 {25, 50, 100, 150}.
- Within each training set: iterative imputation (numeric) and Random Forest completion (categorical), fitted on
  the training set only and applied to the test set. Physical bounds = training-set min/max per numeric column.
- Generators: Sequential CART, Gaussian Copula, CTGAN, CopulaGAN, TVAE, library default settings. Each generates
  n synthetic rows (1:1) from the imputed training set only. No statistical gate.
- Regressors: Random Forest (400 trees) and XGBoost (350 trees, depth 4, learning rate 0.05, subsample 0.9,
  colsample 0.9), same settings as the main pipeline. Each is trained on (a) real training rows only and
  (b) real + synthetic rows, then evaluated on the same test set. Target untransformed.
- Primary metric: test MAE. Primary outcome per draw: relative change d = (MAE_augmented - MAE_real) / MAE_real.

**Decision rule (applied separately to each dataset x size x generator x regressor)**
- Benefit: median d over the 20 draws <= -0.05 AND one-sided Wilcoxon signed-rank test (MAE_augmented <
  MAE_real) p < 0.05 after Holm correction within the dataset (family: all sizes x generators x regressors of
  that dataset; 50 tests for D1, 40 for D2).
- Harm: median d >= +0.05 AND the opposite one-sided Wilcoxon test p < 0.05 after the same Holm correction.
- Otherwise: no detectable effect.

**Main claim**
"Synthetic augmentation helps in the small-data regime" is supported only if the same generator shows Benefit at
n <= 50 on both D1 and D2 for at least one regressor. Otherwise the conclusion is that E1 found no evidence of
benefit.

**Generation failures:** if a generator cannot produce n valid rows in a draw, that draw is excluded for that
generator only (the Wilcoxon test then uses the remaining paired draws), and failure counts are reported.

**Secondary, descriptive only (no decision rule):** R² and RMSE; how d changes with n.

**Deviations:** any change to datasets, sizes, models, metrics or the rule after registration will be reported as
a deviation, with the reason.

**Registration:** this entry must be committed and pushed to a public repository (or posted on OSF) before E1
runs. The E1 script was tested only on synthetic toy data before registration.

**Registered:** commit 0c3b2b0, public repository github.com/dsquarre/SynthDataGen, pushed 2026-11-14 11:01:39 UTC
(GitHub push record). E1 had not been run on any real dataset at that time.

**Deviation (added after registration, 2026-11-14):** at the user's request, E1 is also run on the circular HSS
CFST subset (93 tests, `datasets/cfst_circular_hss.xlsx`, target N_Test_kN), sizes {25, 50}, otherwise the same
design. This dataset was excluded in the registered plan, so its results are exploratory: they are reported
separately and do not count towards the main claim.

**Deviation (code fix during the run, 2026-11-14 19:25 IST):** the D1 run stopped at draw 14 (n = 25) with a
crash in the shared imputation helper (`metrics._decode_categorical_with_encoders`): a confinement type present
in the test set but absent from the 25 training rows was imputed to a category code below -1, which the decoder
rejected. The fix treats every negative imputed code as missing, which is how codes in [-1.5, 0) were already
handled, so it changes only the case that previously crashed; draws 1-13 and all other runs are unaffected. D1
was resumed from draw 14 with the fixed code. The design and decision rule are unchanged.

**Result (2026-11-14, `runs/2026-11-14_E1/`):** main claim not supported. Registered cells: 0 Benefit, 45 Harm,
45 no detectable effect out of 90 (D1: 30 Harm / 20 none; D2: 15 Harm / 25 none). No generation failures.
Exploratory circular CFST: 0 Benefit, 12 Harm, 8 none. Summaries: `e1_registered_summary.csv`,
`e1x_cfst_circ_exploratory_summary.csv`; figure `e1_figure.png`.
