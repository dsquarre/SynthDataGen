# PLAN

Dated log of plans and decisions. Newest entry first.

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
