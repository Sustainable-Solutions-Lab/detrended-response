# Potential bugs and inconsistencies

This file is a running list of items that may merit review before (or during) scientific interpretation of results.
Some items may be intentional design choices, but are flagged because the intent is not explicit in code/docs.

## 1) Persistence pre-history baseline mismatch between fitting vs saved `h(T)` values

**What I observed**

- In the *persistence/decay fitting code* for **Approach DJ** (and similarly **Approach DP**), the pre-history correction term is built assuming the temperature prior to each country's first observation is constant at a **country-specific linear-fit baseline evaluated at that country’s first year**:
  - `T_linear_first = compute_T_linear_at_first_year(data)`
  - `correction_T = compute_pre_first_year_correction(data, h4, T_linear_first)`
  - This is used in the regressors `X1`, `X2` during estimation.

- In contrast, when the pipeline saves observation-level `h_T` values to `bootstrap_h_values.csv` (used by the cumulative-effects scripts), the *same* persistence models (**DJ/DP/DL**) currently construct the pre-history correction assuming a **LOESS temperature baseline at the global base year (1961)**:
  - `T_loess_base = _get_T_loess_at_base_year(..., base_year=1961)`
  - `correction_T = compute_pre_first_year_correction(data, h4, T_loess_base)`

**Why it matters**

For persistence models, the pre-history temperature assumption sets the initial condition for the convolved response and appears in `X1`, `X2`. Using a different baseline when reconstructing/saving `h(T)` than the one used during estimation means:

- The stored `h_T` series is not exactly the model-implied `h_conv(T)` under the same initial-condition convention used in fitting.
- Any downstream quantities computed from stored `h_T` (notably cumulative effects) may shift slightly relative to what would be obtained using the fitting convention.

**Possible resolutions (choose explicitly)**

1. **Consistency with fitting (recommended for internal consistency):** when saving `h_T` values for DJ/DP, use the same pre-history baseline as fitting (e.g., `T_linear_first` for DJ/DP).
2. **Consistency with cumulative-effects baseline (comparability across approaches):** re-fit DJ/DP (and possibly DL) using the LOESS-1961 baseline convention, so estimation and reporting conventions match.
3. **Document as intentional:** keep as-is, but explicitly state in the methods that stored `h_T` values for persistence models are re-based using a 1961 LOESS baseline for cumulative-effects comparability.

---

## 2) Approach DL pre-history convention differs across code paths (may be intentional)

In the DL fitting routine, the pre-first-year correction is computed using the **observed** temperature series (`T`) and separately using the LOESS trend series (`T_trend`).
In bootstrap/output paths (for saving `h_T`), the correction uses a **LOESS 1961 baseline** array.

This is closely related to item (1), but worth noting separately because DL is used as a core “central” approach in figures/tables.

---

## 3) Housekeeping: legacy output artifacts after refactors

After refactors that change saved columns (e.g., removing the imbalance metrics), older output directories may contain CSV/Excel files with columns that no longer exist in the code. This is not a scientific issue, but can cause confusion when comparing runs.

Recommended: keep each run in a timestamped output folder (already done) and rely on the run’s own `run_metadata.json` and generated tables.
