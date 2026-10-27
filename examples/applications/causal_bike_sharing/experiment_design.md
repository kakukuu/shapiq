# Experiment protocol

## Data and models

The daily Bike Sharing dataset contains 731 dates. The target is `cnt`; inputs are
`trend`, `cosyear`, `sinyear`, `temp`, `atemp`, `windspeed` and `hum`. Feature units
and transformations are documented in [data/README.md](data/README.md).

A and B use a random 80/20 split with seed 42: 584 training days and 147 test days.
One XGBoost model is fitted on the training split. All A/B variants share this model,
the full training background and the held-out inputs.

Lag replaces temperature, humidity and wind speed with previous-day readings and
drops the first date. The two designs share 584 training and 146 test target dates,
but fit separate models. Apparent temperature remains same-day. Differences combine
input and model changes and cannot be attributed entirely to causal-structure
changes. The random holdout is an attribution-design comparison, not an assessment
of prospective forecasting performance.

## Explanations and repeats

- Order-2 k-SII separates seven singleton terms from 21 pair interactions.
- A budget of 5,000 covers all 128 coalitions of the seven inputs.
- Causal imputation uses Gaussian copula sampling, with 500 draws per coalition by
  default. Marginal imputation uses the joint empirical training background.
- Discovery seed: 42. Fixed-model reference seeds: 7, 19, 73, 101, 202, 307, 401, 503.
  Case-validation seeds: 809, 907, 1009, 1103, 1201, 1301, 1409, 1511.
- Seeds control imputation draws. Model fits, row identities and background data
  remain fixed within each comparison; both arms are evaluated for each repeat.
- Observed coalition features retain their input values. The empty coalition uses
  the declared background prediction mean. Each decomposition must reconstruct its
  prediction within an absolute tolerance of `1e-4` rentals:
  `baseline + sum(singleton terms) + sum(pair interactions) = model prediction`.
  The tolerance accommodates floating-point regression residuals. Both new and
  cached results are checked; discovery-run residuals are exported to
  `results/tables/efficiency_checks.csv`.

## Comparison rationale

Hashmi et al. note that the definition of a value function includes an imputation
mechanism for features excluded from the coalition, and they use explanation
consistency metrics to compare attributions produced by different value functions.
We adopt this comparative approach: in our A/B experiments, we hold the predictive
model, background data, and explained instances fixed to examine how changes in
the imputation mechanism and its assumptions about causal structure affect
explanations.

We compare k-SII terms of the same order, treating singleton contributions and
second-order interactions as two separate comparison sets and explicitly
distinguishing rankings by signed values from rankings by absolute values. In
addition to ranking comparisons, we report changes in contribution values and
signs and examine individual instances to identify local differences that
aggregate statistics may obscure. The numerical change metrics, case selection
rules, and repeated-sampling checks used here are specific to the experimental
design of this study.

## Comparisons and selection

A compares causal and marginal imputations. B includes eight ordering/confounding
variants. An ordering change can alter multiple ancestor relations, conditioning
sets and joint sampling groups; it is not necessarily a single-edge intervention.

Each comparison retains population summaries, metrics for every date, all seven
singleton terms and 21 pair interactions per date, selected cases and independent
validation repeats. Networks label baseline values, modified values and differences
directly. Additional matrices and six-panel diagnostic views are disabled by default.

All differences are modified minus baseline. Reports distinguish signed mean
change, mean absolute change (`mean(abs(delta))`), and mean attribution-magnitude
change (`mean(abs(modified) - abs(baseline))`). First-order L1 sums absolute changes
across seven singleton terms; second-order L1 sums across 21 pairs. Signed and
absolute Kendall rank agreements are computed per date before averaging.

The six default [case-selection rules](README.md#case-selection) apply to declared
singleton terms or pairs. Singleton targets use the first five rules; pairs use
all six. Each target/rule retains its best eligible representative.
Dates selected more than once within a comparison are merged while preserving all
reasons, with no overall case limit. Rank conditions apply only to the target's
interaction order; material sign reversals require sufficiently large magnitudes
at both endpoints. Complete candidate records retain thresholds, their origins and
reasons for ineligibility.

Cases are selected using discovery values only and frozen before repeat validation.
Cached draws can be reused when their provenance matches. Unstable validation
results do not lead to case replacement. Whole-explanation L1 strong, typical and
weak cases remain supplementary diagnostics.

## Interpretation

These comparisons are exploratory and conditional on the fitted models and assumed
causal structures. Singleton k-SII terms differ from ordinary Shapley values; output
interaction-network edges represent attribution terms, while input graph edges
define imputation assumptions.

Paired row-bootstrap intervals describe test-composition variation conditional on
the computed explanations. They do not cover temporal dependence, imputation error
or all comparisons jointly. The 10th–90th percentile ranges across eight imputation
repeats describe computational variability, not population confidence intervals.
Four-metric rank calibration is configuration-specific. Sensitivity to an assumed
causal graph does not establish that graph's correctness.

Source, data, model, dependency and output checksums govern result reuse, including
statistical JSON outputs. Historical audit files are not runtime dependencies.
Existing case outputs are replaced only after model provenance and requested
selections have passed validation, with previous outputs archived before replacement.
