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
input and model changes; the random holdout is an attribution-design comparison.

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
  prediction within an absolute tolerance of `1e-4` rentals.

## Comparisons and selection

A compares causal and marginal imputations. B includes eight ordering/confounding
variants. An ordering change can alter multiple ancestor relations, conditioning
sets and joint sampling groups.

All differences are modified minus baseline. Reports distinguish signed mean
change, mean absolute change, and mean attribution-magnitude change. L1 changes sum
absolute differences within each interaction order. Signed and absolute Kendall
rank agreements are computed per date before averaging.

The default [case-selection rules](README.md#case-selection) apply to individual
declared singleton terms or pairs. Cases are selected using discovery values and
frozen before repeat validation. Cached draws can be reused when their provenance
matches. Validation does not change the selected dates. Whole-explanation L1
examples remain supplementary diagnostics.

## Interpretation

These comparisons are exploratory and conditional on the fitted models and assumed
causal structures. Singleton k-SII terms differ from ordinary Shapley values; output
interaction-network edges represent attribution terms, while input graph edges
define imputation assumptions.

Paired row-bootstrap intervals describe test-composition variation conditional on
the computed explanations. They do not cover temporal dependence, imputation error
or all comparisons jointly. Eight-seed repeat ranges describe computational
variability. Four-metric rank calibration is configuration-specific.

Source, data, model, dependency and output checksums govern result reuse. Existing
case outputs are replaced only after model provenance and requested selections
have passed validation.
