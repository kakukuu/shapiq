# Causal attribution on daily bike-sharing data

This branch is a compatibility preview based on upstream commit `2a8a4536`.
The causal compatibility workflow builds and tests the code on Linux; the complete
experiments must be rerun before reporting results from this version.

Five notebooks compare second-order k-SII explanations of daily rental predictions:

| Notebook | Comparison |
|---|---|
| [Experiment A](experiment_a.ipynb) | Causal and joint empirical marginal imputation for one fitted model |
| [Experiment B1](experiment_b1.ipynb) | Disable the weather-layer confounding flag |
| [Experiment B2](experiment_b2.ipynb) | Move seasonal codes to the root layer, with additional structural comparisons |
| [Experiment B3](experiment_b3.ipynb) | Merge apparent temperature into the weather layer, with a confounding-flag comparison |
| [Lag experiment](experiment_lag.ipynb) | Same-day versus previous-day weather inputs on matched target dates |

## Setup and execution

Use Python 3.12 or newer. From the repository root, create a virtual environment,
activate it, and install the checkout and example dependencies:

Installing this checkout compiles upstream C++ extensions. On Windows, install
Microsoft C++ Build Tools with the MSVC compiler and Windows SDK first. On Linux,
use a C++17 compiler with OpenMP support. See the upstream contribution guide for
the complete development setup.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r examples/applications/causal_bike_sharing/requirements.txt
python -m jupyterlab examples/applications/causal_bike_sharing
```

Select that environment as the notebook kernel, then **Restart Kernel and Run All**.
Run notebooks sequentially: they share computation and output directories. The
first execution computes all full-holdout comparisons and validates that notebook's
selected cases; subsequent notebooks reuse matching results. Computation can take
tens of minutes or longer depending on the machine and sampling settings.

The seven-feature daily dataset is bundled under [data/](data/README.md). The notebooks
use this checkout's library, or an installed `shapiq` when run outside a checkout.
Executed results are generated locally; no historical output is needed to run them.
Results must be recomputed with this library version: older library sampling behavior
and dependency versions can produce different numerical values.

## Configuration

The setup and case-selection cells expose these settings:

| Setting | Default | Meaning |
|---|---|---|
| `RUN_MODE` | `"auto"` | Compute missing/stale results; `"load"` requires valid saved results; `"recompute"` rebuilds reports while reusing verified draws |
| `CAUSAL_SAMPLE_SIZE` | `500` | Causal draws per coalition |
| `MARGINAL_SAMPLE_SIZE` | `"all"` | Use the complete training background |
| `CASE_TARGETS` | Per experiment | Singleton terms and pairs to inspect |
| `CASE_SELECTION_CONFIG` | Defined in the notebook | Magnitude, rank and quantile thresholds |
| `CASE_SCAN_ALL` | `False` | Scan all seven singleton terms and 21 pairs |
| `SHOW_FULL_TABLES` | `False` | Expand complete tables |
| `SHOW_DIAGNOSTICS` | `False` | Show supplementary distributions and whole-explanation L1 cases |
| `INSPECT_ROW_ID` / `INSPECT_ARM` | Per notebook | Inspect a specific held-out observation |

Use the same sampling settings across the five notebooks. Every case includes a
three-panel baseline/modified/difference network and a separate eight-seed repeat
plot. The figures show raw inputs and predictions alongside the attribution values.

## Case selection

For each declared target, select representative dates for: largest absolute change,
material sign reversal, stable same-order ranks with a large change, median change,
opposite-direction change, and (for pairs) opposing pair/singleton changes. The
magnitude threshold defaults to `max(25, 0.1 * median(abs(baseline target)))` rentals.
The rank criterion uses signed Kendall tau of at least 0.90 and the target's 75th
percentile of absolute changes.

One date is retained per eligible target/criterion; ties use the lowest source row
ID. Duplicate dates are merged with all selection reasons. Selections are frozen
before eight imputation repeats and are retained regardless of repeat stability.
Targets and thresholds are exploratory display choices. See the
[experiment protocol](experiment_design.md) for statistical interpretation.

## Files and reproducibility

- [experiment_support/](experiment_support/) contains the shared computation and plotting code.
- [tests/](tests/) contains regression checks, including data relocation and failed validation.
- `results/` stores computed tables, attribution arrays, model fingerprints and target cases.
- `exports/` stores notebook figure and table exports.
- `archive/` stores superseded local runs when computation settings change.

Generated directories, runtime requests and notebook execution outputs are ignored
by Git. Manifests record source/data checksums, model and split fingerprints,
sampling settings, seeds and dependency versions. Copying the checkout to a new
directory preserves path-independent provenance; changed code or data invalidates
saved results.

## Tests

From the repository root:

```sh
python -m pytest examples/applications/causal_bike_sharing/tests
```

The default tests use small fixtures and do not recompute full experiments. After a
complete run, set `SHAPIQ_VERIFY_SAVED_RESULTS=1` to enable saved-result checks.
To execute all five notebooks in fresh kernels and save rendered copies under the
ignored runtime directory:

```sh
python examples/applications/causal_bike_sharing/tests/execute_notebooks.py
```

The code follows the repository's MIT license. The bundled dataset is separately
licensed under CC BY 4.0; see [data attribution](data/README.md).
