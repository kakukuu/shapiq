"""Shared data, explanation and summary utilities for the five experiments."""

from __future__ import annotations

from typing import Literal, cast

from example_paths import configure_imports

configure_imports()

import numpy as np
import pandas as pd
from bike_data import load_bike_sharing_daily
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from shapiq import TabularExplainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["trend", "cosyear", "sinyear", "temp", "atemp", "windspeed", "hum"]
TARGET = "cnt"

# Integer index mapping: trend=0, cosyear=1, sinyear=2, temp=3, atemp=4, windspeed=5, hum=6

ORDERING = [
    [0],  # Layer 0 (F) — trend: root cause
    [
        1,
        2,
        3,
        5,
        6,
    ],  # Layer 1 (T) — cosyear,sinyear,temp,windspeed,hum: confounded by atmospheric_state
    [4],  # Layer 2 (F) — atemp: mediator
]

CONFOUNDING = [False, True, False]

# Reset the imputer RNG for each explanation; seed=None retains its running state.
EXPLAIN_SEED = 42
ESTIMATOR_REPEAT_SEEDS = (7, 19, 73, 101, 202, 307, 401, 503)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_data():
    """Return the seven model inputs and the daily rental count."""
    X, y = load_bike_sharing_daily()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    return X[FEATURE_NAMES], y


def make_lag_data():
    """Return X_lag, y with temp/hum/windspeed shifted by t-1.

    Lag columns: temp(t-1), hum(t-1), windspeed(t-1); all others same-day.
    First row (NaN lag) is dropped.
    """
    X, y = load_bike_sharing_daily()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    X = X.copy()
    for col in ["temp", "hum", "windspeed"]:
        X[col] = X[col].shift(1)
    # Drop rows where any lag column is NaN
    X = X.dropna(subset=["temp"]).reset_index(drop=True)
    y = y.iloc[-len(X) :].reset_index(drop=True)
    return X[FEATURE_NAMES], y


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------


def get_explainer(
    X,
    y,
    ordering=None,
    confounding=None,
    imputer: Literal["causal", "marginal"] = "causal",
    seed=42,
    model_cls=None,
    model_kwargs=None,
):
    """Return (TabularExplainer, X_explain) with a trained XGBoost model.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    ordering : list[list[str]] | None
    confounding : list[bool] | None
    imputer : str — "causal" or "marginal"
    seed : int
    model_cls : estimator class (default XGBRegressor)
    model_kwargs : dict passed to model_cls

    Returns
    -------
    explainer : TabularExplainer
    X_explain : pd.DataFrame (50 rows held-out subset)
    """
    if ordering is None:
        ordering = ORDERING
    if confounding is None:
        confounding = CONFOUNDING
    if model_cls is None:
        model_cls = XGBRegressor
    if model_kwargs is None:
        model_kwargs = dict(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = model_cls(**model_kwargs)
    model.fit(X_train, y_train)

    # Graph parameters apply only to causal imputation.
    if imputer == "causal":
        explainer = TabularExplainer(
            model=model.predict,
            data=X_train.to_numpy(),
            index="k-SII",
            max_order=2,
            imputer="causal",
            random_state=seed,
            ordering=ordering,
            confounding=confounding,
        )
    else:
        explainer = TabularExplainer(
            model=model.predict,
            data=X_train.to_numpy(),
            index="k-SII",
            max_order=2,
            imputer="marginal",
            random_state=seed,
        )
    return explainer, X_test.head(50)


# ---------------------------------------------------------------------------
# Experiment ordering perturbations
# ---------------------------------------------------------------------------

# --- B1: Omitted confounder ---
ORDERING_B1 = ORDERING  # same ordering, different confounding
CONFOUNDING_B1 = [False, False, False]

# --- B2: Reversed direction (cosyear/sinyear → Layer 0) ---
ORDERING_B2 = [
    [0, 1, 2],  # trend, cosyear, sinyear — cosyear/sinyear moved to root layer
    [3, 5, 6],  # temp, windspeed, hum — Layer 1 (T)
    [4],  # atemp — Layer 2
]
CONFOUNDING_B2 = [False, True, False]

# --- B3: Misidentified mediator (atemp → Layer 1) ---
ORDERING_B3 = [
    [0],  # Layer 0 — trend
    [
        1,
        2,
        3,
        5,
        6,
        4,
    ],  # Layer 1 — all others including atemp (mediator misattributed as same-layer)
]
CONFOUNDING_B3 = [False, True]


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def explain_one(explainer, X_row, budget=2000, seed=EXPLAIN_SEED):
    """Explain one pandas or NumPy row with an optional per-row RNG reset.

    Resetting the imputer and approximator makes a row's explanation independent
    of its position in a batch. Pass ``seed=None`` to retain the running RNG state.
    """
    import warnings

    if seed is not None:
        explainer.set_random_state(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if isinstance(X_row, (pd.Series, pd.DataFrame)):
            X_row = X_row.to_numpy()
        X_row = np.asarray(X_row).reshape(1, -1)
        return explainer.explain(X_row, budget=budget)


def explain_multi(explainer, X, budget=2000, n=None, seed=EXPLAIN_SEED):
    """Explain up to n rows of X, returning list of InteractionValues.

    See ``explain_one`` for the meaning of ``seed``. It is applied before *every*
    row, which makes the batch reproducible and makes each row independent of the
    position it occupies in ``X``.
    """
    if n is None:
        n = len(X)
    results = []
    for i in range(min(n, len(X))):
        sii = explain_one(explainer, X.iloc[i], budget=budget, seed=seed)
        results.append(sii)
    return results


def estimator_repeats(explainer, rows, budget=5000, n=None, seeds=ESTIMATOR_REPEAT_SEEDS):
    """Repeat only explanation RNG; reuse model, background, graph and rows.

    These are conditional diagnostics anchored to the original seed-42 draw,
    not a distribution over retrained models. The row-wise RNG reset is retained.
    """
    if len(set(seeds)) != len(seeds) or EXPLAIN_SEED in seeds or len(seeds) < 2:
        raise ValueError("Use at least two distinct repeat seeds, excluding the anchor.")
    runs = []
    for seed in seeds:
        runs.append(explain_multi(explainer, rows, budget=budget, n=n, seed=seed))
        print(f"  Fixed-model estimator repeat seed={seed} complete.", flush=True)
    return runs


def average_sii(sii_list):
    """Average a list of InteractionValues.

    Returns a dict with keys: baseline, phi (1st-order ndarray),
    interactions (dict of 2nd-order), n_order (raw InteractionValues).
    InteractionValues.values is read-only, so we return a plain dict.
    """
    stacked_values = np.stack([s.values for s in sii_list])
    phi = stacked_values.mean(axis=0)
    baseline = sii_list[0].baseline_value if hasattr(sii_list[0], "baseline_value") else phi[0]

    # Average interactions
    interactions = {}
    if hasattr(sii_list[0], "interactions") and sii_list[0].interactions:
        all_keys = set()
        for s in sii_list:
            if hasattr(s, "interactions") and s.interactions:
                all_keys.update(s.interactions.keys())
        for key in all_keys:
            interactions[key] = float(
                np.mean(
                    [
                        s.interactions.get(key, 0.0)
                        for s in sii_list
                        if hasattr(s, "interactions") and s.interactions
                    ]
                )
            )

    return {
        "baseline": baseline,
        "phi": phi,
        "interactions": interactions,
        "n_order": sii_list[0],  # keep one element for get_n_order access
    }


def _scalar(result):
    """Extract the scalar statistic from a SciPy test result."""
    return float(cast(float, result[0]))


def compute_metrics(sii_base, sii_perturbed, feature_names=None):
    """Compare two SII results: Kendall τ, Spearman ρ, per-feature Δφ, interaction ΔI.

    Parameters
    ----------
    sii_base : InteractionValues or np.ndarray
    sii_perturbed : InteractionValues or np.ndarray
    feature_names : list[str] | None

    Returns
    -------
    dict with keys: tau, rho, delta_phi (dict), delta_I (dict), rank_flips
    """
    from scipy.stats import kendalltau, spearmanr

    if feature_names is None:
        feature_names = FEATURE_NAMES

    # Extract 1st-order values from get_n_order(1).dict_values
    def _extract_phi(sii):
        """Extract 1st-order SII values as a dict (index -> value)."""
        d = sii.get_n_order(1).dict_values
        # If empty (causal imputer malfunction), fall back to values array
        if not d:
            return np.zeros(len(feature_names))
        phi = np.zeros(len(feature_names))
        for (idx,), v in d.items():
            phi[idx] = v
        return phi

    phi_base = _extract_phi(sii_base)
    phi_pert = _extract_phi(sii_perturbed)

    # Per-feature Δφ
    delta_phi = {}
    for i, name in enumerate(feature_names):
        delta_phi[name] = float(phi_pert[i] - phi_base[i])

    # Kendall τ, Spearman ρ
    tau = _scalar(kendalltau(phi_base, phi_pert))
    rho = _scalar(spearmanr(phi_base, phi_pert))

    # Rank flips
    rank_base = np.argsort(np.argsort(-phi_base))
    rank_pert = np.argsort(np.argsort(-phi_pert))
    rank_flips = int(np.sum(rank_base != rank_pert))

    # 2nd-order interaction values
    delta_I = {}
    I_base = sii_base.interactions if hasattr(sii_base, "interactions") else None
    I_pert = sii_perturbed.interactions if hasattr(sii_perturbed, "interactions") else None
    if I_base is not None and I_pert is not None:
        n = len(feature_names)
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                v_base = I_base.get(key, 0.0)
                v_pert = I_pert.get(key, 0.0)
                delta_I[f"{feature_names[i]},{feature_names[j]}"] = float(v_pert - v_base)

    return {
        "tau": float(tau),
        "rho": float(rho),
        "delta_phi": delta_phi,
        "delta_I": delta_I,
        "rank_flips": rank_flips,
    }


def print_metrics(metrics):
    """Pretty-print the metrics dict."""
    print(f"Kendall τ = {metrics['tau']:.4f}")
    print(f"Spearman ρ = {metrics['rho']:.4f}")
    print(f"Rank flips = {metrics['rank_flips']}")
    print()
    print("Per-feature Δφ:")
    for name, dv in sorted(metrics["delta_phi"].items(), key=lambda x: -abs(x[1])):
        print(f"  {name:12s}: {dv:+.6f}")
    if metrics["delta_I"]:
        print()
        print("Top-5 interaction ΔI:")
        sorted_I = sorted(metrics["delta_I"].items(), key=lambda x: -abs(x[1]))
        for pair, dv in sorted_I[:5]:
            print(f"  {pair:20s}: {dv:+.6f}")


def rank_metrics(
    sii_base_list,
    sii_pert_list,
    feature_names=None,
    sii_null_list=None,
    n_boot=5000,
    seed=42,
    *,
    sii_null_runs=None,
    paired=True,
):
    """
    Compute rank-order metrics: 4-variant Kendall τ (1st & 2nd order, signed & abs).

    Compares rankings induced by two SII result lists (base vs perturbed).
    Optionally calibrates against fixed-model estimator repeats on matched rows.
    sii_null_runs is a list of repeat result lists. paired=False disables calibration.
    Difference intervals resample rows jointly and repeat seeds as a crossed factor.
    Single-list null support remains for callers, but does not cover seed variability.

    Parameters
    ----------
    sii_base_list : list[InteractionValues]
        Base explainer results (n_explain samples)
    sii_pert_list : list[InteractionValues]
        Perturbed explainer results (n_explain samples, same shape)
    feature_names : list[str] | None
        Feature names for labeling. Defaults to FEATURE_NAMES.
    sii_null_list : list[InteractionValues] | None
        Null/baseline explainer results (e.g., seed=7). If provided, compute
        null-model τ distribution in parallel.
    n_boot : int
        Number of bootstrap resamples for CI computation.
    seed : int
        RNG seed for bootstrap reproducibility.

    Returns
    -------
    dict with keys:
      - tau1_signed: dict with per_sample, mean, std, ci_lo, ci_hi
      - tau1_abs: dict with per_sample, mean, std, ci_lo, ci_hi
      - tau2_signed: dict with per_sample, mean, std, ci_lo, ci_hi
      - tau2_abs: dict with per_sample, mean, std, ci_lo, ci_hi
      - null: dict (only if sii_null_list provided) with tau1_signed, tau1_abs, tau2_signed, tau2_abs
      - rank_flips1: dict with per_sample, mean
    """
    from scipy.stats import kendalltau

    if feature_names is None:
        feature_names = FEATURE_NAMES

    n_explain = len(sii_base_list)
    n_feat = len(feature_names)
    if n_explain < 2 or len(sii_pert_list) != n_explain:
        raise ValueError("Rank metrics require equal lists with at least two rows.")
    if sii_null_list is not None and sii_null_runs is not None:
        raise ValueError("Provide a single null list or repeated null runs, not both.")
    if not paired and (sii_null_list is not None or sii_null_runs is not None):
        raise ValueError("Unmatched rows must not receive a paired null verdict.")

    # ------- Helper to extract 1st-order φ -------
    def _extract_phi(sii):
        """Extract 1st-order SII φ as np.ndarray of shape (n_feat,)."""
        d = sii.get_n_order(1).dict_values
        phi = np.zeros(n_feat)
        for (idx,), v in d.items():
            phi[idx] = v
        return phi

    # ------- Helper to extract 2nd-order interactions as 21-vector -------
    def _extract_I_vec(sii):
        """
        Extract 2nd-order interactions I_{ij} into a fixed-order 21-vector.
        Order: [(0,1), (0,2), ..., (0,6), (1,2), ..., (1,6), ..., (5,6)]
        Missing pairs padded with 0.
        """
        I_dict = sii.interactions if hasattr(sii, "interactions") else {}
        I_vec = np.zeros(n_feat * (n_feat - 1) // 2)
        idx = 0
        for a in range(n_feat):
            for b in range(a + 1, n_feat):
                I_vec[idx] = I_dict.get((a, b), 0.0)
                idx += 1
        return I_vec

    # ------- Extract per-sample 1st-order φ -------
    phi_base_list = np.array([_extract_phi(sii) for sii in sii_base_list])
    phi_pert_list = np.array([_extract_phi(sii) for sii in sii_pert_list])

    # ------- Extract per-sample 2nd-order I -------
    I_base_list = np.array([_extract_I_vec(sii) for sii in sii_base_list])
    I_pert_list = np.array([_extract_I_vec(sii) for sii in sii_pert_list])

    # ------- Compute per-sample τ (4 variants) -------
    tau1_signed_samples = np.zeros(n_explain)
    tau1_abs_samples = np.zeros(n_explain)
    tau2_signed_samples = np.zeros(n_explain)
    tau2_abs_samples = np.zeros(n_explain)
    rank_flips1_samples = np.zeros(n_explain, dtype=int)

    for i in range(n_explain):
        # 1st-order: signed
        tau = _scalar(kendalltau(phi_base_list[i], phi_pert_list[i]))
        tau1_signed_samples[i] = tau

        # 1st-order: absolute values
        tau = _scalar(kendalltau(np.abs(phi_base_list[i]), np.abs(phi_pert_list[i])))
        tau1_abs_samples[i] = tau

        # 2nd-order: signed
        tau = _scalar(kendalltau(I_base_list[i], I_pert_list[i]))
        tau2_signed_samples[i] = tau

        # 2nd-order: absolute values
        tau = _scalar(kendalltau(np.abs(I_base_list[i]), np.abs(I_pert_list[i])))
        tau2_abs_samples[i] = tau

        # Rank flips in 1st-order
        rank_base = np.argsort(np.argsort(-phi_base_list[i]))
        rank_pert = np.argsort(np.argsort(-phi_pert_list[i]))
        rank_flips1_samples[i] = int(np.sum(rank_base != rank_pert))

    # ------- Bootstrap CI for each τ variant -------
    rng = np.random.default_rng(seed)

    def _bootstrap_ci(samples, n_boot=5000):
        """Resample samples with replacement, compute CI."""
        n_samp = len(samples)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n_samp, size=n_samp)
            boot_means[b] = samples[idx].mean()
        ci_lo = np.percentile(boot_means, 2.5)
        ci_hi = np.percentile(boot_means, 97.5)
        return {
            "per_sample": samples.copy(),
            "mean": float(samples.mean()),
            "std": float(samples.std()),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
        }

    result = {
        "tau1_signed": _bootstrap_ci(tau1_signed_samples, n_boot),
        "tau1_abs": _bootstrap_ci(tau1_abs_samples, n_boot),
        "tau2_signed": _bootstrap_ci(tau2_signed_samples, n_boot),
        "tau2_abs": _bootstrap_ci(tau2_abs_samples, n_boot),
        "rank_flips1": {
            "per_sample": rank_flips1_samples.copy(),
            "mean": float(rank_flips1_samples.mean()),
        },
    }

    # Conditional calibration: paired rows and crossed resampling of repeat seeds.
    # The original observation is fixed at seed 42; uncertainty in that anchor and
    # model fitting is not covered. Four comparisons form one within-arm family.
    if sii_null_list is not None:
        sii_null_runs = [sii_null_list]
    if sii_null_runs is not None:
        if not sii_null_runs or any(len(run) != n_explain for run in sii_null_runs):
            raise ValueError("Every repeat must explain exactly the same rows.")
        repeats = [
            rank_metrics(sii_base_list, run, feature_names, n_boot=1, seed=seed)
            for run in sii_null_runs
        ]
        result["null"] = {}
        result["calibration"] = {
            "repeat_count": len(repeats),
            "anchor_seed": EXPLAIN_SEED,
            "family_size": 4,
            "decision_confidence": 0.9875,
            "scope": "conditional on fitted model, background and seed-42 anchor",
        }
        rng_pair = np.random.default_rng(seed)
        # Use identical row indices for observed and null statistics.
        row_idx = rng_pair.integers(0, n_explain, size=(n_boot, n_explain))
        run_idx = rng_pair.integers(0, len(repeats), size=(n_boot, len(repeats)))
        for key in TAU_VARIANTS:
            draws = np.array([rep[key]["per_sample"] for rep in repeats])
            if not np.isfinite(draws).all() or not np.isfinite(result[key]["per_sample"]).all():
                raise ValueError("Undefined Kendall tau: inspect constant/tied profiles.")
            null_boot = np.array(
                [draws[np.ix_(run_idx[b], row_idx[b])].mean() for b in range(n_boot)]
            )
            obs_boot = result[key]["per_sample"][row_idx].mean(axis=1)
            delta_boot = obs_boot - null_boot
            null_by_row = draws.mean(axis=0)
            result["null"][key] = {
                "per_sample": null_by_row,
                "per_seed": draws,
                "mean": float(draws.mean()),
                "std": float(null_by_row.std()),
                "ci_lo": float(np.percentile(null_boot, 2.5)),
                "ci_hi": float(np.percentile(null_boot, 97.5)),
            }
            result[key]["paired_difference"] = {
                "mean": float(result[key]["mean"] - draws.mean()),
                "ci_lo": float(np.percentile(delta_boot, 2.5)),
                "ci_hi": float(np.percentile(delta_boot, 97.5)),
                "family_ci_lo": float(np.percentile(delta_boot, 0.625)),
                "family_ci_hi": float(np.percentile(delta_boot, 99.375)),
            }
    result["paired"] = paired

    return result


TAU_VARIANTS = ("tau1_signed", "tau1_abs", "tau2_signed", "tau2_abs")


def tau_verdict(obs, null=None):
    """Classify paired obs-minus-null CI; never infer a test from CI overlap.

    Bonferroni percentile intervals target 95% family coverage over four
    within-arm rank variants. Bootstrap coverage is approximate.
    """
    diff = obs.get("paired_difference")
    if diff is None:
        return "descriptive"
    if diff["family_ci_hi"] < -1e-12:
        return "distortion"
    if diff["family_ci_lo"] > 1e-12:
        return "above_floor"
    return "indistinct"


_VERDICT_TEXT = {
    "distortion": "BELOW fixed-model repeat reference (paired adjusted CI < 0)",
    "above_floor": "ABOVE repeat reference; this does not establish rank equivalence",
    "indistinct": "UNRESOLVED: adjusted difference CI includes 0; not evidence of equivalence",
    "descriptive": "DESCRIPTIVE ONLY: no calibrated paired comparison",
}


def print_rank_metrics(rank_result, title, indent=2):
    """Print the four tau variants against the null floor with a directional verdict."""
    pad = " " * indent
    print("\n" + "=" * 80)
    print(f"{pad}{title}")
    print("=" * 80)

    for tau_name in TAU_VARIANTS:
        obs = rank_result[tau_name]
        order_lbl = "1st-order phi" if "1" in tau_name else "2nd-order I"
        sign_lbl = "signed" if "signed" in tau_name else "absolute"

        print(f"\n{pad}{order_lbl} ({sign_lbl}):")
        print(
            f"{pad}  obs  mean {obs['mean']:+.4f}  std {obs['std']:.4f}  "
            f"95% CI [{obs['ci_lo']:+.4f}, {obs['ci_hi']:+.4f}]"
        )

        null = rank_result.get("null", {}).get(tau_name)
        if null is None:
            continue

        print(
            f"{pad}  null mean {null['mean']:+.4f}  std {null['std']:.4f}  "
            f"95% CI [{null['ci_lo']:+.4f}, {null['ci_hi']:+.4f}]"
        )
        diff = obs.get("paired_difference")
        if diff is not None:
            print(
                f"{pad}  obs-null {diff['mean']:+.4f}  paired 95% CI "
                f"[{diff['ci_lo']:+.4f}, {diff['ci_hi']:+.4f}]"
            )
            print(
                f"{pad}  within-arm adjusted 98.75% CI "
                f"[{diff['family_ci_lo']:+.4f}, {diff['family_ci_hi']:+.4f}]"
            )
        print(f"{pad}  verdict: {_VERDICT_TEXT[tau_verdict(obs, null)]}")

    print("=" * 80)


def print_tau_baseline_check(rank_result, frozen_tau):
    """Assert tau1_signed still reproduces the frozen ``compute_metrics`` aggregate tau."""
    obs = rank_result["tau1_signed"]["mean"]
    diff = abs(obs - frozen_tau)
    print("\n" + "=" * 80)
    print("  BASELINE VERIFICATION (tau1_signed must reproduce the frozen number)")
    print("=" * 80)
    print(f"  tau1_signed (rank_metrics)  = {obs:.4f}")
    print(f"  tau         (compute_metrics) = {frozen_tau:.4f}")
    print(
        f"  difference                  = {diff:.6f}  -> {'MATCH' if diff < 1e-6 else 'MISMATCH'}"
    )
    print("=" * 80)

    if not np.isfinite(diff) or diff >= 1e-6:
        raise AssertionError("Primary tau changed: baseline parity failed.")


def print_rank_verdicts(result):
    """Print data-driven conclusions suitable for notebook summaries."""
    for key in TAU_VARIANTS:
        print(f"  {key}: {_VERDICT_TEXT[tau_verdict(result[key])]}")
    if "calibration" in result:
        print(
            "  Conditional on the fitted model/background and seed-42 anchor; "
            "not a claim about retraining or population causal correctness."
        )


def design_amplitude_audit(base, pert, feature_names=None):
    """Explicit post-hoc operationalization of the old underspecified threshold."""
    names = FEATURE_NAMES if feature_names is None else feature_names
    bp = np.array([[s.get_n_order(1).dict_values[(i,)] for i in range(len(names))] for s in base])
    pp = np.array([[s.get_n_order(1).dict_values[(i,)] for i in range(len(names))] for s in pert])
    shift = (pp - bp).mean(axis=0)
    j = int(np.argmax(np.abs(shift)))
    threshold = float(bp[:, j].std(ddof=1))
    passed = bool(abs(shift[j]) > threshold)
    print("  Historical amplitude criterion (post-hoc operationalization):")
    print(
        f"    max |mean delta phi| = {abs(shift[j]):.3f} ({names[j]}); "
        f"same-feature baseline row SD = {threshold:.3f}; met={passed}"
    )
    print("    This was not fully specified prospectively; do not call it preregistered.")
    return {
        "feature": names[j],
        "max_abs_mean_delta": float(abs(shift[j])),
        "baseline_row_sd": threshold,
        "met": passed,
    }
