"""Tabular Explainer class for the shapiq package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

from shapiq.explainer.base import Explainer
from shapiq.game_theory.indices import is_empty_value_the_baseline

from .configuration import setup_approximator
from .custom_types import ExplainerIndices

if TYPE_CHECKING:
    import numpy as np

    from shapiq.approximator.base import Approximator
    from shapiq.imputer.base import Imputer
    from shapiq.interaction_values import InteractionValues
    from shapiq.typing import Model


TabularExplainerApproximators = Literal["spex", "montecarlo", "svarm", "permutation", "regression"]
TabularExplainerImputers = Literal["marginal", "baseline", "conditional", "causal", "do_causal", "no_ml_causal"]
TabularExplainerIndices = ExplainerIndices


class TabularExplainer(Explainer):
    """The tabular explainer as the main interface for the shapiq package.

    The ``TabularExplainer`` class is the main interface for the ``shapiq`` package and tabular
    data. It can be used to explain the predictions of any model by estimating the Shapley
    interaction values.

    Attributes:
        index: Type of Shapley interaction index to use.
        data: A background data to use for the explainer.

    Properties:
        baseline_value: A baseline value of the explainer.

    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        *,
        class_index: int | None = None,
        imputer: Imputer | TabularExplainerImputers = "marginal",
        approximator: (
            Literal["auto"] | TabularExplainerApproximators | Approximator[TabularExplainerIndices]
        ) = "auto",
        index: TabularExplainerIndices = "k-SII",
        max_order: int = 2,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the TabularExplainer.

        Args:
            model: The model to be explained as a callable function expecting data points as input
                and returning 1-dimensional predictions.

            data: A background dataset to be used for imputation.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models.

            imputer: Either an :class:`~shapiq.games.imputer.Imputer` as implemented in the
                :mod:`~shapiq.games.imputer` module, or a literal string from
                ``["marginal", "baseline", "conditional", "causal", "do_causal", "no_ml_causal"]``.
                Defaults to ``"marginal"``, which initializes the default
                :class:`~shapiq.games.imputer.marginal_imputer.MarginalImputer` with its default
                parameters or as provided in ``kwargs``. Use ``"do_causal"`` for interventional
                do-Shapley values (requires ``dag_edges`` in ``kwargs``). Use
                ``"no_ml_causal"`` for model-free causal Shapley values (requires ``y``
                in ``kwargs``; treats Y as the final node in the causal graph).

            approximator: An :class:`~shapiq.approximator.Approximator` object to use for the
                explainer or a literal string from
                ``["auto", "spex", "montecarlo", "svarm", "permutation"]``. Defaults to ``"auto"``
                which will automatically choose the approximator based on the number of features and
                the desired index.
                    - for index ``"SV"``: :class:`~shapiq.approximator.KernelSHAP`
                    - for index ``"SII"`` or ``"k-SII"``: :class:`~shapiq.approximator.KernelSHAPIQ`
                    - for index ``"FSII"``: :class:`~shapiq.approximator.RegressionFSII`
                    - for index ``"FBII"``: :class:`~shapiq.approximator.RegressionFBII`
                    - for index ``"STII"``: :class:`~shapiq.approximator.SVARMIQ`

            index: The index to explain the model with. Defaults to ``"k-SII"`` which computes the
                k-Shapley Interaction Index. If ``max_order`` is set to 1, this corresponds to the
                Shapley value (``index="SV"``). Options are:
                    - ``"SV"``: Shapley value
                    - ``"k-SII"``: k-Shapley Interaction Index
                    - ``"FSII"``: Faithful Shapley Interaction Index
                    - ``"FBII"``: Faithful Banzhaf Interaction Index (becomes ``BV`` for order 1)
                    - ``"STII"``: Shapley Taylor Interaction Index
                    - ``"SII"``: Shapley Interaction Index

            max_order: The maximum interaction order to be computed. Defaults to ``2``. Set to
                ``1`` for no interactions (single feature importance).

            random_state: The random state to initialize Imputer and Approximator with. Defaults to
                ``None``.

            verbose: Whether to show a progress bar during the computation. Defaults to ``False``.

            **kwargs: Additional keyword-only arguments passed to the imputers implemented in
                :mod:`~shapiq.games.imputer`.
        """
        from shapiq.imputer import (
            BaselineImputer,
            CausalImputer,
            DoCausalImputer,
            GenerativeConditionalImputer,
            MarginalImputer,
            NoMLCausalImputer,
            TabPFNImputer,
        )

        super().__init__(model, data, class_index, index=index, max_order=max_order)

        # get class for self
        class_name = self.__class__.__name__
        if self._model_type == "tabpfn" and class_name == "TabularExplainer":
            warn(
                "You are using a TabPFN model with the ``shapiq.TabularExplainer`` directly. This "
                "is not recommended as it uses missing value imputation and not contextualization. "
                "Consider using the ``shapiq.TabPFNExplainer`` instead. For more information see "
                "the documentation and the example notebooks.",
                stacklevel=2,
            )

        if imputer == "marginal":
            self._imputer = MarginalImputer(
                self.predict,
                self._data,
                random_state=random_state,
                **kwargs,
            )
        elif imputer == "conditional":
            self._imputer = GenerativeConditionalImputer(
                self.predict,
                self._data,
                random_state=random_state,
                **kwargs,
            )
        elif imputer == "baseline":
            self._imputer = BaselineImputer(
                self.predict,
                self._data,
                random_state=random_state,
                **kwargs,
            )
        elif imputer == "causal":
            self._imputer = CausalImputer(
                self.predict,
                self._data,
                random_state=random_state,
                ordering=kwargs.pop("ordering", None),
                confounding=kwargs.pop("confounding", None),
                **kwargs,
            )
        elif imputer == "do_causal":
            self._imputer = DoCausalImputer(
                self.predict,
                self._data,
                dag_edges=kwargs.pop("dag_edges"),
                random_state=random_state,
                confounding_pairs=kwargs.pop("confounding_pairs", None),
                method=kwargs.pop("method", "dml"),
                **kwargs,
            )
        elif imputer == "no_ml_causal":
            _no_ml_imputer = NoMLCausalImputer(
                X=self._data,
                y=kwargs.pop("y"),
                ordering=kwargs.pop("ordering", None),
                confounding=kwargs.pop("confounding", None),
                random_state=random_state,
                **kwargs,
            )
            self._imputer = _no_ml_imputer.as_causal_imputer()
        elif isinstance(
            imputer,
            MarginalImputer | GenerativeConditionalImputer | BaselineImputer | TabPFNImputer | CausalImputer | DoCausalImputer,
        ):
            self._imputer = imputer
        else:
            msg = (
                f"Invalid imputer {imputer}. "
                f'Must be one of ["marginal", "baseline", "conditional", "causal", "do_causal", "no_ml_causal"], '
                f"or a valid Imputer object."
            )
            raise ValueError(msg)
        self._n_features: int = self._data.shape[1]
        self.imputer.verbose = verbose  # set the verbose flag for the imputer

        self._approximator = setup_approximator(
            approximator,
            self.index,
            self._max_order,
            self._n_features,
            random_state,
        )

    def explain_function(
        self,
        x: np.ndarray,
        budget: int,
        *,
        random_state: int | None = None,
    ) -> InteractionValues:
        """Explains the model's predictions.

        Args:
            x: The data point to explain as a 2-dimensional array with shape
                (1, n_features).

            budget: The budget to use for the approximation. It indicates how many coalitions are
                sampled, thus high values indicate more accurate approximations, but induce higher
                computational costs.

            random_state: The random state to re-initialize Imputer and Approximator with.
                Defaults to ``None``, which will not set a random state.

        Returns:
            An object of class :class:`~shapiq.interaction_values.InteractionValues` containing
            the computed interaction values.
        """
        self.set_random_state(random_state)

        # initialize the imputer with the explanation point
        self.imputer.fit(x)

        # explain
        interaction_values = self.approximator(budget=budget, game=self.imputer)
        interaction_values.baseline_value = self.baseline_value
        # Adjust the Baseline Value if the empty value is the baseline
        if is_empty_value_the_baseline(interaction_values.index):
            interaction_values[()] = interaction_values.baseline_value
        return interaction_values

    @property
    def baseline_value(self) -> float:
        """Returns the baseline value of the explainer."""
        return self.imputer.empty_prediction
