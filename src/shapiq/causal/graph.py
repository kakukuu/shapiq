"""Causal graph structures for Causal Shapley value computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class CausalGraph:
    """Causal graph structure defining partial ordering and confounding.

    This class represents the causal structure between features using a partial
    ordering (components) and confounding information. It is used by
    :class:`~shapiq.imputer.CausalImputer` to determine how to sample missing
    features respecting causal relationships.

    The causal structure is defined by:

    1. Ordering: A partial order over features, represented as a list of
       components. Features in earlier components are ancestors (causes) of
       features in later components. Features within the same component have
       no direct causal relationship with each other.

    2. Confounding: For each component, whether there exists an unobserved
       confounder among the variables in that component. This affects how we
       condition when sampling.

    Attributes:
        ordering: List of components, where each component is a list of feature
            indices. For example, ``[[0], [1, 2], [3, 4, 5]]`` means:
            - Feature 0 is a root cause (no parents)
            - Features 1, 2 are caused by feature 0
            - Features 3, 4, 5 are caused by features 0, 1, 2

        confounding: List of boolean values, one per component. If ``True`` for
            component ``i``, there is an unobserved confounder among variables
            in that component, so we should NOT condition on other intervened
            variables in the same component when sampling.

    Example:
        >>> from shapiq.causal import CausalGraph
        >>> import numpy as np
        >>>
        >>> # x0 -> x1,x2 -> x3,x4  with confounding between x1 and x2
        >>> graph = CausalGraph(
        ...     ordering=[[0], [1, 2], [3, 4]],
        ...     confounding=[False, True, False],
        ... )
        >>>
        >>> # Get conditioning set for component 2 when x0, x1 are intervened
        >>> intervened = np.array([True, True, False, False, False])
        >>> cond_set = graph.get_conditioning_set(component_idx=2, intervened=intervened)
        >>> print(cond_set)  # [0, 1, 2] - all ancestors

    See Also:
        :class:`~shapiq.imputer.CausalImputer`: Uses this class to define causal structure.
    """

    ordering: list[list[int]] = field(default_factory=list)
    """Partial ordering of features as list of components."""

    confounding: list[bool] = field(default_factory=list)
    """Whether each component has unobserved confounding."""

    n_features: int | None = field(default=None, repr=False)
    """Total number of features (computed from ordering if not provided)."""

    def __post_init__(self) -> None:
        """Validate and initialize the causal graph."""
        if not self.ordering and self.n_features is None:
            raise ValueError("Must provide either ordering or n_features")

        if not self.ordering:
            # Default: all features in one component
            # n_features is guaranteed to be int here due to the check above
            assert self.n_features is not None
            self.ordering = [list(range(self.n_features))]

        if not self.confounding:
            # Default: no confounding
            self.confounding = [False] * len(self.ordering)

        if len(self.confounding) != len(self.ordering):
            raise ValueError(
                f"confounding length ({len(self.confounding)}) must match "
                f"ordering length ({len(self.ordering)})"
            )

        # Compute n_features from ordering if not provided
        if self.n_features is None:
            self.n_features = sum(len(comp) for comp in self.ordering)

        # Compute n_components
        self.n_components = len(self.ordering)

    def get_conditioning_set(
        self,
        component_idx: int,
        intervened: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.intp]:
        """Get the conditioning set for sampling a component.

        When sampling missing features in a component, we condition on:
        1. All features in ancestor components (always)
        2. Intervened features in the same component (only if no confounding)

        The key insight from Causal SHAP is that with confounding, conditioning
        on other intervened variables in the same component would introduce
        spurious dependencies through the unobserved confounder.

        Args:
            component_idx: Index of the component to sample (0-indexed).
            intervened: Boolean array of shape ``(n_features,)`` indicating which
                features are intervened (True) or need to be sampled (False).

        Returns:
            Array of feature indices to condition on when sampling the component.

        Example:
            >>> graph = CausalGraph([[0], [1, 2], [3]], [False, True, False])
            >>> intervened = np.array([True, True, False, False])
            >>> # For component 1 with confounding, only condition on ancestors
            >>> graph.get_conditioning_set(1, intervened)
            array([0])
            >>> # For component 2 without confounding, include same-component intervened
            >>> graph.get_conditioning_set(2, intervened)
            array([0, 1, 2])
        """
        # All features in ancestor components
        ancestors = [v for comp in self.ordering[:component_idx] for v in comp]

        # If no confounding, also include intervened variables in same component
        if not self.confounding[component_idx]:
            same_comp_intervened = np.intersect1d(
                self.ordering[component_idx],
                np.where(intervened)[0],
            )
            ancestors = np.union1d(ancestors, same_comp_intervened)

        return np.array(ancestors, dtype=np.intp)