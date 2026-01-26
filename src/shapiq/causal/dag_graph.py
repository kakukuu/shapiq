"""DAG-based causal graph structure for do-Shapley computation.

This module provides a directed acyclic graph (DAG) representation for
causal structures, used by DoCausalImputer for computing do-Shapley values.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class DAGGraph:
    """Directed Acyclic Graph structure for causal do-interventions.

    This class represents a causal structure as a DAG with explicit directed
    edges. It is used by :class:`~shapiq.imputer.DoCausalImputer` to compute
    do-Shapley values based on interventional distributions.

    The causal structure is defined by:

    1. Directed edges: ``(parent, child)`` pairs indicating parent -> child
       causal relationships.

    2. Confounding pairs: Variable pairs ``(i, j)`` that share an unobserved
       common cause (represented as bidirectional edges i <-> j in the
       causal graph literature, meaning i <- U -> j for hidden confounder U).

    Attributes:
        edges: List of directed edges ``(parent, child)`` defining the DAG.
        confounding_pairs: Set of variable pairs with unobserved confounders.
        n_features: Total number of features in the graph.

    Example:
        >>> from shapiq.causal import DAGGraph
        >>> import numpy as np
        >>>
        >>> # X0 -> X1, X0 -> X2, X1 -> X2, with X0 <-> X2 confounding
        >>> dag = DAGGraph(
        ...     edges=[(0, 1), (0, 2), (1, 2)],
        ...     confounding_pairs={(0, 2)},
        ...     n_features=3,
        ... )
        >>>
        >>> # Get topological order
        >>> print(dag.topo_order)  # [0, 1, 2]
        >>>
        >>> # Get parents of a node
        >>> print(dag.get_parents(2))  # [0, 1]
        >>>
        >>> # Get predecessors (all ancestors in topo order)
        >>> print(dag.get_predecessors(2))  # [0, 1]

    See Also:
        :class:`~shapiq.imputer.DoCausalImputer`: Uses this class for do-Shapley.
        :class:`~shapiq.causal.OrderingGraph`: For Causal SHAP (ordering-based).

    References:
        Jung, Y., Tian, J., & Bareinboim, E. (2022).
        On Measuring Causal Contributions via do-interventions.
        ICML 2022.
    """

    edges: list[tuple[int, int]] = field(default_factory=list)
    """List of directed edges (parent, child) in the DAG."""

    confounding_pairs: set[tuple[int, int]] = field(default_factory=set)
    """Set of variable pairs (i, j) with unobserved common causes."""

    n_features: int | None = field(default=None)
    """Total number of features. Inferred from edges if not provided."""

    def __post_init__(self) -> None:
        """Validate and initialize the DAG structure."""
        # Convert confounding_pairs to set if needed
        if not isinstance(self.confounding_pairs, set):
            self.confounding_pairs = set(self.confounding_pairs)

        # Infer n_features from edges if not provided
        if self.n_features is None:
            if not self.edges:
                raise ValueError("Must provide either edges or n_features")
            max_node = max(max(p, c) for p, c in self.edges)
            self.n_features = max_node + 1

        # Validate edge indices
        for parent, child in self.edges:
            if parent < 0 or parent >= self.n_features:
                raise ValueError(
                    f"Edge ({parent}, {child}) has invalid parent index. "
                    f"n_features={self.n_features}"
                )
            if child < 0 or child >= self.n_features:
                raise ValueError(
                    f"Edge ({parent}, {child}) has invalid child index. "
                    f"n_features={self.n_features}"
                )

        # Build internal structures
        self._build_graph()

        # Validate DAG (no cycles)
        self._validate_dag()

    def _build_graph(self) -> None:
        """Build adjacency lists from edges."""
        assert self.n_features is not None  # Guaranteed by __post_init__
        self._children: dict[int, list[int]] = defaultdict(list)
        self._parents: dict[int, list[int]] = defaultdict(list)

        # Initialize all nodes
        for i in range(self.n_features):
            self._children[i] = []
            self._parents[i] = []

        # Add edges
        for parent, child in self.edges:
            self._children[parent].append(child)
            self._parents[child].append(parent)

    def _validate_dag(self) -> None:
        """Validate that the graph is a DAG (no cycles).

        Raises:
            ValueError: If the graph contains a cycle.
        """
        assert self.n_features is not None  # Guaranteed by __post_init__
        visited = [False] * self.n_features
        rec_stack = [False] * self.n_features

        def has_cycle_dfs(v: int) -> bool:
            visited[v] = True
            rec_stack[v] = True

            for child in self._children[v]:
                if not visited[child]:
                    if has_cycle_dfs(child):
                        return True
                elif rec_stack[child]:
                    return True

            rec_stack[v] = False
            return False

        for i in range(self.n_features):
            if not visited[i]:
                if has_cycle_dfs(i):
                    raise ValueError("Graph contains a cycle and is not a DAG!")

    @cached_property
    def topo_order(self) -> list[int]:
        """Compute topological ordering of variables (ancestors first).

        Returns:
            List of feature indices in topological order.
        """
        assert self.n_features is not None  # Guaranteed after __post_init__
        n_features = self.n_features  # Local variable for type narrowing
        visited = [False] * n_features
        result: list[int] = []

        def dfs(v: int) -> None:
            visited[v] = True
            for child in self._children[v]:
                if not visited[child]:
                    dfs(child)
            result.insert(0, v)

        for i in range(self.n_features):
            if not visited[i]:
                dfs(i)

        return result

    def get_parents(self, node: int) -> list[int]:
        """Get direct parents of a node.

        Args:
            node: The node index.

        Returns:
            List of parent node indices.
        """
        return self._parents.get(node, [])

    def get_children(self, node: int) -> list[int]:
        """Get direct children of a node.

        Args:
            node: The node index.

        Returns:
            List of child node indices.
        """
        return self._children.get(node, [])

    def get_predecessors(self, node: int) -> list[int]:
        """Get all predecessors (ancestors) of a node in topological order.

        Predecessors are all nodes that appear before this node in the
        topological ordering.

        Args:
            node: The node index.

        Returns:
            List of predecessor indices in topological order.
        """
        node_idx = self.topo_order.index(node)
        return self.topo_order[:node_idx]

    def sort_by_topo(self, nodes: list[int] | npt.NDArray[np.intp]) -> list[int]:
        """Sort a subset of nodes according to topological order.

        Args:
            nodes: Subset of node indices.

        Returns:
            Nodes sorted by topological order.
        """
        node_set = set(nodes)
        return [v for v in self.topo_order if v in node_set]

    def is_confounded(self, node1: int, node2: int) -> bool:
        """Check if two nodes have an unobserved confounder.

        Args:
            node1: First node index.
            node2: Second node index.

        Returns:
            True if the nodes share a hidden confounder.
        """
        pair = (min(node1, node2), max(node1, node2))
        return pair in self.confounding_pairs or (node2, node1) in self.confounding_pairs

    def get_ancestors(self, node: int) -> set[int]:
        """Get all ancestors (transitive parents) of a node.

        Args:
            node: The node index.

        Returns:
            Set of all ancestor node indices.
        """
        ancestors = set()
        stack = list(self._parents[node])

        while stack:
            parent = stack.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                stack.extend(self._parents[parent])

        return ancestors

    def get_descendants(self, node: int) -> set[int]:
        """Get all descendants (transitive children) of a node.

        Args:
            node: The node index.

        Returns:
            Set of all descendant node indices.
        """
        descendants = set()
        stack = list(self._children[node])

        while stack:
            child = stack.pop()
            if child not in descendants:
                descendants.add(child)
                stack.extend(self._children[child])

        return descendants

    @classmethod
    def from_ordering(
        cls,
        ordering: list[list[int]],
        confounding: list[bool] | None = None,
    ) -> "DAGGraph":
        """Create a DAGGraph from component-based ordering.

        This factory method converts the ordering format used by
        :class:`~shapiq.causal.OrderingGraph` into a full DAG.
        All variables in earlier components are parents of all
        variables in later components.

        Args:
            ordering: List of components, where each component is a list
                of feature indices. Earlier components are ancestors.
            confounding: List of booleans indicating confounding per component.
                If ``confounding[i]`` is True, all pairs within component i
                are marked as confounded.

        Returns:
            A new DAGGraph instance.

        Example:
            >>> dag = DAGGraph.from_ordering(
            ...     ordering=[[0], [1, 2], [3]],
            ...     confounding=[False, True, False],
            ... )
            >>> # Creates edges: 0->1, 0->2, 0->3, 1->3, 2->3
            >>> # Confounding pairs: {(1, 2)}
        """
        if confounding is None:
            confounding = [False] * len(ordering)

        edges = []
        confounding_pairs = set()

        # Create edges between components
        for i, component in enumerate(ordering):
            # Variables in later components are children
            for j in range(i + 1, len(ordering)):
                for parent in component:
                    for child in ordering[j]:
                        edges.append((parent, child))

            # Mark confounding pairs within component
            if confounding[i] and len(component) > 1:
                for idx1, v1 in enumerate(component):
                    for v2 in component[idx1 + 1:]:
                        confounding_pairs.add((min(v1, v2), max(v1, v2)))

        n_features = sum(len(comp) for comp in ordering)

        return cls(
            edges=edges,
            confounding_pairs=confounding_pairs,
            n_features=n_features,
        )

    def to_ordering(self) -> tuple[list[list[int]], list[bool]]:
        """Convert DAG to component-based ordering format.

        This produces a valid ordering compatible with
        :class:`~shapiq.causal.OrderingGraph`. Note that the conversion
        may lose some DAG structure information if the DAG cannot be
        perfectly represented by the ordering format.

        Returns:
            Tuple of (ordering, confounding) in OrderingGraph format.
        """
        # Use topological order, each node in its own component
        # (conservative conversion that preserves all ordering info)
        ordering = [[node] for node in self.topo_order]
        confounding = [False] * len(ordering)

        return ordering, confounding
