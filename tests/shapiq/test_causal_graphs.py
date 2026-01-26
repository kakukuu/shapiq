"""Tests for DAGGraph and OrderingGraph causal graph structures."""

import numpy as np
import pytest

from shapiq.causal import DAGGraph, OrderingGraph, CausalGraph


class TestDAGGraph:
    """Tests for DAGGraph class."""

    def test_basic_initialization(self):
        """Test basic DAGGraph initialization."""
        edges = [(0, 1), (0, 2), (1, 2)]
        dag = DAGGraph(edges=edges, n_features=3)

        assert dag.n_features == 3
        assert dag.edges == edges
        assert len(dag.confounding_pairs) == 0

    def test_infer_n_features(self):
        """Test that n_features is inferred from edges."""
        edges = [(0, 1), (1, 2)]
        dag = DAGGraph(edges=edges)

        assert dag.n_features == 3  # max(0,1,1,2) + 1

    def test_confounding_pairs(self):
        """Test confounding pairs handling."""
        edges = [(0, 1), (1, 2)]
        confounding = {(0, 2)}
        dag = DAGGraph(edges=edges, confounding_pairs=confounding, n_features=3)

        assert dag.is_confounded(0, 2)
        assert dag.is_confounded(2, 0)  # Order doesn't matter
        assert not dag.is_confounded(0, 1)

    def test_topological_order(self):
        """Test topological ordering."""
        edges = [(0, 1), (0, 2), (1, 2)]
        dag = DAGGraph(edges=edges, n_features=3)

        topo = dag.topo_order
        # 0 must come before 1 and 2
        assert topo.index(0) < topo.index(1)
        assert topo.index(0) < topo.index(2)
        # 1 must come before 2
        assert topo.index(1) < topo.index(2)

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        # Create a cycle: 0 -> 1 -> 2 -> 0
        edges = [(0, 1), (1, 2), (2, 0)]

        with pytest.raises(ValueError, match="cycle|DAG"):
            DAGGraph(edges=edges, n_features=3)

    def test_invalid_edge_index(self):
        """Test that invalid edge indices are caught."""
        edges = [(0, 1), (1, 10)]  # Feature 10 doesn't exist

        with pytest.raises(ValueError, match="invalid.*index"):
            DAGGraph(edges=edges, n_features=3)

    def test_get_parents(self):
        """Test get_parents method."""
        edges = [(0, 2), (1, 2)]  # Both 0 and 1 are parents of 2
        dag = DAGGraph(edges=edges, n_features=3)

        assert set(dag.get_parents(2)) == {0, 1}
        assert dag.get_parents(0) == []
        assert dag.get_parents(1) == []

    def test_get_children(self):
        """Test get_children method."""
        edges = [(0, 1), (0, 2)]
        dag = DAGGraph(edges=edges, n_features=3)

        assert set(dag.get_children(0)) == {1, 2}
        assert dag.get_children(1) == []

    def test_get_predecessors(self):
        """Test get_predecessors method."""
        edges = [(0, 1), (1, 2)]
        dag = DAGGraph(edges=edges, n_features=3)

        # Predecessors of 2 should be all nodes before it in topo order
        preds = dag.get_predecessors(2)
        assert 0 in preds
        assert 1 in preds
        assert 2 not in preds

    def test_sort_by_topo(self):
        """Test sort_by_topo method."""
        edges = [(0, 1), (1, 2)]
        dag = DAGGraph(edges=edges, n_features=3)

        # Sort [2, 0] should give [0, 2]
        sorted_nodes = dag.sort_by_topo([2, 0])
        assert sorted_nodes[0] == 0
        assert sorted_nodes[1] == 2

    def test_get_ancestors(self):
        """Test get_ancestors method."""
        edges = [(0, 1), (1, 2), (0, 2)]
        dag = DAGGraph(edges=edges, n_features=3)

        # Ancestors of 2 are 0 and 1
        assert dag.get_ancestors(2) == {0, 1}
        # Ancestors of 1 is just 0
        assert dag.get_ancestors(1) == {0}
        # 0 has no ancestors
        assert dag.get_ancestors(0) == set()

    def test_get_descendants(self):
        """Test get_descendants method."""
        edges = [(0, 1), (1, 2), (0, 2)]
        dag = DAGGraph(edges=edges, n_features=3)

        # Descendants of 0 are 1 and 2
        assert dag.get_descendants(0) == {1, 2}
        # Descendants of 1 is just 2
        assert dag.get_descendants(1) == {2}
        # 2 has no descendants
        assert dag.get_descendants(2) == set()

    def test_from_ordering(self):
        """Test creating DAGGraph from ordering format."""
        ordering = [[0], [1, 2], [3]]
        confounding = [False, True, False]

        dag = DAGGraph.from_ordering(ordering, confounding)

        # Check edges exist
        assert dag.n_features == 4

        # All nodes in earlier components should be parents of later ones
        # So 0 -> 1, 0 -> 2, 0 -> 3, 1 -> 3, 2 -> 3
        assert 0 in dag.get_parents(1)
        assert 0 in dag.get_parents(2)
        assert 0 in dag.get_parents(3)
        assert 1 in dag.get_parents(3)
        assert 2 in dag.get_parents(3)

        # Confounding in component 1 means (1, 2) are confounded
        assert dag.is_confounded(1, 2)

    def test_to_ordering(self):
        """Test converting DAGGraph back to ordering format."""
        edges = [(0, 1), (1, 2)]
        dag = DAGGraph(edges=edges, n_features=3)

        ordering, confounding = dag.to_ordering()

        # Should have 3 components (conservative conversion)
        assert len(ordering) == 3
        assert len(confounding) == 3


class TestOrderingGraph:
    """Tests for OrderingGraph class (renamed from CausalGraph)."""

    def test_basic_initialization(self):
        """Test basic OrderingGraph initialization."""
        ordering = [[0], [1, 2], [3]]
        confounding = [False, True, False]

        graph = OrderingGraph(ordering=ordering, confounding=confounding)

        assert graph.n_features == 4
        assert graph.n_components == 3
        assert graph.ordering == ordering
        assert graph.confounding == confounding

    def test_default_confounding(self):
        """Test default confounding is all False."""
        ordering = [[0], [1, 2]]
        graph = OrderingGraph(ordering=ordering)

        assert graph.confounding == [False, False]

    def test_get_conditioning_set_no_confounding(self):
        """Test conditioning set without confounding."""
        ordering = [[0], [1, 2], [3]]
        confounding = [False, False, False]
        graph = OrderingGraph(ordering=ordering, confounding=confounding)

        # Intervened on features 0 and 1
        intervened = np.array([True, True, False, False])

        # For component 2 (feature 3), should condition on all ancestors + same component intervened
        cond_set = graph.get_conditioning_set(2, intervened)
        # Should include 0, 1, 2 (ancestors) - note 2 is not intervened so not included from same component
        assert 0 in cond_set
        assert 1 in cond_set
        assert 2 in cond_set

    def test_get_conditioning_set_with_confounding(self):
        """Test conditioning set with confounding."""
        ordering = [[0], [1, 2], [3]]
        confounding = [False, True, False]  # Confounding in component 1
        graph = OrderingGraph(ordering=ordering, confounding=confounding)

        # Intervened on feature 1
        intervened = np.array([False, True, False, False])

        # For component 1 with confounding, should NOT condition on same-component intervened
        cond_set = graph.get_conditioning_set(1, intervened)
        # Should only include ancestors (component 0 = feature 0)
        assert 0 in cond_set
        assert 1 not in cond_set  # Not included because confounding


class TestBackwardCompatibility:
    """Test backward compatibility of CausalGraph alias."""

    def test_causal_graph_alias(self):
        """Test that CausalGraph is an alias for OrderingGraph."""
        assert CausalGraph is OrderingGraph

    def test_causal_graph_works(self):
        """Test that using CausalGraph still works."""
        ordering = [[0], [1, 2]]
        graph = CausalGraph(ordering=ordering)

        assert graph.n_features == 3
        assert isinstance(graph, OrderingGraph)
