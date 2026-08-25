"""Regression tests for Mesh.merge_coincident_nodes.

In particular, a point shared by three or more parts must collapse to a single
survivor — a single greedy pass would merge only one pair of the cluster and
leave the rest split.
"""

import numpy as np
import pytest

import fedoo as fd


def test_merge_pair():
    nodes = np.array([[0, 0, 0], [1e-9, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    elements = np.array([[0, 2, 3], [1, 3, 2]])
    m = fd.Mesh(nodes, elements, "tri3")
    merged = m.merge_coincident_nodes(tol=1e-6)
    assert merged == 1
    assert m.n_nodes == 3
    # The two coincident corners now reference the same surviving node.
    assert m.elements[0, 0] == m.elements[1, 0]


def test_merge_triple_point_collapses_to_single_node():
    """Three parts meeting at one point must all share a single node."""
    eps = 1e-9
    nodes = np.array(
        [
            [0, 0, 0],
            [eps, 0, 0],
            [0, eps, 0],  # three near-coincident corners
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 0],
            [2, 1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
        ],
        dtype=float,
    )
    elements = np.array([[0, 3, 4], [1, 5, 6], [2, 7, 8]])
    m = fd.Mesh(nodes, elements, "tri3")

    merged = m.merge_coincident_nodes(tol=1e-6)

    assert merged == 2  # two pairs collapse the 3-node cluster to 1
    assert m.n_nodes == 7  # 9 - 2
    shared = m.elements[:, 0]
    assert (
        len(set(shared.tolist())) == 1
    ), f"triple point not fully merged: corner ids {shared}"


def test_merge_noop_when_no_coincident_nodes():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    elements = np.array([[0, 1, 2]])
    m = fd.Mesh(nodes, elements, "tri3")
    assert m.merge_coincident_nodes(tol=1e-6) == 0
    assert m.n_nodes == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
