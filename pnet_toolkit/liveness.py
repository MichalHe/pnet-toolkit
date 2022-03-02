from collections import defaultdict
import math
from typing import (
    Dict,
    List,
    Set,
)

from pnet_toolkit.data import PetriNet

import numpy as np



def build_adjacency_matrix_from_pnet(pnet: PetriNet) -> np.array:
    """Construct the adjacency matrix: columns are transitions and rows are places."""

    # Collect arcs for individual transitions
    transition_arcs: Dict[str, List[PNArc]] = defaultdict(list)
    for arc in pnet.arcs:
        transition = arc.source if arc.source in pnet.transitions else arc.target
        transition_arcs[transition].append(arc)

    # Initialize the matrix
    adj_mat = np.zeros((len(pnet.places), len(pnet.transitions)), dtype=np.int64)

    # Assign indices for transitions and places
    transition_indices: Dict[str, int] = dict((transition, i) for i, transition in enumerate(sorted(pnet.transitions)))
    place_indices: Dict[str, int] = dict((place, i) for i, place in enumerate(sorted(pnet.places)))

    # Go over the transitions and populate the matrix
    for transition in pnet.transitions:
        transition_index = transition_indices[transition]
        for arc in transition_arcs[transition]:
            # Determine the mark flow
            if transition == arc.source:
                sign = 1
                place_index = place_indices[arc.target]
            else:
                # The transition is arc target, therefore we are taking away marks from the place
                sign = -1
                place_index = place_indices[arc.source]

            adj_mat[place_index][transition_index] = sign * arc.weight

    return adj_mat



def compute_p_invariants(adj_matrix: np.array):
    """Implementation of Farkas algorithm for computing P invariants."""
    # Input matrix - rows=places, columns=transitions
    place_count, transition_count = adj_matrix.shape

    # Construct a juxtaposition of (adj_matrix | E)
    jux_matrix = np.zeros((place_count, transition_count + place_count), dtype=np.int64)
    for row_i, row in enumerate(adj_matrix):
        for col_i, value in enumerate(row):
            jux_matrix[row_i][col_i] = value

        jux_matrix[row_i][transition_count + row_i] = 1

    row_count = place_count
    for column_i in range(transition_count):
        for ri in range(row_count):
            for rj in range(ri + 1, row_count):
                val_i = jux_matrix[ri][column_i]
                val_j = jux_matrix[rj][column_i]
                if (val_i >= 0 and val_j >= 0) or ((val_i <= 0 and val_j <= 0)):
                    # Process only values with different signs
                    continue

                new_row = abs(val_j) * jux_matrix[ri] + abs(val_i) * jux_matrix[rj]
                gcd = np.gcd.reduce(new_row)
                new_row = np.floor_divide(new_row, gcd)

                jux_matrix = np.vstack([jux_matrix, new_row])
                row_count += 1

        nonzero_row_indices = []
        for ri in range(row_count):
            if jux_matrix[ri][column_i] != 0:
                nonzero_row_indices.append(ri)
                row_count -= 1

        jux_matrix = np.delete(jux_matrix, tuple(nonzero_row_indices), axis=0)

    return jux_matrix[:, transition_count:transition_count + place_count]


def compute_liveness_coin_values(pn: PetriNet, verify_result: bool = False):
    """
    Generate the frobenius coin problem coin values that determine
    the liveness of the given petri net.
    """

    adj_matrix = build_adjacency_matrix_from_pnet(pn)
    p_invariants = compute_p_invariants(adj_matrix)

    if not p_invariants.any():
        return None

    # Select the smallest invariant
    minimal_invariant = p_invariants[0]
    minimal_invariant_norm = np.linalg.norm(minimal_invariant)

    for invariant in p_invariants:
        norm = np.linalg.norm(invariant)
        if norm < minimal_invariant_norm:
            minimal_invariant = invariant
            minimal_invariant_norm = norm

    if verify_result:
        transition_count = adj_matrix.shape[1]
        expected = np.zeros(transition_count, dtype=np.int64)
        actual = np.dot(minimal_invariant, adj_matrix)

        assert (actual == expected).all()

    return minimal_invariant


def reduce_invariant_to_coprimes(invariant: List[int]) -> List[int]:
    """Filter out those invariant elements that are not coprimes."""
    sorted_invariant = sorted(invariant)
    reduced_invariant: Set[int] = set()
    for inv in sorted_invariant:
        for rinv in reduced_invariant:
            gcd = math.gcd(inv, rinv)
            if gcd > 1:
                # They are not coprimes
                inv = 0  # Set it to 0, so that we will not add it to reduced invariant
                break
        if inv != 0:
            reduced_invariant.add(inv)
    return sorted(reduced_invariant)
