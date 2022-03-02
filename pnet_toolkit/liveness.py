from collections import defaultdict
import math
from typing import (
    Dict,
    List,
    Set,
)

from pnet_toolkit.data import PetriNet

import numpy as np



def compute_liveness_coin_values(pn: PetriNet):
    """
    Generate the frobenius coin problem coin values that determine 
    the liveness of the given petri net.
    """

    # Construct the adjacency matrix: columns are transitions and rows are place deltas
    
    # Collect arcs for individual transitions
    transition_arcs: Dict[str, List[PNArc]] = defaultdict(list)
    for arc in pn.arcs:
        transition = arc.source if arc.source in pn.transitions else arc.target
        transition_arcs[transition].append(arc)

    # Initialize the matrix
    adj_mat = np.zeros((len(pn.places), len(pn.transitions)), dtype=np.int64)

    # Assign indices for transitions and places
    transition_indices: Dict[str, int] = dict((transition, i) for i, transition in enumerate(sorted(pn.transitions))) 
    place_indices: Dict[str, int] = dict((place, i) for i, place in enumerate(sorted(pn.places))) 

    # Go over the transitions and populate the matrix
    for transition in pn.transitions:
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

    # Use the adjacency matrix A and calculate a P-invariant p such that
    p_invariant = np.ones(len(pn.places), dtype=np.int64)
    _adj_mat = np.copy(adj_mat)  # Make a copy of the adj_mat as we will modify it
    for i in range(len(pn.transitions)):
        column = _adj_mat[:, i]
        
        nonzero_column = np.where((column == 0), np.ones(len(pn.places), dtype=np.int64), column)
        lcm = np.lcm.reduce(nonzero_column)

        row_multipliers_zero = np.floor_divide(lcm, column, dtype=np.int64)
        row_multipliers = np.where((row_multipliers_zero == 0), np.ones(len(pn.places), dtype=np.int64), row_multipliers_zero)
        
        # Multiply the rows in the matrix, so that they have all the same value
        _adj_mat = np.multiply(_adj_mat, row_multipliers[:, np.newaxis])

        # Find the first nonzero row and use it to zero out all rows 
        first_nonzero_row_i = None 
        for row_i, row_value in enumerate(column):
            if row_value != 0:
                first_nonzero_row_i = row_i
                break
        
        # The entire column has been zeroed out before
        if first_nonzero_row_i is None:
            continue
        else:
            a = np.transpose(
                    np.reshape(
                        np.repeat(_adj_mat[first_nonzero_row_i], len(pn.places)),
                        (len(pn.places), len(pn.transitions))
                    )
                )
            m = np.where((row_multipliers_zero != 0), np.ones(len(pn.places), dtype=np.int64), row_multipliers_zero)

            _adj_mat -= np.multiply(a, m[:, np.newaxis])

        p_invariant *= np.abs(row_multipliers)
    
    return p_invariant


def reduce_invariant_to_coprimes(invariant: List[int]) -> List[int]:
    """Filter out those invariant elements that are not coprimes."""
    sorted_invariant = sorted(invariant)
    print(sorted_invariant)
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


