from pnet_toolkit.data import PetriNet, PNArc
from pnet_toolkit.liveness import (
    build_adjacency_matrix_from_pnet,
    compute_liveness_coin_values,
    compute_p_invariants,
)

import pytest
import numpy as np

"""
There are no tests at the moment, only PNets used to debug the toolkit
for future reference.
"""

# Very simple pnet to check whether the adj matrix extraction works
pn = PetriNet(
    filepath='',
    name='',
    places={'p1', 'p2'},
    transitions={'t1', 't2'},
    arcs=[
        PNArc(_id='', source='p1', target='t1', weight=1),
        PNArc(_id='', source='t1', target='p2', weight=2),

        PNArc(_id='', source='p2', target='t2', weight=2),
        PNArc(_id='', source='t2', target='p1', weight=2),
    ]
)

@pytest.fixture()
def chrzastowski_pnet() -> PetriNet:
    """
    Petri net extracted from the chrzastowski paper

    Expected p-invariant: [4, 12, 15, 5]
    Expected coin values: [4, 5]
    """
    return PetriNet(
        filepath='',
        name='test2',
        places={'p1', 'p2', 'p3', 'p4'},
        transitions={'t1', 't2', 't3', 't4'},
        arcs=[
            PNArc(_id='t1-1', source='p4', target='t1', weight=4),
            PNArc(_id='t1-2', source='t1', target='p1', weight=5),

            PNArc(_id='t2-1', source='p1', target='t2', weight=9),
            PNArc(_id='t2-2', source='t2', target='p2', weight=3),

            PNArc(_id='t3-1', source='p2', target='t3', weight=5),
            PNArc(_id='t3-2', source='t3', target='p3', weight=4),

            PNArc(_id='t4-1', source='p3', target='t4', weight=3),
            PNArc(_id='t4-2', source='t4', target='p4', weight=9),
        ]
    )


def test_invariants_from_chrzastowski_paper(chrzastowski_pnet: PetriNet):
    adj_mat = build_adjacency_matrix_from_pnet(chrzastowski_pnet)
    actual = compute_p_invariants(adj_mat)
    expected = np.array([[4, 12, 15, 5]])

    assert actual.shape == (1, 4)
    assert (actual == expected).all()


def test_coins_from_chrzastowski_paper(chrzastowski_pnet: PetriNet):
    actual = compute_liveness_coin_values(chrzastowski_pnet, verify_result=True)
    expected = np.array([4, 12, 15, 5])

    assert (actual == expected).all()


def test_pinvariant_computation():
    mat = np.array([
        [-1, 1, 1, -1],
        [1, -1, -1, 1],
        [0, 0, 1, 0],
        [1, 0, 0, -1],
        [-1, 0, 0, 1],
    ], dtype=np.int64)

    actual = compute_p_invariants(mat)
    expected = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1],
    ])

    assert actual.shape == (3, 5)
    assert (actual == expected).all()
