#!/usr/bin/env python3
import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
import os
from typing import (
    Dict,
    List,
    Set,
)
import xml.etree.ElementTree as ET
import statistics
import sys

import numpy as np

# Constants
CSV_SEPARATOR = '|'
pnet_prefix = '{http://www.pnml.org/version-2009/grammar/pnml}'
arc_tag = f'{pnet_prefix}arc'
inscription_tag = f'{pnet_prefix}inscription'
net_tag = f'{pnet_prefix}net'
name_tag = f'{pnet_prefix}name'
page_tag = f'{pnet_prefix}page'
place_tag = f'{pnet_prefix}place'
text_tag = f'{pnet_prefix}text'
transition_tag = f'{pnet_prefix}transition'

# Parsers
arg_parser = argparse.ArgumentParser(
    description='PNML reading utility'
)

arg_subparsers = arg_parser.add_subparsers(help='Toolkit mode', dest='mode')

# Statistics mode arguments
stat_parser = arg_subparsers.add_parser('stats')
stat_parser.add_argument('-f', 
                        '--format',
                        help='Output format',
                        default='csv',
                        choices=['csv'])
stat_parser.add_argument('-a',
                        '--add-field',
                        action='append',
                        default=[],
                        dest='output_flags',
                        choices=['arc_weights', 'filename', 'filepath'])
stat_parser.add_argument('file_path', nargs='?')

# Liveness analysis 
stat_parser = arg_subparsers.add_parser('compute-liveness')
stat_parser.add_argument('file_path', nargs='?')

@dataclass
class CSVOutputConfig(object):
    arc_weights: bool = False
    filename: bool = False
    filepath: bool = False

@dataclass 
class PNArc(object):
    _id: str
    source: str
    target: str
    weight: int = 1


@dataclass
class PetriNet(object):
    filepath: str
    name: str
    places: Set[str]
    transitions: Set[str]
    arcs: List[PNArc]

    def into_csv(self, config: CSVOutputConfig) -> str:
        arc_weights = tuple(arc.weight for arc in self.arcs)
        fields = []

        if config.filename:
            fields.append(os.path.basename(self.filepath))

        if config.filepath:
            fields.append(os.path.abspath(self.filepath))

        fields.append(self.name)
        fields.append(str(len(self.places)))
        fields.append(str(len(self.transitions)))
        fields.append(str(min(arc_weights)))
        fields.append(str(max(arc_weights)))
        fields.append(str(statistics.mean(arc_weights)))
        fields.append(str(statistics.stdev(arc_weights)))

        if config.arc_weights:
            fields.append(','.join(map(str, arc_weights)))

        return CSV_SEPARATOR.join(fields) 


def extract_pn(filepath: str, pn_dom: ET.ElementTree) -> PetriNet:
    """
    Extract the petri net from fiven PNML XML document.
    """
    
    net_name_node = pn_dom.find(f'./{net_tag}/{name_tag}/{text_tag}')
    net_name = net_name_node.text

    page_xpath = f'./{net_tag}/{page_tag}'

    places = set(p.attrib['id'] for p in pn_dom.iterfind(f'{page_xpath}/{place_tag}'))
    transitions = set(t.attrib['id'] for t in pn_dom.iterfind(f'{page_xpath}/{transition_tag}'))
    
    arcs: List[PNArc] = []
    for arc_node in pn_dom.iterfind(f'{page_xpath}/{arc_tag}'):
        arc = PNArc(_id=arc_node.get('id'),
                    source=arc_node.get('source'),
                    target=arc_node.get('target'))
            
        has_weight = False
        inscription_node = arc_node.find(f'./{inscription_tag}')
        if inscription_node:
            text_node = inscription_node.find(f'./{text_tag}')
            if text_node is not None:
                arc.weight = int(text_node.text)
                has_weight = True
        arcs.append(arc)

        if not has_weight and len(arc_node) > 1:
            print(f'Anomaly detected when processing {net_name=}')

    return PetriNet(filepath=filepath,
                    name=net_name,
                    places=places,
                    transitions=transitions,
                    arcs=arcs)


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
    p_invariant = np.zeros(len(pn.places), dtype=np.int32)
    _adj_mat = np.copy(adj_mat)  # Make a copy of the adj_mat as we will modify it
    for i in range(len(pn.transitions)):
        column = _adj_mat[:, i]
        nonzero_column = np.where((column == 0), np.ones(len(pn.places), dtype=np.int64), column)
        lcm = np.lcm.reduce(nonzero_column)
        p_fragment = []
        for j, c in enumerate(column):
            if c != 0:
                multiplier = abs(int(lcm / c))
                if multiplier == 1:
                    p_fragment.append(0)
                else:
                    _adj_mat[j] *= multiplier
                    p_fragment.append(multiplier)
            else:
                p_fragment.append(0)

        p_fragment = np.abs(np.array(p_fragment))
        p_invariant += p_fragment
    
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


args = arg_parser.parse_args()
if args.mode is None:
    arg_parser.print_help()
    sys.exit(1)

if args.file_path is not None:
    filename = args.file_path
    root = ET.parse(args.file_path)
else:
    filename = 'stdin'
    root = ET.fromstring(sys.stdin.read())
pn = extract_pn(filename, root)

if args.mode == 'stats':
    csv_config = CSVOutputConfig()
    for output_option in ('arc_weights', 'filename', 'filepath'):
        setattr(csv_config, output_option, output_option in args.output_flags)
    print(pn.into_csv(csv_config))
elif args.mode == 'compute-liveness':
    invariant = compute_liveness_coin_values(pn)
    reduced_inv = reduce_invariant_to_coprimes(invariant)
    print(','.join(map(str, reduced_inv)))

