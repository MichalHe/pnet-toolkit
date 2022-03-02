#!/usr/bin/env python3
import argparse
import os
import sys
import xml.etree.ElementTree as ET

from pnet_toolkit.data import CSVOutputConfig
from pnet_toolkit.liveness import compute_liveness_coin_values
from pnet_toolkit.parsing import extract_pn

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

