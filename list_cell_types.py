#!/usr/bin/env python3
"""
Search the FlyWire cell-type annotations to find names you can plug into
stimuli.py, or pass directly to simulate.py --primary-types / --secondary-types.

Examples
--------
    python list_cell_types.py ORN              # all olfactory receptor neuron glomeruli
    python list_cell_types.py MBON              # mushroom body output neurons
    python list_cell_types.py --class gustatory # everything in the gustatory class
    python list_cell_types.py --class-list      # show all available cell_class values
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('query', nargs='?', default='', help='Substring to search for in cell_type (case-insensitive)')
ap.add_argument('--class', dest='cell_class', default=None,
                 help='Filter by exact cell_class instead, e.g. olfactory, gustatory, '
                      'mechanosensory, thermosensory, hygrosensory, Kenyon_Cell, MBON, ALPN ...')
ap.add_argument('--class-list', action='store_true', help='List all available cell_class values and exit')
ap.add_argument('--annotations',
                 default='flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv')
args = ap.parse_args()

ann = pd.read_csv(args.annotations, sep='\t', usecols=['cell_type', 'cell_class', 'super_class'])

if args.class_list:
    print(sorted(ann['cell_class'].dropna().unique().tolist()))
    raise SystemExit

if args.cell_class:
    sub = ann[ann['cell_class'].str.lower() == args.cell_class.lower()]
else:
    sub = ann[ann['cell_type'].str.contains(args.query, case=False, na=False)]

counts = sub['cell_type'].value_counts()
if counts.empty:
    print('No matches. Try --class-list to see available cell_class values, '
          'or a shorter/different substring.')
else:
    print(counts.to_string())
    print(f'\n{len(counts)} distinct cell types, {int(counts.sum())} neurons total')
