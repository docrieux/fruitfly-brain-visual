#!/usr/bin/env python3
"""
Run a whole-brain leaky integrate-and-fire simulation (Shiu et al. 2023's
model, on the FlyWire 783 connectome) driven by a configurable stimulus.

Examples
--------
    python simulate.py --list-stimuli                 # see the catalog
    python simulate.py --stimulus food_odor_vinegar
    python simulate.py --stimulus reward_vs_punishment --t-run 500
    python simulate.py --stimulus sugar_taste --r-poi 100

    # custom stimulus, not in the library:
    python simulate.py --primary-types ORN_DC3 --label "my odor" --out my_odor

Requires the two data repos to be cloned next to this script (see
setup_data.sh / README.md):
    ./Drosophila_brain_model/
    ./flywire_annotations/

Notes on speed: with the 'numpy' Brian2 backend (the default here, no
compiler required) a 700ms whole-brain run takes about 30-60s on one CPU
core. If you have gcc/g++ available, you can pass --backend cython for a
real C++ build, which is faster for long runs but pays a one-off compile
cost (often *slower* for a single short run - benchmark before committing).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from stimuli import load_library, resolve_stimulus


def _rate_hz(token, args):
    """Resolve a channel rate (a number, or the 'primary'/'secondary' token)."""
    if token == 'primary':
        return args.r_poi
    if token == 'secondary':
        return args.r_poi2
    return float(token)


def _print_library(library):
    """List stimuli grouped by category (for --list-stimuli)."""
    by_cat = {}
    for name, defn in library.items():
        by_cat.setdefault(defn.get('category') or 'uncategorized', []).append((name, defn['label']))
    for cat in sorted(by_cat):
        print(f'\n[{cat}]')
        for name, label in sorted(by_cat[cat]):
            print(f'    {name:<24} {label}')
    print()


def main():
    library = load_library()

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--stimulus', choices=sorted(library), default=None,
                     help='Name of a stimulus from the stimulus_library/ YAML files')
    ap.add_argument('--list-stimuli', action='store_true',
                     help='List the available stimuli grouped by category, then exit')
    ap.add_argument('--primary-types', nargs='*', default=[],
                     help='Cell types (e.g. ORN_DM1) driven at the primary rate')
    ap.add_argument('--secondary-types', nargs='*', default=[],
                     help='Cell types driven at the secondary (weaker) rate')
    ap.add_argument('--primary-ids', nargs='*', type=int, default=[],
                     help='Explicit FlyWire root_ids, driven at the primary rate')
    ap.add_argument('--secondary-ids', nargs='*', type=int, default=[],
                     help='Explicit FlyWire root_ids, driven at the secondary rate')
    ap.add_argument('--label', default=None,
                     help='Human-readable label, used later in the GIF title')
    ap.add_argument('--out', default=None,
                     help='Output name prefix (default: the stimulus name)')
    ap.add_argument('--t-run', type=float, default=700.0,
                     help='Simulated duration in ms (default: 700)')
    ap.add_argument('--r-poi', type=float, default=150.0,
                     help='Primary input firing rate in Hz (default: 150)')
    ap.add_argument('--r-poi2', type=float, default=60.0,
                     help='Secondary input firing rate in Hz (default: 60)')
    ap.add_argument('--backend', default='numpy', choices=['numpy', 'cython'],
                     help="Brian2 codegen target. 'numpy' needs no compiler. "
                          "'cython' needs gcc/g++ and is worth it mainly for long/many runs.")
    ap.add_argument('--model-dir', default='Drosophila_brain_model',
                     help='Path to the cloned philshiu/Drosophila_brain_model repo')
    ap.add_argument('--annotations',
                     default='flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv',
                     help='Path to the FlyWire annotation TSV (cell types + 3D positions)')
    ap.add_argument('--results-dir', default='results',
                     help='Where to write spikes_<name>.parquet etc. (default: results/)')
    args = ap.parse_args()

    if args.list_stimuli:
        _print_library(library)
        return

    import brian2
    brian2.prefs.codegen.target = args.backend
    from brian2 import ms, Hz, Network, PoissonInput

    sys.path.insert(0, args.model_dir)
    from model import create_model, default_params  # noqa: E402  (path set above)

    stim = resolve_stimulus(args, pd, library)
    name = args.out or stim['name']

    tag = f"  [{stim['category']}/{stim['valence']}]" if stim['category'] else ''
    print(f">>> Stimulus:          {stim['label']}{tag}")

    params = dict(default_params)
    params['t_run'] = args.t_run * ms

    df_comp = pd.read_csv(f'{args.model_dir}/Completeness_783.csv', index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {i: j for j, i in flyid2i.items()}

    t0 = time.time()
    neu, syn, spk_mon = create_model(
        f'{args.model_dir}/Completeness_783.csv',
        f'{args.model_dir}/Connectivity_783.parquet',
        params,
    )

    # Build one PoissonInput per neuron per channel, at that channel's rate.
    # Mirrors model.poi() (model.py) but supports an arbitrary number of rate
    # channels instead of just primary/secondary.
    weight = params['w_syn'] * params['f_poi']
    pois = []
    n_missing = 0
    channel_rows = []  # rows for stimulus_neurons_<name>.csv
    for ci, ch in enumerate(stim['channels']):
        rate = _rate_hz(ch['rate'], args)
        idx = [flyid2i[i] for i in ch['ids'] if i in flyid2i]
        n_missing += len(ch['ids']) - len(idx)
        print(f"    channel {ci}: {len(idx):>4} neurons  (rate={rate} Hz)")
        for i in idx:
            pois.append(PoissonInput(target=neu[i], target_var='v', N=1,
                                     rate=rate * Hz, weight=weight))
            neu[i].rfc = 0 * ms  # no refractory period for Poisson targets
        channel_rows.extend({'root_id': rid, 'channel': ci, 'rate_hz': rate} for rid in ch['ids'])
    if n_missing:
        print(f"    note: {n_missing} requested neuron(s) not found in the connectivity table "
              f"(could be a different connectome materialization)")

    net = Network(neu, syn, spk_mon, *pois)

    print(f"    model built in {time.time() - t0:.1f}s — running {args.t_run:.0f} ms of simulated time...")
    net.run(params['t_run'])
    print(f"    simulation done, total wall time {time.time() - t0:.1f}s")

    spk_trn = spk_mon.spike_trains()
    idx, t_ms = [], []
    for i, times in spk_trn.items():
        if len(times):
            idx.extend([i] * len(times))
            t_ms.extend((times / ms))

    df = pd.DataFrame({'brian_idx': idx, 't_ms': t_ms})
    df['root_id'] = df['brian_idx'].map(i2flyid)
    df = df.drop(columns=['brian_idx'])

    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(res_dir / f'spikes_{name}.parquet')
    pd.DataFrame(channel_rows, columns=['root_id', 'channel', 'rate_hz']).to_csv(
        res_dir / f'stimulus_neurons_{name}.csv', index=False)
    (res_dir / f'meta_{name}.txt').write_text(
        f"label={stim['label']}\ncategory={stim['category']}\n"
        f"valence={stim['valence']}\nt_run_ms={args.t_run}\n")

    print(f"    total spikes: {len(df)}, unique neurons that fired: {df['root_id'].nunique()}")
    print(f"    saved to {res_dir}/: spikes_{name}.parquet, stimulus_neurons_{name}.csv, meta_{name}.txt")


if __name__ == '__main__':
    main()
