#!/usr/bin/env python3
"""
Turn spike data from simulate.py into a rotating, anatomically-positioned
GIF of the Drosophila brain lighting up.

Examples
--------
    python render_gif.py --name food_odor_vinegar
    python render_gif.py --name co2_avoidance --out co2.gif
    python render_gif.py --name custom --label "My custom odor"

This reads spikes_<name>.parquet and stimulus_neurons_<name>.csv, which
simulate.py writes (use --out there to control <name>, default is the
stimulus name).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--name', required=True,
                     help='Name used by simulate.py (its --stimulus or --out value)')
    ap.add_argument('--label', default=None,
                     help='Override the title text (default: read from meta_<name>.txt)')
    ap.add_argument('--out', default=None,
                     help='Output GIF path (default: <results-dir>/fly_brain_<name>.gif)')
    ap.add_argument('--results-dir', default='results',
                     help='Where spikes_<name>.parquet etc. live, and where the GIF is written (default: results/)')
    ap.add_argument('--annotations',
                     default='flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv',
                     help='Path to the FlyWire annotation TSV (cell types + 3D positions)')

    # visual tuning knobs
    ap.add_argument('--fine-dt', type=float, default=4.0, help='Bin width (ms) during the fast early cascade')
    ap.add_argument('--fine-until', type=float, default=100.0, help='How long (ms) to use fine bins for')
    ap.add_argument('--coarse-dt', type=float, default=16.0, help='Bin width (ms) for the rest of the run')
    ap.add_argument('--flash-decay', type=float, default=0.30, help='Trailing afterglow of the bright flash layer')
    ap.add_argument('--rotation-deg', type=float, default=22.0, help='Max camera rotation (degrees, pendulum sweep)')
    ap.add_argument('--frame-duration', type=float, default=0.07, help='Seconds per GIF frame')
    ap.add_argument('--hold-start', type=int, default=4, help='Extra repeats of the first frame')
    ap.add_argument('--hold-end', type=int, default=8, help='Extra repeats of the last frame')
    ap.add_argument('--dpi', type=int, default=200, help='Output resolution; higher = sharper, larger file')
    ap.add_argument('--fig-width', type=float, default=9.0, help='Figure width in inches (canvas size)')
    ap.add_argument('--fig-height', type=float, default=7.0, help='Figure height in inches (canvas size)')
    ap.add_argument('--point-scale', type=float, default=1.0,
                     help='Multiplies every neuron marker size. <1 = smaller/sharper dots, >1 = chunkier')
    ap.add_argument('--keep-frames', action='store_true', help='Keep the individual PNG frames on disk')
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    label = args.label
    if label is None:
        meta_path = res_dir / f'meta_{args.name}.txt'
        if meta_path.exists():
            for line in meta_path.read_text().splitlines():
                if line.startswith('label='):
                    label = line[len('label='):]
        label = label or args.name

    out_path = args.out or str(res_dir / f'fly_brain_{args.name}.gif')
    frame_dir = res_dir / f'.frames_{args.name}'
    frame_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ---------- load data ----------
    ann = pd.read_csv(
        args.annotations, sep='\t',
        usecols=['root_id', 'pos_x', 'pos_y', 'pos_z'],
    ).set_index('root_id')

    spikes = pd.read_parquet(res_dir / f'spikes_{args.name}.parquet')
    stim = pd.read_csv(res_dir / f'stimulus_neurons_{args.name}.csv')

    if len(spikes) == 0:
        raise SystemExit('No spikes recorded for this stimulus - nothing to animate. '
                          'Try a higher --r-poi / --r-poi2 or different neurons in simulate.py.')

    # all-neuron positions (background). z is in coarser EM-section voxels
    # (~40nm) than x/y (~4nm); rescale by 10x so 3D rotation is proportionate.
    X_all = ann['pos_x'].values.astype(np.float32)
    Y_all = ann['pos_y'].values.astype(np.float32)
    Z_all = ann['pos_z'].values.astype(np.float32) * 10.0
    cx, cy, cz = X_all.mean(), Y_all.mean(), Z_all.mean()
    X_all -= cx; Y_all -= cy; Z_all -= cz

    # ---------- adaptive time binning (fine during the fast early cascade) ----------
    T_RUN = float(spikes['t_ms'].max())
    fine_end = min(args.fine_until, T_RUN)
    edges = list(np.arange(0, fine_end, args.fine_dt))
    if T_RUN > fine_end:
        edges += list(np.arange(fine_end, T_RUN, args.coarse_dt))
    edges.append(T_RUN)
    bin_edges = np.unique(np.array(edges, dtype=float))
    N_BINS = len(bin_edges) - 1
    print(f'simulated duration: {T_RUN:.0f} ms, {N_BINS} frames')

    active_ids = spikes['root_id'].unique()
    id2row = {rid: i for i, rid in enumerate(active_ids)}
    n_active = len(active_ids)
    print('neurons that fired at least once:', n_active)

    counts = np.zeros((n_active, N_BINS), dtype=np.float32)
    bin_idx = np.clip(np.searchsorted(bin_edges, spikes['t_ms'].values, side='right') - 1, 0, N_BINS - 1)
    row_idx = spikes['root_id'].map(id2row).values
    np.add.at(counts, (row_idx, bin_idx), 1.0)

    # "flash" layer: brief afterglow only, so individual spikes pop visibly
    flash = np.zeros_like(counts)
    flash[:, 0] = counts[:, 0]
    for b in range(1, N_BINS):
        flash[:, b] = counts[:, b] + args.flash_decay * flash[:, b - 1]

    # "recruited" layer: persistent, whether a neuron has ever fired by bin b
    ever_fired = (np.cumsum(counts, axis=1) > 0)

    flash_max = np.percentile(flash[flash > 0], 98.0)

    sub = ann.loc[active_ids]
    Xa = sub['pos_x'].values.astype(np.float32) - cx
    Ya = sub['pos_y'].values.astype(np.float32) - cy
    Za = sub['pos_z'].values.astype(np.float32) * 10.0 - cz

    stim_sub = ann.loc[ann.index.intersection(stim['root_id'].values)]
    Xs = stim_sub['pos_x'].values.astype(np.float32) - cx
    Ys = stim_sub['pos_y'].values.astype(np.float32) - cy
    Zs = stim_sub['pos_z'].values.astype(np.float32) * 10.0 - cz

    heat = LinearSegmentedColormap.from_list(
        'fly_heat', ['#3a0d02', '#b3360b', '#ff8a00', '#ffe28a', '#ffffff']
    )

    XLIM = (-140000, 140000)
    YLIM = (-65000, 65000)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), facecolor='black', dpi=args.dpi)
    ps = args.point_scale  # global marker-size multiplier

    def render_frame(b, angle_deg):
        ax.cla()
        ax.set_facecolor('black')
        theta = np.radians(angle_deg)
        ct, st = np.cos(theta), np.sin(theta)

        xr = X_all * ct + Z_all * st
        zr = -X_all * st + Z_all * ct
        order = np.argsort(zr)
        ax.scatter(xr[order], -Y_all[order], s=0.4 * ps, c='#1f3650', alpha=0.20, linewidths=0)

        rec_mask = ever_fired[:, b]
        if rec_mask.any():
            xr2 = Xa[rec_mask] * ct + Za[rec_mask] * st
            zr2 = -Xa[rec_mask] * st + Za[rec_mask] * ct
            o2 = np.argsort(zr2)
            ax.scatter(xr2[o2], -Ya[rec_mask][o2], s=1.4 * ps, c='#a8551a', alpha=0.45, linewidths=0)

        inten = np.clip(flash[:, b] / flash_max, 0, 1) ** 0.5
        mask = inten > 0.05
        if mask.any():
            xa_r = Xa[mask] * ct + Za[mask] * st
            za_r = -Xa[mask] * st + Za[mask] * ct
            ord2 = np.argsort(za_r)
            colors = heat(inten[mask][ord2])
            sizes = (2.5 + 16 * inten[mask][ord2]) * ps
            ax.scatter(xa_r[ord2], -Ya[mask][ord2], s=sizes, c=colors, linewidths=0, alpha=0.95)

        xs_r = Xs * ct + Zs * st
        ax.scatter(xs_r, -Ys, s=6 * ps, facecolors='none', edgecolors='#ff2fd0', linewidths=0.6, alpha=0.85)

        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.set_aspect('equal'); ax.axis('off')

        t_ms = bin_edges[b]
        n_recruited = int(rec_mask.sum())
        n_flashing = int(mask.sum())
        ax.text(0.5, 0.98, f'Drosophila brain \u2014 response to: {label}',
                transform=ax.transAxes, color='white', fontsize=11, va='top', ha='center')
        ax.text(0.5, 0.015,
                f't = {t_ms:5.0f} ms    recruited: {n_recruited:5d}    flashing now: {n_flashing:4d}    '
                f'(magenta = stimulus input)',
                transform=ax.transAxes, color='#cfd8e3', fontsize=8.5, va='bottom', ha='center', family='monospace')

    frame_paths = []
    for b in range(N_BINS):
        angle = args.rotation_deg * np.sin(2 * np.pi * b / N_BINS)
        render_frame(b, angle)
        p = frame_dir / f'f_{b:03d}.png'
        fig.savefig(p, facecolor='black')
        frame_paths.append(p)

    print(f'rendered {len(frame_paths)} frames in {time.time() - t0:.1f}s')

    images = [imageio.imread(frame_paths[0])] * args.hold_start
    images += [imageio.imread(p) for p in frame_paths]
    images += [imageio.imread(frame_paths[-1])] * args.hold_end
    imageio.mimsave(out_path, images, duration=args.frame_duration, loop=0)

    if not args.keep_frames:
        for p in frame_paths:
            p.unlink()
        frame_dir.rmdir()

    print(f'saved {out_path} ({time.time() - t0:.1f}s total)')


if __name__ == '__main__':
    main()
