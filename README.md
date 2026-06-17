# Fly Brain Stimulus GIF Kit

Simulates the *Drosophila melanogaster* whole-brain connectome responding to a sensory stimulus, and renders the result as a rotating, anatomically positioned GIF. Two steps: `simulate.py` runs the network, `render_gif.py` turns the spikes into a GIF. Swapping the stimulus is a one line change.

## Motivations

I wanted to make cool visuals for creative purposes, and I got inspired from OpenWorm, the virtualization of the *C. elegans* nematode project, jumping then to a more complex brain that is completely mapped, the *Drosophila melanogaster*.

## What this actually is

- **Connectome / model**: the leaky integrate-and-fire whole-brain model from Shiu et al. 2023 ("A leaky integrate-and-fire computational model
  based on the connectome of the entire adult *Drosophila* brain..."), built on the FlyWire 783 connectome (~138,600 neurons, ~15M synapses). Code: https://github.com/philshiu/Drosophila_brain_model
- **3D positions / cell types**: the FlyWire community annotation tables (real reconstructed neuron coordinates + cell-type labels). Code: https://github.com/flyconnectome/flywire_annotations
- **Simulator**: Brian2 2.9.0 (spiking neural network simulator), CPU only. No GPU is needed at this scale - a 700ms whole-brain run takes roughly 1-2 minutes on a single CPU core (model build ~2s, plus simulation time).

This is a real simulation on a real connectome, not a recording of actual neural activity - it's a simplified point-neuron model (no dendritic computation, no neuromodulation, weights derived from raw synapse counts).

## Setup

Dependencies are pinned in `requirements.txt` to a known-good set (verified on Python 3.11). Install with [uv](https://docs.astral.sh/uv/):

```bash
uv venv                            # create .venv (uses Python 3.11)
uv pip install -r requirements.txt
bash setup_data.sh                 # downloads ~280MB of public connectome data
```

Then activate the environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere) before running the scripts below, or prefix commands with `uv run`.

You need Python 3.11. No GPU, no special hardware. If you have gcc/g++ installed you can optionally use `--backend cython` for a real compiled build (only worth it for long or repeated runs - there's a one-off compile cost that can make it *slower* for a single short run).

> **NumPy is pinned to `<2` on purpose.** Brian2 2.9.0 references  `np.ndarray.ptp`, which NumPy 2.0 removed, so a newer NumPy breaks `import brian2`. Keep the pin unless you also upgrade Brian2.

## Quick start

```bash
python simulate.py --stimulus food_odor_vinegar
python render_gif.py --name food_odor_vinegar
```

This writes the simulation outputs to `results/` and produces `results/fly_brain_food_odor_vinegar.gif`:

![Drosophila brain responding to a vinegar-like food odor](docs/fly_brain_food_odor_vinegar.gif)

The magenta rings are the stimulated input neurons; warm dots are downstream neurons firing as the response propagates through the connectome.

## Project layout

```
simulate.py            run the network for a chosen stimulus -> results/spikes_<name>.parquet
render_gif.py          turn those spikes into results/fly_brain_<name>.gif
stimuli.py             loads + resolves the YAML stimulus library
list_cell_types.py     search the FlyWire annotations for cell types/classes
stimulus_library/      the stimulus catalog, as YAML grouped by internal state
results/               all generated outputs (git-ignored)
docs/                  static assets for this README
setup_data.sh          clones the two public datasets below
Drosophila_brain_model/   Shiu et al. LIF model + connectivity (cloned, git-ignored)
flywire_annotations/      FlyWire cell types + 3D positions (cloned, git-ignored)
```

## Switching the stimulus

Stimuli live as YAML files in `stimulus_library/`, grouped by the fly internal state they target. List the whole catalog with:

```bash
python simulate.py --list-stimuli
```

A few of the presets:

| name                    | category   | what it is                                         |
| ----------------------- | ---------- | -------------------------------------------------- |
| `food_odor_vinegar`   | appetitive | attractive food odor (Or42b/Or92a glomeruli)       |
| `geosmin_aversion`    | aversive   | geosmin, a dedicated "toxic microbe" odor (Or56a)  |
| `sugar_taste`         | appetitive | sugar-sensing gustatory neurons (taste, not smell) |
| `reward_pam`          | reward     | PAM dopaminergic "reward" teaching signal          |
| `punishment_ppl1`     | punishment | PPL1 dopaminergic "punishment" teaching signal     |
| `appetite_vs_disgust` | conflict   | food attraction driven against CO2 aversion        |

### Example: run a preset

Drive food attraction and CO2 aversion at the same time, then render it:

```bash
python simulate.py --stimulus appetite_vs_disgust
python render_gif.py --name appetite_vs_disgust
```

This writes `results/spikes_appetite_vs_disgust.parquet` (alongside a `stimulus_neurons_*.csv` and `meta_*.txt`) and renders `results/fly_brain_appetite_vs_disgust.gif`. Swap `appetite_vs_disgust` for any name from `--list-stimuli` to run a different one.

> **Conflict stimuli** (category `conflict`) drive two contrary states at once. Because this is a pure leaky integrate-and-fire model with no plasticity or neuromodulation, they show the two populations' activity *colliding* in shared downstream targets - a neural interference pattern, not a resolved decision.

### Using your own stimulus

Two ways:

**1. Quick one-off, from the command line** - no need to edit any file:

```bash
python simulate.py \
    --primary-types ORN_DC3 ORN_DC4 \
    --label "My custom odor" \
    --out my_odor

python render_gif.py --name my_odor
```

`--primary-types` takes any FlyWire `cell_type` name. Use `list_cell_types.py` to find them:

```bash
python list_cell_types.py ORN              # every olfactory glomerulus
python list_cell_types.py --class gustatory # everything tagged "gustatory"
python list_cell_types.py --class-list      # see all available classes
```

You can also target explicit neurons by FlyWire root_id with `--primary-ids 720575940624963786 ...` (mix and match with `--secondary-ids` for a weaker second channel).

**2. Add a permanent preset** - drop a new entry into any `stimulus_library/*.yaml` file (or add your own `.yaml` there), following the existing examples. A stimulus selects neurons by `cell_type`, explicit `root_id`, or an annotation query (e.g. `cell_class: [DAN]`), can drive several
rate channels, and can `compose` other stimuli. Then `--stimulus your_name` works everywhere. See the header of `stimulus_library/appetitive.yaml` for the full schema.

## Other knobs worth knowing about

`simulate.py`:

- `--t-run` - simulated duration in ms (default 700). Longer runs cost roughly linear extra wall-clock time.
- `--r-poi` / `--r-poi2` - firing rate (Hz) for the primary/secondary input neurons. Higher = stronger/more "concentrated" stimulus.

`render_gif.py`:

- `--rotation-deg` - how far the camera swings (default a gentle ±22°).
- `--fine-dt` / `--fine-until` / `--coarse-dt` - control time resolution; the default uses fine 4ms bins for the first 100ms (where most of the
  interesting cascade happens) and coarser 16ms bins after that.
- `--frame-duration` - seconds per GIF frame (controls playback speed).
- `--dpi` - output resolution (default 200). Higher = sharper but larger file and slower render.
- `--fig-width` / `--fig-height` - canvas size in inches (default 9x7). Combined with `--dpi` this sets the pixel dimensions.
- `--point-scale` - global multiplier on every neuron marker (default 1.0). Each neuron is a single dot at its soma position; use `<1` (e.g. `0.6`) for finer, more pointillistic detail, `>1` for chunkier blobs.
- `--keep-frames` - keep the individual PNGs instead of deleting them.

## How it works, briefly

1. `simulate.py` picks the requested neurons, drives them with Poisson spike trains (the "stimulus"), runs the whole-brain LIF network for `--t-run` ms, and saves every spike (neuron + time) to a parquet file.
2. `render_gif.py` bins those spikes in time, builds two visual layers per frame - a persistent dim "recruited" footprint and a bright "flashing right now" overlay with brief afterglow - places every neuron at its real 3D FlyWire coordinate, applies a small rotation each frame for a 3D feel, and stitches the frames into a GIF.

## Troubleshooting

- *"No spikes recorded"*: your chosen neurons may not connect strongly enough into the network, or the rate is too low - try a higher `--r-poi` or different cell types.
- *Slow simulation*: this environment only has CPU. The `numpy` Brian2 backend (default) is normally fastest for a single short run; `cython` helps mainly when running many trials or much longer durations.
- *IDs not found warning*: the connectivity table and annotation table are both pinned to FlyWire materialization `783`, so this should be rare - it usually means a typo'd root_id.
- *`AttributeError: ... 'numpy.ndarray' has no attribute 'ptp'` on  `import brian2`*: you have NumPy 2.x installed. Reinstall the pinned set (`uv pip install -r requirements.txt`) or downgrade with  `uv pip install "numpy<2"`.

## AI Statement

The code from this repo has been generated entirely with assistance of *Claude Opus 4.8.* There's no license since I don't claim ownership of the development of the tool.
