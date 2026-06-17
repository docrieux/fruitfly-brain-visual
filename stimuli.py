"""
Stimulus library for simulate.py / render_gif.py.

Stimuli are no longer hardcoded here: they live as YAML files under
``stimulus_library/`` and are loaded by :func:`load_library`. Each stimulus
says which neurons receive direct Poisson input, on one or more "channels"
(each channel has its own firing rate). Neurons are chosen by FlyWire
``cell_type`` name, by explicit ``root_id``, or by an annotation-table query
(e.g. ``cell_class: [DAN]``), all resolved against the annotation TSV at run
time.

See stimulus_library/appetitive.yaml for the full schema documentation, and
list_cell_types.py to discover cell types / classes you can select.

Public API:
    load_library(lib_dir=...)  -> {name: definition}     (validated)
    resolve_stimulus(args, pd) -> {name, label, category, valence, channels}
        where channels = [{'rate': <Hz float | 'primary' | 'secondary'>,
                           'ids':  [root_id, ...]}, ...]
The 'primary'/'secondary' rate tokens mean "use the --r-poi / --r-poi2 CLI
defaults"; simulate.py resolves them to concrete Hz.
"""
from pathlib import Path

import yaml

DEFAULT_LIB_DIR = Path(__file__).resolve().parent / 'stimulus_library'

# keys inside a `select` that are not annotation-column filters
_SELECT_SPECIAL = {'types', 'ids', 'cell_type_contains'}


# --------------------------------------------------------------------------- #
# loading & validation
# --------------------------------------------------------------------------- #
def load_library(lib_dir=DEFAULT_LIB_DIR):
    """Load and validate every stimulus_library/*.yaml into one name->def dict."""
    lib_dir = Path(lib_dir)
    if not lib_dir.is_dir():
        raise SystemExit(f'Stimulus library directory not found: {lib_dir}')

    library = {}
    source = {}  # name -> file, for nicer duplicate errors
    for path in sorted(lib_dir.glob('*.yaml')):
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not data:
            continue
        if not isinstance(data, dict):
            raise SystemExit(f'{path.name}: top level must be a mapping of stimulus names')
        for name, defn in data.items():
            if name in library:
                raise SystemExit(
                    f"Duplicate stimulus name '{name}' in {path.name} "
                    f"(already defined in {source[name]})")
            _validate_def(name, defn, path.name)
            library[name] = defn
            source[name] = path.name

    # cross-file checks: compose references must exist and be acyclic
    for name in library:
        _check_compose(name, library, stack=())
    return library


def _validate_def(name, defn, fname):
    if not isinstance(defn, dict):
        raise SystemExit(f"{fname}: stimulus '{name}' must be a mapping")
    if 'label' not in defn:
        raise SystemExit(f"{fname}: stimulus '{name}' is missing required 'label'")

    groups = [
        'channels' in defn,
        ('primary' in defn or 'secondary' in defn),
        'compose' in defn,
    ]
    if sum(groups) != 1:
        raise SystemExit(
            f"{fname}: stimulus '{name}' must use exactly one of: "
            f"'channels', 'primary'/'secondary' shorthand, or 'compose'")

    if 'channels' in defn:
        if not isinstance(defn['channels'], list) or not defn['channels']:
            raise SystemExit(f"{fname}: '{name}'.channels must be a non-empty list")
        for ch in defn['channels']:
            if 'select' not in ch:
                raise SystemExit(f"{fname}: a channel of '{name}' is missing 'select'")
    for role in ('primary', 'secondary'):
        if role in defn and 'select' not in defn[role]:
            raise SystemExit(f"{fname}: '{name}'.{role} is missing 'select'")


def _check_compose(name, library, stack):
    if name not in library:
        raise SystemExit(f"compose reference to unknown stimulus '{name}'")
    if name in stack:
        cycle = ' -> '.join(stack + (name,))
        raise SystemExit(f'compose cycle detected: {cycle}')
    for target in _compose_targets(library[name]):
        _check_compose(target, library, stack + (name,))


def _compose_targets(defn):
    """Names referenced by a compose entry (str or {stimulus: name})."""
    targets = []
    for entry in defn.get('compose', []):
        targets.append(entry['stimulus'] if isinstance(entry, dict) else entry)
    return targets


# --------------------------------------------------------------------------- #
# expansion: stimulus definition -> flat list of {'rate', 'select'} channels
# --------------------------------------------------------------------------- #
def _expand(name, library, stack=()):
    defn = library[name]

    if 'compose' in defn:
        channels = []
        for target in _compose_targets(defn):
            channels.extend(_expand(target, library, stack + (name,)))
        return channels

    if 'channels' in defn:
        out = []
        for ch in defn['channels']:
            rate = ch.get('rate_hz')
            out.append({'rate': 'primary' if rate is None else rate, 'select': ch['select']})
        return out

    # primary/secondary shorthand
    out = []
    if 'primary' in defn:
        out.append({'rate': 'primary', 'select': defn['primary']['select']})
    if 'secondary' in defn:
        out.append({'rate': 'secondary', 'select': defn['secondary']['select']})
    return out


# --------------------------------------------------------------------------- #
# resolving selects -> root_ids against the annotation table
# --------------------------------------------------------------------------- #
def _referenced_columns(channels):
    cols = set()
    for ch in channels:
        cols.update(k for k in ch['select'] if k not in _SELECT_SPECIAL)
    return cols


def _resolve_select(select, ann):
    """Return a set of root_ids matching a single `select` block."""
    ids = set()
    if 'ids' in select:
        ids.update(int(x) for x in select['ids'])
    if 'types' in select:
        ids.update(ann.loc[ann['cell_type'].isin(select['types']), 'root_id'].tolist())

    col_keys = [k for k in select if k not in _SELECT_SPECIAL]
    if col_keys or 'cell_type_contains' in select:
        sub = ann
        for k in col_keys:  # AND across columns, OR within a column's value list
            sub = sub[sub[k].isin(select[k])]
        if 'cell_type_contains' in select:
            sub = sub[sub['cell_type'].str.contains(
                select['cell_type_contains'], case=False, na=False)]
        ids.update(sub['root_id'].tolist())
    return ids


def _select_from_cli(types, ids):
    sel = {}
    if types:
        sel['types'] = list(types)
    if ids:
        sel['ids'] = list(ids)
    return sel


def resolve_stimulus(args, pd, library=None):
    """Build a fully-resolved stimulus dict from CLI args + the YAML library.

    Returns {'name', 'label', 'category', 'valence', 'channels'} where each
    channel is {'rate': <Hz | 'primary' | 'secondary'>, 'ids': [root_id, ...]}.
    Channel order is priority order: a root_id is assigned to the first channel
    that selects it (earlier channels win on overlap).
    """
    if library is None:
        library = load_library(getattr(args, 'library_dir', DEFAULT_LIB_DIR))

    if args.stimulus:
        defn = library[args.stimulus]
        name = args.stimulus
        label = args.label or defn['label']
        category = defn.get('category')
        valence = defn.get('valence')
        channels = _expand(args.stimulus, library)
    else:
        if not (args.primary_types or args.primary_ids):
            raise SystemExit(
                'Specify either --stimulus NAME (see --list-stimuli for choices), '
                'or a custom stimulus via --primary-types / --primary-ids.')
        name = args.out or 'custom'
        label = args.label or 'Custom stimulus'
        category = valence = None
        channels = []
        if args.primary_types or args.primary_ids:
            channels.append({'rate': 'primary',
                             'select': _select_from_cli(args.primary_types, args.primary_ids)})
        if args.secondary_types or args.secondary_ids:
            channels.append({'rate': 'secondary',
                             'select': _select_from_cli(args.secondary_types, args.secondary_ids)})

    need_ann = any(set(ch['select']) - {'ids'} for ch in channels)
    ann = None
    if need_ann:
        cols = ['root_id', 'cell_type'] + sorted(_referenced_columns(channels))
        ann = pd.read_csv(args.annotations, sep='\t', usecols=cols)

    resolved, taken = [], set()
    for ch in channels:
        ids = sorted(_resolve_select(ch['select'], ann) - taken)
        taken.update(ids)
        resolved.append({'rate': ch['rate'], 'ids': ids})

    return {'name': name, 'label': label, 'category': category,
            'valence': valence, 'channels': resolved}
