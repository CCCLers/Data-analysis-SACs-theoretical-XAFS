import os, csv, logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


def sanitize_label(label):
    s = label.replace('\u03b1', 'a').replace('\u03b2', 'b')
    s = s.replace(' ', '_').replace('(', '').replace(')', '')
    s = s.replace('-', '_').replace('.', '_').replace('__', '_')
    return s.strip('_')


def make_shared_x_grid(curves, npts=2000):
    if not curves:
        return np.array([])
    x_min = min(c['x'].min() for c in curves if len(c['x']) > 0)
    x_max = max(c['x'].max() for c in curves if len(c['x']) > 0)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        return np.linspace(0, 1, npts)
    return np.linspace(x_min, x_max, npts)


def write_panel_csv(filepath, panel_name, curves, metadata=None):
    if not curves:
        logger.warning(f"No curves for panel {panel_name}, skipping CSV.")
        return

    x_grid = make_shared_x_grid(curves)

    col_labels = ['Energy (eV)']
    columns = [('Energy (eV)', x_grid)]
    seen = {}

    for c in curves:
        base = sanitize_label(c.get('label', 'unknown'))
        if base in seen:
            seen[base] += 1
            label = f"{base}_{seen[base]}"
        elif base == '' or base == '_nolegend_':
            label = f'curve_{len(seen)}'
            seen[base] = 0
        else:
            seen[base] = 0
            label = base
        y_interp = np.interp(x_grid, c['x'], c['y'])
        columns.append((label, y_interp))
        col_labels.append(label)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}: {v}\n")
        writer = csv.writer(f)
        writer.writerow(col_labels)
        for i in range(len(x_grid)):
            writer.writerow([col[1][i] for col in columns])


def save_figure_png(fig, filepath):
    fig.savefig(filepath, dpi=300, bbox_inches='tight')


def build_metadata(tool, **kwargs):
    parts = []
    if kwargs.get('fwhm') is not None:
        parts.append(f"fwhm={kwargs['fwhm']:.2f}")
    if kwargs.get('is_gaussian'):
        parts.append('gaussian')
    if kwargs.get('shift') is not None:
        s = kwargs['shift']
        if isinstance(s, (int, float)):
            parts.append(f"shift={s:.3f}")
        else:
            parts.append(f"shift={s}")
    if kwargs.get('edge'):
        parts.append(f"edge={kwargs['edge']}")
    if kwargs.get('normalized'):
        parts.append('normalized=True')
    if kwargs.get('split_spin'):
        parts.append('split_spin=True')
    if kwargs.get('gs_unocc_only'):
        parts.append('gs_unocc_only=True')
    parts.append(f"tool={tool}")
    parts.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return {'info': ' | '.join(parts)}
