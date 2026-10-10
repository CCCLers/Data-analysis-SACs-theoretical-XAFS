import os
import re
import glob
import logging
import numpy as np
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinterdnd2 import DND_FILES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import export_utils as xport

logger = logging.getLogger(__name__)

_ORBITAL_GROUPS = {
    's': ['s'],
    'p': ['px', 'py', 'pz'],
    'd': ['d-2', 'd-1', 'd0', 'd+1', 'd+2'],
    'f': ['f-3', 'f-2', 'f-1', 'f0', 'f+1', 'f+2', 'f+3'],
}

_SELECTION_RULES = {
    'K': {'allowed': 'p',  'desc': '1s → np'},
    'L3': {'allowed': 'd', 'desc': '2p → nd'},
}

_EDGE_OPTIONS = ['K', 'L3']

_D_SUBGROUP_NAMES = {
    'd_xy':   'd-2',
    'd_yz':   'd-1',
    'd_z2':   'd0',
    'd_xz':   'd+1',
    'd_x2y2': 'd+2',
}

_D_SUBGROUP_LABELS = {
    'd_xy':   'd_{xy}    (t2g)',
    'd_yz':   'd_{yz}    (t2g)',
    'd_z2':   'd_{z^2}   (eg)',
    'd_xz':   'd_{xz}    (t2g)',
    'd_x2y2': 'd_{x^2-y^2} (eg)',
}

_D_SUBGROUP_KEYS = ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2')

VALID_HYB_GROUPS = {'s', 'p', 'd', 'f', 'd_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'}

_CB_PALETTE = [
    ('#0072B2', '--'),  #  0: Blue
    ('#D55E00', '-.'),  #  1: Vermillion
    ('#009E73', ':'),   #  2: Bluish green
    ('#CC79A7', '--'),  #  3: Reddish purple
    ('#E69F00', '-.'),  #  4: Orange
    ('#332288', ':'),   #  5: Indigo
    ('#999933', '--'),  #  6: Olive
    ('#44AA99', '-.'),  #  7: Teal
    ('#CC6677', ':'),   #  8: Rose
    ('#117733', '--'),  #  9: Dark green
    ('#882255', '-.'),  # 10: Wine
    ('#AA4499', ':'),   # 11: Magenta-purple
    ('#DDAA33', '--'),  # 12: Dark gold
    ('#CC3311', '-.'),  # 13: Dark red
]


def _parse_pdos_header(filepath):
    with open(filepath, 'r') as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()
    m = re.search(r'atomic kind (\w+)', line1)
    atom = m.group(1) if m else '?'
    fermi_au = float(line1.split('=')[-1].strip().split()[0])
    cols = line2.split()[3:]
    return atom, fermi_au, cols


def parse_pdos_file_list(pdos_files):
    if not pdos_files:
        return None

    file_info = []
    for fp in pdos_files:
        try:
            atom, fermi, cols = _parse_pdos_header(fp)
        except Exception as e:
            logger.error(f"Skipping {fp}: header parse failed ({e})")
            continue
        fname = os.path.basename(fp)
        spin = 'alpha' if 'ALPHA' in fname else ('beta' if 'BETA' in fname else 'unknown')
        file_info.append({'path': fp, 'atom': atom, 'spin': spin, 'fermi': fermi, 'cols': cols})

    if not file_info:
        return None

    fermi_au = file_info[0]['fermi']

    atoms_data = {}
    for info in file_info:
        atom = info['atom']
        if atom not in atoms_data:
            atoms_data[atom] = {'files': [], 'orbital_names': []}
        atoms_data[atom]['files'].append(info)
        atoms_data[atom]['orbital_names'] = [c for c in info['cols'] if c not in ('[a.u.]', 'Occupation')]

    result = {'fermi_au': fermi_au, 'fermi_ev': fermi_au * 27.2114}
    result['atoms'] = {}

    for atom, info in atoms_data.items():
        result['atoms'][atom] = {'orbital_names': info['orbital_names']}

        for finfo in info['files']:
            spin = finfo['spin']
            try:
                data = np.loadtxt(finfo['path'], skiprows=2)
            except Exception as e:
                logger.error(f"Failed to load {finfo['path']}: {e}")
                continue
            e_au = data[:, 1]
            occ = data[:, 2]
            e_ev = (e_au - fermi_au) * 27.2114

            orb_data = {}
            col_names = finfo['cols']
            for i, cname in enumerate(col_names):
                if cname in ('[a.u.]', 'Occupation'):
                    continue
                orb_data[cname] = data[:, 1 + i]

            grouped = {}
            for group, members in _ORBITAL_GROUPS.items():
                present = [m for m in members if m in col_names]
                if present:
                    grouped[group] = np.sum([orb_data[m] for m in present], axis=0)
            for subgroup, member in _D_SUBGROUP_NAMES.items():
                if member in col_names:
                    grouped[subgroup] = orb_data[member]

            result['atoms'][atom][spin] = {
                'energy': e_ev,
                'occupation': occ,
                'orbitals': orb_data,
                'groups': grouped,
            }

    return result


def parse_pdos_directory(directory):
    pdos_files = sorted(glob.glob(os.path.join(directory, '*.pdos')))
    return parse_pdos_file_list(pdos_files)


def _compute_d_composition(data, atom='Fe', occ_threshold=0.5):
    if data is None or atom not in data['atoms']:
        return None
    ad = data['atoms'][atom]
    keys = list(_D_SUBGROUP_NAMES.keys())
    comp = {k: {'alpha': 0.0, 'beta': 0.0, 'total': 0.0} for k in keys}
    for spin in ('alpha', 'beta'):
        if spin not in ad:
            continue
        occ = ad[spin]['occupation']
        mask = occ > occ_threshold
        orb = ad[spin].get('orbitals', {})
        spin_total = 0.0
        for sub, col in _D_SUBGROUP_NAMES.items():
            if col in orb:
                val = np.sum(orb[col][mask])
                comp[sub][spin] = val
                comp[sub]['total'] += val
                spin_total += val
        if spin_total > 0:
            for sub in keys:
                comp[sub][spin] = 100.0 * comp[sub][spin] / spin_total if comp[sub][spin] > 0 else 0.0
    grand_total = sum(comp[k]['total'] for k in keys)
    if grand_total <= 0:
        return None
    for k in keys:
        comp[k]['total'] = 100.0 * comp[k]['total'] / grand_total
        for spin in ('alpha', 'beta'):
            if comp[k][spin] > 0 and comp[k][spin] < 0.01:
                comp[k][spin] = 0.0
    return comp


def get_total_pdos(pdos, atom, spin, group=None):
    """Get total PDOS for a given atom and spin, optionally for a specific orbital group."""
    if atom not in pdos['atoms'] or spin not in pdos['atoms'][atom]:
        return None, None
    d = pdos['atoms'][atom][spin]
    e = d['energy']
    if group and group in d['groups']:
        dos = d['groups'][group]
    else:
        dos = np.sum([d['groups'].get(g, np.zeros_like(e)) for g in ['s', 'p', 'd', 'f']], axis=0)
    return e, dos


def get_spin_sum(pdos, atom, group=None):
    """Sum alpha + beta PDOS for a given atom and orbital group."""
    e_a, dos_a = get_total_pdos(pdos, atom, 'alpha', group)
    e_b, dos_b = get_total_pdos(pdos, atom, 'beta', group)
    if e_a is None and e_b is None:
        return None, None
    if e_a is None:
        return e_b, dos_b
    if e_b is None:
        return e_a, dos_a
    # Interpolate beta onto alpha grid (different MO counts possible)
    if len(e_a) != len(e_b):
        common_min = max(e_a.min(), e_b.min())
        common_max = min(e_a.max(), e_b.max())
        npts = min(len(e_a), len(e_b))
        e_common = np.linspace(common_min, common_max, npts)
        dos_a_interp = np.interp(e_common, e_a, dos_a)
        dos_b_interp = np.interp(e_common, e_b, dos_b)
        return e_common, dos_a_interp + dos_b_interp
    return e_a, dos_a + dos_b


def align_energy(e_pdos, xas_e0, method='auto'):
    """
    Align PDOS energy axis to XAS energy axis.
    - auto: align first unoccupied PDOS peak to XAS E0
    - manual: shift by user-specified value
    Returns (aligned_energy, shift_eV).
    """
    if method == 'auto' and xas_e0 is not None:
        return e_pdos + xas_e0, xas_e0
    return e_pdos, 0.0


def broaden_pdos(energy, dos, x_grid, fwhm, is_gaussian=False):
    """Broaden PDOS sticks into continuous DOS using same algorithm as XAS."""
    if len(energy) == 0 or len(dos) == 0:
        return np.zeros_like(x_grid)
    from process_xas import calculate_y
    regions = [(energy[0], energy[-1], fwhm)]
    return calculate_y(energy, dos, x_grid, regions, is_gaussian)


def compute_tme(xas_e, xas_y, pdos_e, pdos_y, x_grid):
    """Compute Transition Matrix Element: TME(E) = XAS(E) / PDOS(E)."""
    f_xas = interp1d(xas_e, xas_y, kind='cubic', bounds_error=False, fill_value=0.0)
    f_pdos = interp1d(pdos_e, pdos_y, kind='cubic', bounds_error=False, fill_value=0.0)
    xas_interp = f_xas(x_grid)
    pdos_interp = f_pdos(x_grid)
    pdos_safe = np.clip(pdos_interp, 1e-10, None)
    tme = xas_interp / pdos_safe
    tme = np.clip(tme, 0, np.percentile(tme[tme < np.inf], 99) if np.any(np.isfinite(tme)) else 1)
    return x_grid, tme, xas_interp, pdos_interp


def _robust_load_csv(filepath):
    try:
        n_skip = 1
        with open(filepath, 'r') as f:
            first = f.readline()
            if first.startswith('# E0_VAL:'):
                n_skip = 2
        data = np.loadtxt(filepath, delimiter=',', skiprows=n_skip)
        if data.ndim < 2 or data.shape[1] < 2:
            return None, None, None
        return data[:, 0], data[:, 1], n_skip
    except:
        return None, None, None


class PDOSXASWindow:
    def __init__(self, master, directory=None):
        self.master = master
        master.title("PDOS-XAS Combined Analysis")
        master.geometry("1400x900")

        self.pdos_data = None
        self.xas_energy = None
        self.xas_intensity = None
        self.xas_e0 = None
        self.xas_filepath = None
        self.pdos_dir = directory or ''

        # Control variables
        self.edge_var = tk.StringVar(value='K')
        self.fwhm_var = tk.DoubleVar(value=0.8)
        self.is_gaussian_var = tk.BooleanVar(value=False)
        self.energy_shift_var = tk.DoubleVar(value=0.0)
        self.show_spin_alpha = tk.BooleanVar(value=True)
        self.show_spin_beta = tk.BooleanVar(value=True)
        self.show_xas = tk.BooleanVar(value=True)
        self.show_tme = tk.BooleanVar(value=True)
        self.unocc_only_var = tk.BooleanVar(value=False)
        self.normalize_var = tk.BooleanVar(value=True)
        self.split_spin_var = tk.BooleanVar(value=True)
        self.show_d_subplot_var = tk.BooleanVar(value=False)
        self.show_d_components_var = tk.BooleanVar(value=False)
        self.mirror_spin_var = tk.BooleanVar(value=False)
        self.align_mode_var = tk.StringVar(value='E0')

        self.orbital_vars = {}
        self.orbital_styles = {}
        self._broad_cache = {}
        self._plotted_curves = {}

        self._setup_ui()

        if directory:
            self._load_pdos(directory)

    def _setup_ui(self):
        main_panel = ttk.Frame(self.master)
        main_panel.pack(fill=tk.BOTH, expand=True)

        ctrl_frame = ttk.LabelFrame(main_panel, text="Controls", width=320)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        ctrl_frame.pack_propagate(False)

        scroll_cv = tk.Canvas(ctrl_frame, highlightthickness=0, width=310)
        scroll_sb = ttk.Scrollbar(ctrl_frame, orient="vertical", command=scroll_cv.yview)
        scroll_frame = ttk.Frame(scroll_cv)
        scroll_frame.bind("<Configure>", lambda e: scroll_cv.configure(scrollregion=scroll_cv.bbox("all")))
        scroll_cv.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_cv.configure(yscrollcommand=scroll_sb.set)
        scroll_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_sb.pack(side=tk.RIGHT, fill=tk.Y)
        def _on_mousewheel(event):
            scroll_cv.yview_scroll(int(-event.delta / 120), "units")
        def _bind_wheel(event):
            scroll_cv.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_wheel(event):
            scroll_cv.unbind_all("<MouseWheel>")
        scroll_cv.bind("<Enter>", _bind_wheel)
        scroll_cv.bind("<Leave>", _unbind_wheel)

        plot_frame = ttk.Frame(main_panel)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === XAS Controls ===
        xas_section = ttk.LabelFrame(scroll_frame, text="XAS Spectrum", padding=5)
        xas_section.pack(fill=tk.X, pady=2)

        self.lbl_xas = ttk.Label(xas_section, text="No file loaded", foreground="gray")
        self.lbl_xas.pack(fill=tk.X)

        xas_btn_frame = ttk.Frame(xas_section)
        xas_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(xas_btn_frame, text="Load CSV", command=self._load_xas_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(xas_btn_frame, text="Load Spectrum", command=self._load_xas_spectrum).pack(side=tk.LEFT, padx=2)

        # Drag-drop zone for XAS
        self.xas_drop = tk.Text(xas_section, height=2, relief="solid", bd=1)
        self.xas_drop.pack(fill=tk.X, pady=2)
        self.xas_drop.insert(tk.END, "\n  Drop XAS CSV here")
        self.xas_drop.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.xas_drop.dnd_bind('<<Drop>>', self._on_xas_drop)  # type: ignore[attr-defined]
        self.xas_drop.config(state="disabled")

        # === PDOS Controls ===
        pdos_section = ttk.LabelFrame(scroll_frame, text="PDOS Data", padding=5)
        pdos_section.pack(fill=tk.X, pady=2)

        self.lbl_pdos = ttk.Label(pdos_section, text="No PDOS loaded", foreground="gray")
        self.lbl_pdos.pack(fill=tk.X)

        pdos_btn_frame = ttk.Frame(pdos_section)
        pdos_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(pdos_btn_frame, text="Load Directory", command=self._load_pdos_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(pdos_btn_frame, text="Load .pdos Files", command=self._load_pdos_files).pack(side=tk.LEFT, padx=2)

        # Drag-drop zone for PDOS
        self.pdos_drop = tk.Text(pdos_section, height=2, relief="solid", bd=1)
        self.pdos_drop.pack(fill=tk.X, pady=2)
        self.pdos_drop.insert(tk.END, "\n  Drop .pdos files/folder here")
        self.pdos_drop.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.pdos_drop.dnd_bind('<<Drop>>', self._on_pdos_drop)  # type: ignore[attr-defined]
        self.pdos_drop.config(state="disabled")

        # === Alignment ===
        align_section = ttk.LabelFrame(scroll_frame, text="Energy Alignment", padding=5)
        align_section.pack(fill=tk.X, pady=2)

        align_row1 = ttk.Frame(align_section)
        align_row1.pack(fill=tk.X)
        ttk.Label(align_row1, text="Edge:").pack(side=tk.LEFT)
        edge_cb = ttk.Combobox(align_row1, textvariable=self.edge_var, values=_EDGE_OPTIONS, state="readonly", width=5)
        edge_cb.pack(side=tk.LEFT, padx=5)
        edge_cb.bind("<<ComboboxSelected>>", lambda e: self._update_plot())
        ttk.Label(align_row1, text="Selection Rule:").pack(side=tk.LEFT, padx=5)
        self.lbl_rule = ttk.Label(align_row1, text="", foreground="blue")
        self.lbl_rule.pack(side=tk.LEFT)
        self._update_rule_label()

        align_modes = [
            ('E0', 'LUMO → E0'),
            ('Pre-edge', 'Pre-edge → Fe-d'),
            ('Main-edge', 'Main-edge → Fe-p'),
        ]
        for val, txt in align_modes:
            rb = ttk.Radiobutton(align_section, text=txt, variable=self.align_mode_var,
                                 value=val, command=self._auto_align)
            rb.pack(anchor=tk.W, padx=10)

        shift_row = ttk.Frame(align_section)
        shift_row.pack(fill=tk.X)
        ttk.Label(shift_row, text="Fine Shift:").pack(side=tk.LEFT)
        self.ent_shift = ttk.Entry(shift_row, textvariable=self.energy_shift_var, width=8)
        self.ent_shift.pack(side=tk.LEFT, padx=2)
        ttk.Label(shift_row, text="eV").pack(side=tk.LEFT)
        self.ent_shift.bind('<Return>', lambda e: self._update_plot())

        self.lbl_align_info = ttk.Label(align_section, text="Align info: --",
                                        foreground="gray", font=("Arial", 8), wraplength=300)
        self.lbl_align_info.pack(anchor=tk.W, padx=2, fill=tk.X)

        ttk.Button(align_section, text="Auto Align", command=self._auto_align).pack(pady=2)

        # Core-hole note
        ttk.Label(align_section,
                  text="Note: Ground-state PDOS lacks core-hole effect.\n  Pre-edge/main-edge spacing may differ from XAS.",
                  foreground="#888", font=("Arial", 7)).pack(anchor=tk.W, padx=2)

        # === Broadening ===
        broad_section = ttk.LabelFrame(scroll_frame, text="PDOS Broadening", padding=5)
        broad_section.pack(fill=tk.X, pady=2)

        broad_row = ttk.Frame(broad_section)
        broad_row.pack(fill=tk.X)
        ttk.Label(broad_row, text="FWHM:").pack(side=tk.LEFT)
        self.ent_fwhm = ttk.Entry(broad_row, textvariable=self.fwhm_var, width=6)
        self.ent_fwhm.pack(side=tk.LEFT, padx=2)
        ttk.Label(broad_row, text="eV").pack(side=tk.LEFT)
        self.ent_fwhm.bind('<Return>', lambda e: self._update_plot())
        ttk.Checkbutton(broad_row, text="Gaussian", variable=self.is_gaussian_var,
                        command=self._update_plot).pack(side=tk.LEFT, padx=5)

        # === Orbital Selection ===
        orb_section = ttk.LabelFrame(scroll_frame, text="Orbitals to Display", padding=5)
        orb_section.pack(fill=tk.X, pady=2)

        d_toggle_row = ttk.Frame(orb_section)
        d_toggle_row.pack(fill=tk.X, pady=(0, 2))
        self.btn_split_d = ttk.Button(d_toggle_row, text="\u25b6 d components",
                                      command=self._toggle_d_components, width=16)
        self.btn_split_d.pack(side=tk.LEFT)

        self.orbital_frame = ttk.Frame(orb_section)
        self.orbital_frame.pack(fill=tk.X)

        # === Display Toggles ===
        display_section = ttk.LabelFrame(scroll_frame, text="Display Options", padding=5)
        display_section.pack(fill=tk.X, pady=2)

        ttk.Checkbutton(display_section, text="Show XAS", variable=self.show_xas,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="Show TME", variable=self.show_tme,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="Spin Alpha", variable=self.show_spin_alpha,
                        command=self._update_plot).pack(anchor=tk.W, side=tk.LEFT, padx=5)
        ttk.Checkbutton(display_section, text="Spin Beta", variable=self.show_spin_beta,
                        command=self._update_plot).pack(anchor=tk.W, side=tk.LEFT, padx=5)
        ttk.Checkbutton(display_section, text="Normalize [0,1]", variable=self.normalize_var,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="Unoccupied Only", variable=self.unocc_only_var,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="Split α/β Spins", variable=self.split_spin_var,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="d-Subplot Row", variable=self.show_d_subplot_var,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="Mirror β Spin (up/down)", variable=self.mirror_spin_var,
                        command=self._update_plot).pack(anchor=tk.W)

        # === Export ===
        export_sec = ttk.LabelFrame(scroll_frame, text="Export", padding=5)
        export_sec.pack(fill=tk.X, pady=2)
        export_row = ttk.Frame(export_sec)
        export_row.pack(fill=tk.X)
        ttk.Button(export_row, text="Export CSV", command=self._export_csv).pack(side=tk.LEFT, padx=3, pady=2)
        ttk.Button(export_row, text="Export Image", command=self._export_image).pack(side=tk.LEFT, padx=3, pady=2)

        # === d-orbital Composition ===
        dcomp_sec = ttk.LabelFrame(scroll_frame, text="d-orbital Composition", padding=5)
        dcomp_sec.pack(fill=tk.X, pady=2)
        ttk.Button(dcomp_sec, text="Compute (occupied)", command=self._compute_d_comp).pack(anchor=tk.W, pady=2)
        self._dcomp_text = tk.Text(dcomp_sec, height=8, width=34, font=("Consolas", 8), wrap=tk.NONE, state="disabled")
        self._dcomp_text.pack(fill=tk.X, pady=2)

        # === Plot Area ===
        self.fig = plt.Figure(figsize=(12, 10), dpi=100)
        self.gs = GridSpec(2, 2, figure=self.fig, width_ratios=[2.5, 1], height_ratios=[1, 1],
                           hspace=0.35, wspace=0.30)

        self.ax_xas = self.fig.add_subplot(self.gs[0, 0])
        self.ax_tme = self.fig.add_subplot(self.gs[0, 1])
        self.ax_hyb = self.fig.add_subplot(self.gs[1, :])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame)

        self._init_plot_labels()

    def _init_plot_labels(self):
        for ax, t, xl, yl in [
            (self.ax_xas, "XAS + PDOS Comparison", "Energy (eV)", "Intensity / DOS"),
            (self.ax_tme, "Transition Matrix Element", "Energy (eV)", "TME (a.u.)"),
            (self.ax_hyb, "Orbital Hybridization Analysis", "Energy (eV)", "PDOS (states/eV)"),
        ]:
            ax.set_title(t, fontsize=11)
            ax.set_xlabel(xl, fontsize=10)
            ax.set_ylabel(yl, fontsize=10)
            ax.grid(True, alpha=0.25, linestyle='--')

    def _update_rule_label(self):
        edge = self.edge_var.get()
        rule = _SELECTION_RULES.get(edge, {})
        self.lbl_rule.config(text=rule.get('desc', ''))

    # --- Data Loading ---
    def _load_xas_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if fp:
            self._read_xas_csv(fp)

    def _load_xas_spectrum(self):
        fp = filedialog.askopenfilename(filetypes=[("Spectrum files", "*.spectrum")])
        if fp:
            self._read_xas_spectrum(fp)

    def _parse_dropped_files(self, event_data):
        pattern = r'\{.*?\}|\S+'
        matches = re.findall(pattern, event_data)
        return [m.strip('{}') for m in matches]

    def _on_xas_drop(self, event):
        files = self._parse_dropped_files(event.data)
        if not files:
            return
        fp = files[0]
        if fp.lower().endswith('.csv'):
            self._read_xas_csv(fp)
        elif fp.lower().endswith('.spectrum'):
            self._read_xas_spectrum(fp)

    def _on_pdos_drop(self, event):
        files = self._parse_dropped_files(event.data)
        if not files:
            return
        pdos_files = [f for f in files if f.lower().endswith('.pdos')]
        dirs = [f for f in files if os.path.isdir(f)]
        if dirs:
            self._load_pdos(dirs[0])
        elif pdos_files:
            self._load_pdos_from_files(pdos_files)

    def _read_xas_csv(self, filepath):
        e, y, n = _robust_load_csv(filepath)
        if e is None:
            messagebox.showerror("Error", "Failed to parse CSV.")
            return
        e0 = None
        with open(filepath, 'r') as f:
            first = f.readline()
            if first.startswith('# E0_VAL:'):
                try:
                    e0 = float(first.split(':')[1].strip())
                except:
                    pass

        self.xas_energy = e
        self.xas_intensity = y
        self.xas_e0 = e0 if e0 else e[np.argmax(np.abs(y))]
        self.xas_filepath = filepath
        self.lbl_xas.config(text=f"XAS: {os.path.basename(filepath)}" + (f" (E0={self.xas_e0:.1f})" if self.xas_e0 else ""))
        logger.info(f"Loaded XAS: {filepath}, E0={self.xas_e0}")
        self._auto_align()
        self._update_plot()
        self.master.lift()

    def _read_xas_spectrum(self, filepath):
        from process_xas import parse_spectrum_data, compute_spectrum
        cfg = {'atom_kind': 'Fe', 'donor_type': '1s', 'excitation_type': '2p',
               'emin': '700', 'emax': '730', 'npts': '3000',
               'auto_split': True, 'split_strategy': 'std3',
               'calculated_e0': None,
               'broadening_regions_lorentz': '0.4,1.0'}
        raw = parse_spectrum_data(filepath, cfg)
        if raw is None:
            messagebox.showerror("Error", "Failed to parse spectrum.")
            return
        result = compute_spectrum(raw, cfg)
        if result is None:
            return
        self.xas_energy = result['x']
        self.xas_intensity = result['y']
        self.xas_e0 = result['e0']
        self.xas_filepath = filepath
        self.lbl_xas.config(text=f"XAS: {os.path.basename(filepath)} (E0={self.xas_e0:.1f})")
        logger.info(f"Loaded spectrum: {filepath}, E0={self.xas_e0}")
        self._auto_align()
        self._update_plot()
        self.master.lift()

    def _load_pdos_dir(self):
        initial = self.pdos_dir if self.pdos_dir else (os.path.dirname(self.xas_filepath) if self.xas_filepath else '.')
        d = filedialog.askdirectory(title="Select folder containing .pdos files", initialdir=initial)
        if d:
            self._load_pdos(d)

    def _load_pdos_files(self):
        initial = self.pdos_dir if self.pdos_dir else (os.path.dirname(self.xas_filepath) if self.xas_filepath else '.')
        fps = filedialog.askopenfilenames(
            title="Select .pdos Files",
            filetypes=[("PDOS files", "*.pdos")],
            initialdir=initial,
        )
        if fps:
            self._load_pdos_from_files(list(fps))

    def _load_pdos_from_files(self, file_paths):
        self._broad_cache.clear()
        es_files = [f for f in file_paths if '-x-' in os.path.basename(f)]
        data = parse_pdos_file_list(es_files) if es_files else parse_pdos_file_list(file_paths)
        if data is None:
            self.lbl_pdos.config(text="No valid .pdos files found", foreground="red")
            return
        self.pdos_data = data
        self.pdos_dir = os.path.dirname(file_paths[0])
        atoms = list(data['atoms'].keys())
        self.lbl_pdos.config(text=f"PDOS: {', '.join(atoms)} | Fermi={data['fermi_ev']:.2f}eV", foreground="black")
        logger.info(f"Loaded PDOS from files: atoms={atoms}, Fermi={data['fermi_ev']:.2f}eV")
        self._rebuild_orbital_controls()
        lumo = self._compute_lumo()
        lumo_txt = f"LUMO={lumo:.2f} eV" if lumo is not None else "LUMO: N/A (all occ)"
        self.lbl_align_info.config(text=f"PDOS {lumo_txt} | Load XAS & click Auto Align",
                                   foreground="gray")
        self._auto_align()
        self._update_plot()
        self.master.lift()

    def _load_pdos(self, directory):
        self._broad_cache.clear()
        es_files = sorted(glob.glob(os.path.join(directory, '*-x-*.pdos')))
        data = parse_pdos_file_list(es_files) if es_files else None
        if data is None:
            self.lbl_pdos.config(text="No ES (core-excited) .pdos files found", foreground="red")
            return
        self.pdos_data = data
        self.pdos_dir = directory
        atoms = list(data['atoms'].keys())
        self.lbl_pdos.config(text=f"PDOS: {', '.join(atoms)} | Fermi={data['fermi_ev']:.2f}eV", foreground="black")
        logger.info(f"Loaded ES PDOS: atoms={atoms}, Fermi={data['fermi_ev']:.2f}eV")
        self._rebuild_orbital_controls()
        lumo = self._compute_lumo()
        lumo_txt = f"LUMO={lumo:.2f} eV" if lumo is not None else "LUMO: N/A (all occ)"
        self.lbl_align_info.config(text=f"PDOS {lumo_txt} | Load XAS & click Auto Align",
                                   foreground="gray")
        self._auto_align()
        self._update_plot()
        self.master.lift()

    def _get_default_orbital_style(self, atom, group):
        color, ls = _CB_PALETTE[0]
        return {'color': color, 'linestyle': ls, 'linewidth': 1.5, 'alpha': 0.8,
                'beta_linestyle': ls, 'beta_linewidth': 1.5}

    def _rebuild_orbital_controls(self):
        for w in self.orbital_frame.winfo_children():
            w.destroy()
        old_vars = self.orbital_vars
        old_styles = self.orbital_styles
        self.orbital_vars = {}
        self.orbital_styles = {}

        if not self.pdos_data:
            return

        palette_idx = 0
        show_d_comp = self.show_d_components_var.get()
        for atom, ad in self.pdos_data['atoms'].items():
            spin_key = 'alpha' if 'alpha' in ad else ('beta' if 'beta' in ad else None)
            if not spin_key:
                continue
            avail_groups = list(ad[spin_key]['groups'].keys())

            for g in avail_groups:
                key = f"{atom}_{g}"
                default = (g == 'p') or (atom in ('Fe', 'Fex', 'Co', 'Ni', 'Mn', 'Cu', 'Ti', 'Zn') and g == 'd')
                if g in _D_SUBGROUP_NAMES:
                    default = False
                self.orbital_vars[key] = old_vars.get(key, tk.BooleanVar(value=default))

                if old_styles.get(key):
                    style = old_styles[key]
                else:
                    color, ls = _CB_PALETTE[palette_idx % len(_CB_PALETTE)]
                    lw = 1.5
                    style = {
                        'color': color, 'linestyle': ls, 'linewidth': lw, 'alpha': 0.8,
                        'beta_linestyle': ls, 'beta_linewidth': lw,
                    }
                    palette_idx += 1
                self.orbital_styles[key] = style

                style = self.orbital_styles[key]
                row = ttk.Frame(self.orbital_frame)
                row.pack(fill=tk.X, pady=1)

                cb = ttk.Checkbutton(row, text=f"{atom}-{g}", variable=self.orbital_vars[key],
                                     command=self._update_plot, width=8)
                cb.pack(side=tk.LEFT)

                btn_color = tk.Button(row, text='\u25a0', fg=style['color'],
                                      width=2, bd=1, relief='ridge',
                                      command=lambda k=key: self._pick_orbital_color(k))
                btn_color.pack(side=tk.LEFT, padx=1)

                ttk.Label(row, text="\u03b1:", font=("Arial", 7)).pack(side=tk.LEFT)
                cb_ls_a = ttk.Combobox(row, values=['-', '--', '-.', ':'], width=3, state='readonly')
                cb_ls_a.set(style['linestyle'])
                cb_ls_a.bind('<<ComboboxSelected>>', lambda e, k=key: self._set_orbital_style(k, 'linestyle', e.widget.get()))
                cb_ls_a.pack(side=tk.LEFT, padx=1)
                sb_lw_a = ttk.Spinbox(row, from_=0.5, to=5.0, increment=0.5, width=3)
                sb_lw_a.set(style['linewidth'])
                sb_lw_a.bind('<Return>', lambda e, k=key, w=sb_lw_a: self._set_orbital_style(k, 'linewidth', float(w.get())))
                sb_lw_a.bind('<FocusOut>', lambda e, k=key, w=sb_lw_a: self._set_orbital_style(k, 'linewidth', float(w.get())))
                sb_lw_a.pack(side=tk.LEFT, padx=1)

                ttk.Label(row, text="\u03b2:", font=("Arial", 7)).pack(side=tk.LEFT, padx=(4, 0))
                cb_ls_b = ttk.Combobox(row, values=['-', '--', '-.', ':'], width=3, state='readonly')
                cb_ls_b.set(style.get('beta_linestyle', '--'))
                cb_ls_b.bind('<<ComboboxSelected>>', lambda e, k=key: self._set_orbital_style(k, 'beta_linestyle', e.widget.get()))
                cb_ls_b.pack(side=tk.LEFT, padx=1)
                sb_lw_b = ttk.Spinbox(row, from_=0.5, to=5.0, increment=0.5, width=3)
                sb_lw_b.set(style.get('beta_linewidth', 1.0))
                sb_lw_b.bind('<Return>', lambda e, k=key, w=sb_lw_b: self._set_orbital_style(k, 'beta_linewidth', float(w.get())))
                sb_lw_b.bind('<FocusOut>', lambda e, k=key, w=sb_lw_b: self._set_orbital_style(k, 'beta_linewidth', float(w.get())))
                sb_lw_b.pack(side=tk.LEFT, padx=1)

    def _toggle_d_components(self):
        new = not self.show_d_components_var.get()
        self.show_d_components_var.set(new)
        self.btn_split_d.config(text="\u25bc d components" if new else "\u25b6 d components")
        self._rebuild_orbital_controls()
        self._update_plot()

    def _pick_orbital_color(self, key):
        curr = self.orbital_styles.get(key, {}).get('color', '#000000')
        from tkinter import colorchooser
        res = colorchooser.askcolor(initialcolor=curr, title=f"Color for {key}")
        if res and res[1]:
            self.orbital_styles[key]['color'] = res[1]
            self._update_plot()

    def _set_orbital_style(self, key, attr, value):
        if key not in self.orbital_styles:
            self.orbital_styles[key] = {}
        self.orbital_styles[key][attr] = value
        self._update_plot()

    def _compute_lumo(self):
        """Find the lowest unoccupied state energy (eV, relative to Fermi) across all atoms."""
        lumo = np.inf
        for atom, ad in self.pdos_data['atoms'].items():
            for spin in ['alpha', 'beta']:
                if spin not in ad:
                    continue
                mask = ad[spin]['occupation'] < 0.5
                if mask.any():
                    e_first = ad[spin]['energy'][mask][0]
                    if e_first < lumo:
                        lumo = e_first
        return lumo if np.isfinite(lumo) else None

    def _detect_xas_preedge_peak(self):
        """Find the first significant local maximum in XAS before E0."""
        if self.xas_energy is None:
            return self.xas_e0 or 0
        e, y = self.xas_energy, self.xas_intensity
        e0 = self.xas_e0 or e[len(e)//2]
        # Restrict to region before E0, ignore very small fluctuations
        mask = (e < e0 - 0.5) & (e > e.min() + 2)
        if mask.sum() < 3:
            return e[e < e0][-1] if (e < e0).any() else e0 - 2
        ye, ee = y[mask], e[mask]
        # Find local maxima (1st derivative sign change)
        d = np.diff(ye)
        peaks = []
        for i in range(1, len(d)):
            if d[i-1] > 0 and d[i] <= 0:
                peaks.append((ee[i], ye[i]))
        if not peaks:
            # Fallback: return E0 - 2
            return ee[-1]
        # Return the highest pre-edge peak
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[0][0]

    def _detect_xas_mainedge_peak(self):
        """Find the global maximum (white line) of XAS."""
        if self.xas_energy is None:
            return self.xas_e0 or 0
        e, y = self.xas_energy, self.xas_intensity
        idx = np.argmax(y)
        return e[idx]

    def _detect_pdos_first_peak(self, atom, group):
        """Broaden PDOS (atom, group) with shift=0, find the first local maximum above Fermi."""
        if atom not in self.pdos_data['atoms']:
            return 0
        e_src, dos_src = self._extract_pdos(atom, group, 0,
                                            self.pdos_data['atoms'][atom])
        if e_src is None or len(e_src) < 3:
            return 0
        fwhm = self.fwhm_var.get()
        is_g = self.is_gaussian_var.get()
        xg = np.linspace(e_src.min(), e_src.max(), 2000)
        dos = self._broaden_safe(e_src, dos_src, xg, fwhm, is_g)
        thresh = np.max(dos) * 0.05
        d = np.diff(dos)
        for i in range(1, len(d)):
            if d[i-1] > 0 and d[i] <= 0 and dos[i] > thresh:
                return xg[i]
        return xg[np.argmax(dos)]

    def _detect_pdos_main_peak(self, atom, group):
        """Broaden PDOS (atom, group) with shift=0, find the highest peak."""
        if atom not in self.pdos_data['atoms']:
            return 0
        e_src, dos_src = self._extract_pdos(atom, group, 0,
                                            self.pdos_data['atoms'][atom])
        if e_src is None or len(e_src) < 3:
            return 0
        fwhm = self.fwhm_var.get()
        is_g = self.is_gaussian_var.get()
        xg = np.linspace(e_src.min(), e_src.max(), 2000)
        dos = self._broaden_safe(e_src, dos_src, xg, fwhm, is_g)
        return xg[np.argmax(dos)]

    def _auto_align(self):
        if self.xas_e0 is None or self.pdos_data is None:
            return
        mode = self.align_mode_var.get()
        prev_shift = self.energy_shift_var.get()
        info_parts = []

        if mode == 'E0':
            lumo = self._compute_lumo()
            if lumo is None:
                shift = 0.0
                info_parts.append("LUMO: N/A (all states occupied)")
                info_parts.append("shift=E0-LUMO: 0.0 (defer to PDOS absolute scale)")
            else:
                shift = self.xas_e0 - lumo
                info_parts.append(f"LUMO={lumo:.2f} eV")
                info_parts.append(f"shift=E0-LUMO={shift:.2f}")

        elif mode == 'Pre-edge':
            xas_peak = self._detect_xas_preedge_peak()
            pdos_peak = self._detect_pdos_first_peak('Fe', 'd')
            shift = xas_peak - pdos_peak
            info_parts.append(f"XAS pre-edge @ {xas_peak:.2f} eV")
            info_parts.append(f"Fe-d first peak @ {pdos_peak:.2f} eV")
            info_parts.append(f"shift={shift:.2f}")

        elif mode == 'Main-edge':
            xas_peak = self._detect_xas_mainedge_peak()
            pdos_peak = self._detect_pdos_main_peak('Fe', 'p')
            shift = xas_peak - pdos_peak
            info_parts.append(f"XAS white line @ {xas_peak:.2f} eV")
            info_parts.append(f"Fe-p main peak @ {pdos_peak:.2f} eV")
            info_parts.append(f"shift={shift:.2f}")

        else:
            return

        self.energy_shift_var.set(shift)
        self.ent_shift.update()
        self.lbl_align_info.config(text=" | ".join(info_parts), foreground="black")
        logger.info(f"Auto-align: mode={mode}, shift={shift:.2f}")
        self._update_plot()

    # --- Plotting ---
    def _extract_pdos(self, atom, group, shift, ad, spin=None):
        if spin is not None:
            if spin not in ad:
                return None, None
            do_spin = (spin == 'alpha' and self.show_spin_alpha.get()) or \
                      (spin == 'beta' and self.show_spin_beta.get())
            if not do_spin:
                return None, None
            e_raw = ad[spin]['energy'].copy()
            dos_raw = ad[spin]['groups'].get(group, np.zeros_like(e_raw)).copy()
            if self.unocc_only_var.get():
                mask = ad[spin]['occupation'] < 0.5
                e_raw = e_raw[mask]
                dos_raw = dos_raw[mask]
            return e_raw + shift, dos_raw

        dos_total = None
        energy_src = None
        for s in ['alpha', 'beta']:
            do_spin = (s == 'alpha' and self.show_spin_alpha.get()) or \
                      (s == 'beta' and self.show_spin_beta.get())
            if not do_spin or s not in ad:
                continue
            e_raw = ad[s]['energy'].copy()
            dos_raw = ad[s]['groups'].get(group, np.zeros_like(e_raw)).copy()
            if self.unocc_only_var.get():
                mask = ad[s]['occupation'] < 0.5
                e_raw = e_raw[mask]
                dos_raw = dos_raw[mask]
            e = e_raw + shift
            if dos_total is None:
                dos_total = dos_raw
                energy_src = e
            else:
                if len(e) != len(energy_src):
                    lo = max(energy_src.min(), e.min())
                    hi = min(energy_src.max(), e.max())
                    n = min(len(energy_src), len(e))
                    e_new = np.linspace(lo, hi, n)
                    dos_total = np.interp(e_new, energy_src, dos_total)
                    dos_raw = np.interp(e_new, e, dos_raw)
                    energy_src = e_new
                dos_total += dos_raw
        if dos_total is not None:
            return energy_src, dos_total
        return None, None

    def _broaden_safe(self, energy, dos, x_grid, fwhm, is_gauss):
        if energy is None or len(energy) < 2:
            return np.zeros_like(x_grid)
        return broaden_pdos(energy, dos, x_grid, fwhm, is_gauss)

    def _export_csv(self):
        if not any(self._plotted_curves.values()):
            messagebox.showwarning("No Data", "No curves to export. Load data first.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                          title="Export CSV (base name for multi-panel)")
        if not fp:
            return
        md = xport.build_metadata('pdos_es',
                                  fwhm=self.fwhm_var.get(), is_gaussian=self.is_gaussian_var.get(),
                                  shift=self.energy_shift_var.get(), edge=self.edge_var.get(),
                                  normalized=self.normalize_var.get(), split_spin=self.split_spin_var.get())
        base, ext = os.path.splitext(fp)
        panels = [('xas', 'XAS+PDOS'), ('tme', 'TME'), ('hyb', 'Hybridization'), ('dcomp', 'd-Orbital Components')]
        for key, pname in panels:
            if self._plotted_curves.get(key):
                xport.write_panel_csv(f"{base}_{key}{ext}", pname, self._plotted_curves[key], metadata=md)
        self._write_dcomp_csv(base)
        messagebox.showinfo("Done", f"CSV exported:\n  {base}_[xas,tme,hyb,dcomp]{ext} + dcomp")

    def _write_dcomp_csv(self, base_path):
        if self.pdos_data is None:
            return
        atom = 'Fex' if 'Fex' in self.pdos_data['atoms'] else ('Fe' if 'Fe' in self.pdos_data['atoms'] else None)
        if atom is None:
            return
        comp = _compute_d_composition(self.pdos_data, atom)
        if comp is None:
            return
        fp = f"{base_path}_dcomp.csv"
        with open(fp, 'w') as f:
            f.write("# d-orbital composition (occupied, occ>0.5)\n")
            f.write("component,alpha_percent,beta_percent,total_percent\n")
            for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
                c = comp[sub]
                f.write(f"{sub},{c['alpha']:.1f},{c['beta']:.1f},{c['total']:.1f}\n")

    def _compute_d_comp(self):
        if self.pdos_data is None:
            self._dcomp_text.config(state="normal")
            self._dcomp_text.delete("1.0", tk.END)
            self._dcomp_text.insert(tk.END, "Load PDOS first")
            self._dcomp_text.config(state="disabled")
            return
        atom = 'Fex' if 'Fex' in self.pdos_data['atoms'] else ('Fe' if 'Fe' in self.pdos_data['atoms'] else None)
        if atom is None:
            self._dcomp_text.config(state="normal")
            self._dcomp_text.delete("1.0", tk.END)
            self._dcomp_text.insert(tk.END, "No Fe/Fex atom in PDOS")
            self._dcomp_text.config(state="disabled")
            return
        comp = _compute_d_composition(self.pdos_data, atom)
        if comp is None:
            self._dcomp_text.config(state="normal")
            self._dcomp_text.delete("1.0", tk.END)
            self._dcomp_text.insert(tk.END, "No d character found")
            self._dcomp_text.config(state="disabled")
            return
        lines = [f"Fe d-orbital composition (all ES MOs):", "",
                 f"                  alpha  beta  total",
                 f"  {'─'*30}"]
        for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
            c = comp[sub]
            a_str = f"{c['alpha']:5.1f}" if c['alpha'] > 0 else "   — "
            b_str = f"{c['beta']:5.1f}" if c['beta'] > 0 else "   — "
            t_str = f"{c['total']:5.1f}"
            lines.append(f"  {_D_SUBGROUP_LABELS[sub]:20s} {a_str} {b_str} {t_str}")
        self._dcomp_text.config(state="normal")
        self._dcomp_text.delete("1.0", tk.END)
        self._dcomp_text.insert(tk.END, "\n".join(lines))
        self._dcomp_text.config(state="disabled")

    def _export_image(self):
        fp = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                          title="Export Image")
        if not fp:
            return
        xport.save_figure_png(self.fig, fp)
        messagebox.showinfo("Done", f"Image saved to:\n{fp}")

    def _update_plot(self, *args):
        if self.pdos_data is None and self.xas_energy is None:
            return

        shift = self.energy_shift_var.get()
        fwhm = self.fwhm_var.get()
        is_gauss = self.is_gaussian_var.get()
        edge = self.edge_var.get()
        allowed_orb = _SELECTION_RULES.get(edge, {}).get('allowed', 'p')
        do_norm = self.normalize_var.get()
        mirror = self.mirror_spin_var.get()

        # Complete figure rebuild to eliminate stale twin axes
        self.fig.clear()
        self._plotted_curves = {'xas': [], 'tme': [], 'hyb': [], 'dcomp': []}
        show_d = self.show_d_subplot_var.get()
        if show_d and mirror:
            n_rows, ratios = 4, [1, 1, 0.6, 0.6]
        elif show_d:
            n_rows, ratios = 3, [1, 1, 0.8]
        else:
            n_rows, ratios = 2, [1, 1]
        self.gs = GridSpec(n_rows, 2, figure=self.fig, width_ratios=[2.5, 1],
                           height_ratios=ratios, hspace=0.45, wspace=0.30)
        self.ax_xas = self.fig.add_subplot(self.gs[0, 0])
        self.ax_tme = self.fig.add_subplot(self.gs[0, 1])
        self.ax_hyb = self.fig.add_subplot(self.gs[1, :])
        self._init_plot_labels()

        e_fermi_shifted = shift
        self._update_rule_label()

        e_min, e_max = np.inf, -np.inf
        if self.xas_energy is not None:
            e_min = min(e_min, self.xas_energy[0])
            e_max = max(e_max, self.xas_energy[-1])

        # ---- Subplot 1: XAS + PDOS (dual Y-axis) ----
        ax1 = self.ax_xas
        ax1.clear(); ax1.set_title("XAS + PDOS Comparison", fontsize=11)
        ax1.set_xlabel("Energy (eV)", fontsize=10)
        ax1.set_ylabel("XAS Intensity (±)" if mirror else "XAS Intensity", fontsize=10, color='black')
        ax1.grid(True, alpha=0.25, linestyle='--')

        ax1_pdos = ax1.twinx()
        ax1_pdos.set_ylabel("PDOS (α↑ / β↓)" if mirror else ("PDOS (norm)" if do_norm else "PDOS (states/eV)"),
                            fontsize=10, color='tab:red')

        if self.xas_energy is not None and self.show_xas.get():
            y = self.xas_intensity
            xas_max = np.max(y)
            if do_norm and xas_max > 0:
                y = y / xas_max
            ax1.plot(self.xas_energy, y, 'k-', linewidth=2.0, label="XAS")
            self._plotted_curves['xas'].append({'label': 'XAS', 'x': self.xas_energy.copy(), 'y': y.copy()})
            if mirror:
                ax1.plot(self.xas_energy, -y, 'k-', linewidth=2.0, alpha=0.5, label="XAS (β mirror)")
                self._plotted_curves['xas'].append({'label': 'XAS (mirror)', 'x': self.xas_energy.copy(), 'y': (-y).copy()})
            if self.xas_e0:
                ax1.axvline(self.xas_e0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        all_pdos_curves = []

        if self.pdos_data:
            for atom, ad in self.pdos_data['atoms'].items():
                for group in ad.get('alpha', ad.get('beta', {})).get('groups', {}):
                    key = f"{atom}_{group}"
                    if not self.orbital_vars.get(key, tk.BooleanVar()).get():
                        continue
                    style = self.orbital_styles.get(key, {'color': '#7f7f7f', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7})

                    spin_specs = []
                    if self.split_spin_var.get():
                        for sn, sl in [('α', 'alpha'), ('β', 'beta')]:
                            do_spin = (sn == 'α' and self.show_spin_alpha.get()) or (sn == 'β' and self.show_spin_beta.get())
                            if do_spin and sl in ad:
                                spin_specs.append((sl, '', f" ({sn})"))
                    else:
                        spin_specs.append((None, style['linestyle'], ''))

                    for spin_name, ls_mod, spin_tag in spin_specs:
                        e_src, dos_src = self._extract_pdos(atom, group, shift, ad, spin=spin_name)
                        if e_src is None:
                            continue
                        npts = 2000
                        x_grid = np.linspace(e_min, e_max, npts) if np.isfinite(e_min) else np.linspace(shift - 5, shift + 20, npts)
                        ck = (atom, group, spin_name, round(shift, 3), round(fwhm, 2), is_gauss, self.unocc_only_var.get(), npts, round(x_grid[0], 2), round(x_grid[-1], 2))
                        if ck in self._broad_cache:
                            dos_b = self._broad_cache[ck]
                        else:
                            dos_b = self._broaden_safe(e_src, dos_src, x_grid, fwhm, is_gauss)
                            self._broad_cache[ck] = dos_b
                        all_pdos_curves.append((dos_b, x_grid))

                        if do_norm and np.max(dos_b) > 0:
                            dos_plot = dos_b / np.max(dos_b)
                        else:
                            dos_plot = dos_b

                        if mirror and spin_name == 'beta':
                            dos_plot = -dos_plot
                        if spin_name == 'beta':
                            ls = style.get('beta_linestyle', '--')
                            lw = style.get('beta_linewidth', style['linewidth'])
                            lbl = f"{atom}-{group} (β↓)" if mirror else f"{atom}-{group} (β)"
                        elif spin_name == 'alpha':
                            ls = style['linestyle']
                            lw = style['linewidth']
                            lbl = f"{atom}-{group} (α↑)" if mirror else f"{atom}-{group} (α)"
                        else:
                            ls = style['linestyle']
                            lw = style['linewidth']
                            lbl = f"{atom}-{group}" + spin_tag
                        ax1_pdos.plot(x_grid, dos_plot, color=style['color'], linestyle=ls,
                                      linewidth=lw, alpha=style['alpha'], label=lbl)
                        self._plotted_curves['xas'].append({'label': lbl, 'x': x_grid.copy(), 'y': dos_plot.copy()})

                        if e_min > x_grid[0]: e_min = x_grid[0]
                        if e_max < x_grid[-1]: e_max = x_grid[-1]

        # Dual-axis legend: combine from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_pdos.get_legend_handles_labels()
        if lines1 or lines2:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')
        ax1.set_xlim(e_min, e_max)

        # ---- Subplot 2: TME ----
        ax2 = self.ax_tme
        ax2.clear(); ax2.set_title("Transition Matrix Element", fontsize=11)
        ax2.set_xlabel("Energy (eV)", fontsize=10)
        ax2.set_ylabel("TME (a.u.)", fontsize=10)
        ax2.grid(True, alpha=0.25, linestyle='--')

        if self.xas_energy is not None and self.pdos_data and self.show_tme.get():
            tme_sum = None
            tme_e = None
            for atom, ad in self.pdos_data['atoms'].items():
                key = f"{atom}_{allowed_orb}"
                if not self.orbital_vars.get(key, tk.BooleanVar()).get():
                    continue
                e_src, dos_src = self._extract_pdos(atom, allowed_orb, shift, ad)
                if e_src is not None:
                    if tme_sum is None:
                        tme_sum = dos_src.copy()
                        tme_e = e_src
                    else:
                        if len(e_src) != len(tme_e):
                            lo = max(tme_e.min(), e_src.min())
                            hi = min(tme_e.max(), e_src.max())
                            n = min(len(tme_e), len(e_src))
                            en = np.linspace(lo, hi, n)
                            tme_sum = np.interp(en, tme_e, tme_sum)
                            dos_src = np.interp(en, e_src, dos_src)
                            tme_e = en
                        tme_sum += dos_src

            if tme_sum is not None:
                npts = 2000
                x_grid = np.linspace(e_min, e_max, npts)
                dos_b = self._broaden_safe(tme_e, tme_sum, x_grid, fwhm, is_gauss)
                xas_i = np.interp(x_grid, self.xas_energy, self.xas_intensity)
                if do_norm and np.max(xas_i) > 0:
                    xas_i = xas_i / np.max(xas_i)
                if do_norm and np.max(dos_b) > 0:
                    dos_b = dos_b / np.max(dos_b)
                pdos_s = np.clip(dos_b, 1e-10, None)
                tme = xas_i / pdos_s
                tme[np.isinf(tme)] = 0
                cap = np.percentile(tme[tme > 0], 98) if np.any(tme > 0) else 1
                tme = np.clip(tme, 0, cap)
                ax2.plot(x_grid, tme, 'r-', linewidth=1.5)
                self._plotted_curves['tme'].append({'label': 'TME', 'x': x_grid.copy(), 'y': tme.copy()})
                ax2.fill_between(x_grid, 0, tme, color='red', alpha=0.1)
                ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.4)
                ax2.set_xlim(e_min, e_max)

        # ---- Subplot 3: Hybridization ----
        ax3 = self.ax_hyb
        ax3.clear(); ax3.set_title("Orbital Hybridization Analysis", fontsize=11)
        ax3.set_xlabel("Energy (eV)", fontsize=10)
        if do_norm:
            ax3.set_ylabel("PDOS (normalized)", fontsize=10)
        else:
            ax3.set_ylabel("PDOS (states/eV)", fontsize=10)
        ax3.grid(True, alpha=0.25, linestyle='--')

        # Fermi level marker
        ax3.axvline(e_fermi_shifted, color='black', linestyle=':', alpha=0.6, linewidth=1.0, label=f'Fermi(E={e_fermi_shifted:.1f})')

        if self.pdos_data:
            for key, var in self.orbital_vars.items():
                if not var.get():
                    continue
                parts = key.split('_', 1)
                atom, group = parts[0], parts[1] if len(parts) > 1 else ''
                if atom not in self.pdos_data['atoms'] or group not in VALID_HYB_GROUPS:
                    continue
                style = self.orbital_styles.get(key, {'color': '#7f7f7f', 'alpha': 0.7})

                spin_specs = []
                if self.split_spin_var.get():
                    for sn, sl in [('α', 'alpha'), ('β', 'beta')]:
                        do_spin = (sn == 'α' and self.show_spin_alpha.get()) or (sn == 'β' and self.show_spin_beta.get())
                        if do_spin and sl in self.pdos_data['atoms'][atom]:
                            spin_specs.append((sl, '', f" ({sn})"))
                else:
                    spin_specs.append((None, style['linestyle'], ''))

                for spin_name, ls_mod, spin_tag in spin_specs:
                    e_src, dos_src = self._extract_pdos(atom, group, shift, self.pdos_data['atoms'][atom], spin=spin_name)
                    if e_src is None:
                        continue
                    npts = 2000
                    x_grid = np.linspace(e_min, e_max, npts)
                    ck = (atom, group, spin_name, round(shift, 3), round(fwhm, 2), is_gauss, self.unocc_only_var.get(), npts, round(x_grid[0], 2), round(x_grid[-1], 2))
                    if ck in self._broad_cache:
                        dos_b = self._broad_cache[ck]
                    else:
                        dos_b = self._broaden_safe(e_src, dos_src, x_grid, fwhm, is_gauss)
                        self._broad_cache[ck] = dos_b
                    if do_norm and np.max(dos_b) > 0:
                        dos_b = dos_b / np.max(dos_b)
                    if spin_name == 'beta':
                        if mirror:
                            dos_b = -dos_b
                        ls_eff = style.get('beta_linestyle', '--')
                        lw_eff = style.get('beta_linewidth', style['linewidth'])
                        lbl = f"{key} (β↓)" if mirror else f"{key} (β)"
                    elif spin_name == 'alpha':
                        ls_eff = style['linestyle']
                        lw_eff = style['linewidth']
                        lbl = f"{key} (α↑)" if mirror else f"{key} (α)"
                    else:
                        ls_eff = style['linestyle']
                        lw_eff = style['linewidth']
                        lbl = f"{key}" + spin_tag
                    c = style['color']
                    ax3.plot(x_grid, dos_b, linewidth=lw_eff, linestyle=ls_eff, color=c, label=lbl)
                    self._plotted_curves['hyb'].append({'label': lbl, 'x': x_grid.copy(), 'y': dos_b.copy()})
                    ax3.fill_between(x_grid, 0, dos_b, color=c, alpha=0.08)

            if self.xas_e0:
                ax3.axvline(self.xas_e0, color='gray', linestyle='--', alpha=0.5, label=f'E0={self.xas_e0:.1f}')
            ax3.legend(loc='upper left', fontsize='small')
            ax3.set_xlim(e_min, e_max)

        # ---- d-Subplot Rows ----
        if self.show_d_subplot_var.get() and self.pdos_data:
            atom = 'Fex' if 'Fex' in self.pdos_data['atoms'] else ('Fe' if 'Fe' in self.pdos_data['atoms'] else None)
            if atom:
                ad = self.pdos_data['atoms'][atom]
                if mirror:
                    ax_da = self.fig.add_subplot(self.gs[2, :])
                    ax_da.set_title("Fe d-orbital components (α↑)", fontsize=10)
                    ax_da.grid(True, alpha=0.25, linestyle='--')
                    ax_da.axvline(self.xas_e0 if self.xas_e0 else shift, color='gray', ls=':', alpha=0.5, lw=1)
                    ax_db = self.fig.add_subplot(self.gs[3, :])
                    ax_db.set_title("Fe d-orbital components (β↓)", fontsize=10)
                    ax_db.set_xlabel("Energy (eV)", fontsize=9)
                    ax_db.grid(True, alpha=0.25, linestyle='--')
                    ax_db.axvline(self.xas_e0 if self.xas_e0 else shift, color='gray', ls=':', alpha=0.5, lw=1)
                    for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
                        st_key = f"{atom}_{sub}"
                        sty = self.orbital_styles.get(st_key, {})
                        for spin_name, ax_d, sign, ls_attr, lw_attr in \
                            [('alpha', ax_da, 1, 'linestyle', 'linewidth'),
                             ('beta', ax_db, -1, 'beta_linestyle', 'beta_linewidth')]:
                            e_src, dos_src = self._extract_pdos(atom, sub, shift, ad, spin=spin_name)
                            if e_src is None:
                                continue
                            npts = 2000
                            x_grid = np.linspace(e_min, e_max, npts)
                            dos_b = self._broaden_safe(e_src, dos_src, x_grid, fwhm, is_gauss)
                            if do_norm and np.max(dos_b) > 0:
                                dos_b = dos_b / np.max(dos_b)
                            ls = sty.get(ls_attr, ':')
                            lw = sty.get(lw_attr, 1.0)
                            col = sty.get('color', '#7f7f7f')
                            ax_d.plot(x_grid, dos_b * sign, color=col, linestyle=ls,
                                      linewidth=lw, alpha=0.8, label=sub)
                            self._plotted_curves['dcomp'].append({'label': f'{sub} ({spin_name})',
                                                                   'x': x_grid.copy(), 'y': (dos_b * sign).copy()})
                    ax_da.legend(loc='upper right', fontsize='x-small', ncol=2)
                    ax_da.set_xlim(e_min, e_max)
                    ax_db.legend(loc='upper right', fontsize='x-small', ncol=2)
                    ax_db.set_xlim(e_min, e_max)
                else:
                    ax_d = self.fig.add_subplot(self.gs[2, :])
                    ax_d.set_title("Fe d-orbital components", fontsize=10)
                    ax_d.set_xlabel("Energy (eV)", fontsize=9)
                    ax_d.grid(True, alpha=0.25, linestyle='--')
                    ax_d.axvline(self.xas_e0 if self.xas_e0 else shift, color='gray', ls=':', alpha=0.5, lw=1)
                    for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
                        e_src, dos_src = self._extract_pdos(atom, sub, shift, ad)
                        if e_src is None:
                            continue
                        npts = 2000
                        x_grid = np.linspace(e_min, e_max, npts)
                        dos_b = self._broaden_safe(e_src, dos_src, x_grid, fwhm, is_gauss)
                        if do_norm and np.max(dos_b) > 0:
                            dos_b = dos_b / np.max(dos_b)
                        st_key = f"{atom}_{sub}"
                        sty = self.orbital_styles.get(st_key, {})
                        col = sty.get('color', '#7f7f7f')
                        ls = sty.get('linestyle', '-')
                        lw = sty.get('linewidth', 1.5)
                        ax_d.plot(x_grid, dos_b, color=col, linestyle=ls,
                                  linewidth=lw, alpha=0.8, label=sub)
                        self._plotted_curves['dcomp'].append({'label': sub, 'x': x_grid.copy(), 'y': dos_b.copy()})
                    ax_d.legend(loc='upper right', fontsize='x-small', ncol=2)
                    ax_d.set_xlim(e_min, e_max)

        self.canvas.draw_idle()
