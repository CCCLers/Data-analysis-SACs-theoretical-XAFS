import os, re, glob, logging
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinterdnd2 import DND_FILES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec

import pdos_tool as es
from pdos_gs_tool import parse_gs_pdos, parse_gs_pdos_files
import export_utils as xport

logger = logging.getLogger(__name__)

_ORBITAL_GROUPS = es._ORBITAL_GROUPS
_SELECTION_RULES = es._SELECTION_RULES


def parse_es_pdos(directory):
    files = sorted(glob.glob(os.path.join(directory, '*-x-*.pdos')))
    return es.parse_pdos_file_list(files)


class GSvsESWindow:
    def __init__(self, master):
        self.master = master
        master.title("GS vs ES PDOS Comparison")
        master.geometry("1400x900")

        self.gs_data = None
        self.es_data = None
        self.xas_energy = None
        self.xas_intensity = None
        self.xas_e0 = None

        self.fwhm_var = tk.DoubleVar(value=0.8)
        self.is_gaussian_var = tk.BooleanVar(value=False)
        self.energy_shift_gs = tk.DoubleVar(value=0.0)
        self.energy_shift_es = tk.DoubleVar(value=0.0)
        self.show_xas = tk.BooleanVar(value=True)
        self.unocc_only_gs = tk.BooleanVar(value=True)
        self.normalize_var = tk.BooleanVar(value=True)
        self.edge_var = tk.StringVar(value='K')
        self.split_spin_var = tk.BooleanVar(value=True)
        self.show_d_subplot_var = tk.BooleanVar(value=False)
        self.mirror_spin_var = tk.BooleanVar(value=False)
        self.show_spin_alpha = tk.BooleanVar(value=True)
        self.show_spin_beta = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=False)

        self.gs_orbital_vars = {}
        self.es_orbital_vars = {}
        self.gs_orbital_styles = {}
        self.es_orbital_styles = {}
        self.show_d_components_gs_var = tk.BooleanVar(value=False)
        self.show_d_components_es_var = tk.BooleanVar(value=False)
        self._plotted_curves = {}
        self.gs_base_shift = 0.0

        self._setup_ui()

    def _setup_ui(self):
        main_panel = ttk.Frame(self.master)
        main_panel.pack(fill=tk.BOTH, expand=True)

        ctrl_frame = ttk.LabelFrame(main_panel, text="Controls", width=340)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        ctrl_frame.pack_propagate(False)

        scroll_cv = tk.Canvas(ctrl_frame, highlightthickness=0, width=330)
        scroll_sb = ttk.Scrollbar(ctrl_frame, orient="vertical", command=scroll_cv.yview)
        scroll_frame = ttk.Frame(scroll_cv)
        scroll_frame.bind("<Configure>", lambda e: scroll_cv.configure(scrollregion=scroll_cv.bbox("all")))
        scroll_cv.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_cv.configure(yscrollcommand=scroll_sb.set)
        scroll_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_sb.pack(side=tk.RIGHT, fill=tk.Y)
        def _on_mw(event):
            scroll_cv.yview_scroll(int(-event.delta / 120), "units")
        def _bw(event):
            scroll_cv.bind_all("<MouseWheel>", _on_mw)
        def _uw(event):
            scroll_cv.unbind_all("<MouseWheel>")
        scroll_cv.bind("<Enter>", _bw)
        scroll_cv.bind("<Leave>", _uw)

        plot_frame = ttk.Frame(main_panel)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # XAS
        xas_sec = ttk.LabelFrame(scroll_frame, text="XAS Reference", padding=5)
        xas_sec.pack(fill=tk.X, pady=2)
        self.lbl_xas = ttk.Label(xas_sec, text="No XAS loaded (optional)", foreground="gray")
        self.lbl_xas.pack(fill=tk.X)
        xas_btn_frame = ttk.Frame(xas_sec)
        xas_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(xas_btn_frame, text="Load CSV", command=self._load_xas_csv).pack(side=tk.LEFT, padx=2)

        # GS PDOS
        gs_sec = ttk.LabelFrame(scroll_frame, text="GS PDOS (charge)", padding=5)
        gs_sec.pack(fill=tk.X, pady=2)
        self.lbl_gs = ttk.Label(gs_sec, text="No GS PDOS loaded", foreground="gray")
        self.lbl_gs.pack(fill=tk.X)
        gs_btn_frame = ttk.Frame(gs_sec)
        gs_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(gs_btn_frame, text="Load Directory", command=lambda: self._load_pdos('gs')).pack(side=tk.LEFT, padx=2)
        ttk.Button(gs_btn_frame, text="Load Files", command=lambda: self._load_pdos_files('gs')).pack(side=tk.LEFT, padx=2)
        self.gs_drop = tk.Text(gs_sec, height=1, relief="solid", bd=1)
        self.gs_drop.pack(fill=tk.X, pady=1)
        self.gs_drop.insert(tk.END, "Drop GS PDOS")
        self.gs_drop.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.gs_drop.dnd_bind('<<Drop>>', lambda e: self._on_pdos_drop(e, 'gs'))  # type: ignore[attr-defined]
        self.gs_drop.config(state="disabled")

        ttk.Label(gs_sec, text="GS Fine Shift:", font=("Arial", 8)).pack(anchor=tk.W)
        gs_shift_row = ttk.Frame(gs_sec)
        gs_shift_row.pack(fill=tk.X)
        ent_gs = ttk.Entry(gs_shift_row, textvariable=self.energy_shift_gs, width=6)
        ent_gs.pack(side=tk.LEFT, padx=2)
        ttk.Label(gs_shift_row, text="eV").pack(side=tk.LEFT)
        ent_gs.bind('<Return>', lambda e: self._update_plot())

        # GS Orbitals
        gs_orb_sec = ttk.LabelFrame(scroll_frame, text="GS Orbitals", padding=5)
        gs_orb_sec.pack(fill=tk.X, pady=(2, 0))
        gs_d_toggle = ttk.Frame(gs_orb_sec)
        gs_d_toggle.pack(fill=tk.X, pady=(0, 2))
        self.btn_d_gs = ttk.Button(gs_d_toggle, text="\u25b6 d components",
                                   command=lambda: self._toggle_d_components('gs'), width=16)
        self.btn_d_gs.pack(side=tk.LEFT)
        self.gs_orbital_frame = ttk.Frame(gs_orb_sec)
        self.gs_orbital_frame.pack(fill=tk.X)

        # ES PDOS
        es_sec = ttk.LabelFrame(scroll_frame, text="ES PDOS (xas)", padding=5)
        es_sec.pack(fill=tk.X, pady=2)
        self.lbl_es = ttk.Label(es_sec, text="No ES PDOS loaded", foreground="gray")
        self.lbl_es.pack(fill=tk.X)
        es_btn_frame = ttk.Frame(es_sec)
        es_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(es_btn_frame, text="Load Directory", command=lambda: self._load_pdos('es')).pack(side=tk.LEFT, padx=2)
        ttk.Button(es_btn_frame, text="Load Files", command=lambda: self._load_pdos_files('es')).pack(side=tk.LEFT, padx=2)
        self.es_drop = tk.Text(es_sec, height=1, relief="solid", bd=1)
        self.es_drop.pack(fill=tk.X, pady=1)
        self.es_drop.insert(tk.END, "Drop ES PDOS")
        self.es_drop.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.es_drop.dnd_bind('<<Drop>>', lambda e: self._on_pdos_drop(e, 'es'))  # type: ignore[attr-defined]
        self.es_drop.config(state="disabled")

        ttk.Label(es_sec, text="ES Fine Shift:", font=("Arial", 8)).pack(anchor=tk.W)
        es_shift_row = ttk.Frame(es_sec)
        es_shift_row.pack(fill=tk.X)
        ent_es = ttk.Entry(es_shift_row, textvariable=self.energy_shift_es, width=6)
        ent_es.pack(side=tk.LEFT, padx=2)
        ttk.Label(es_shift_row, text="eV").pack(side=tk.LEFT)
        ent_es.bind('<Return>', lambda e: self._update_plot())

        # ES Orbitals
        es_orb_sec = ttk.LabelFrame(scroll_frame, text="ES Orbitals", padding=5)
        es_orb_sec.pack(fill=tk.X, pady=(2, 0))
        es_d_toggle = ttk.Frame(es_orb_sec)
        es_d_toggle.pack(fill=tk.X, pady=(0, 2))
        self.btn_d_es = ttk.Button(es_d_toggle, text="\u25b6 d components",
                                   command=lambda: self._toggle_d_components('es'), width=16)
        self.btn_d_es.pack(side=tk.LEFT)
        self.es_orbital_frame = ttk.Frame(es_orb_sec)
        self.es_orbital_frame.pack(fill=tk.X)

        # === GS vs ES Alignment ===
        align_sec = ttk.LabelFrame(scroll_frame, text="GS / ES Alignment", padding=5)
        align_sec.pack(fill=tk.X, pady=2)
        self.align_mode_var = tk.StringVar(value='Fermi_E0')
        align_modes = [
            ('Fermi_E0', 'Fermi → E0 (default)'),
            ('Fe_p_Fe_p', 'Fe-p → Fe-p (white line)'),
        ]
        for val, txt in align_modes:
            rb = ttk.Radiobutton(align_sec, text=txt, variable=self.align_mode_var,
                                 value=val, command=self._auto_align)
            rb.pack(anchor=tk.W, padx=10)
        align_btn_frame = ttk.Frame(align_sec)
        align_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(align_btn_frame, text="Apply Alignment", command=self._auto_align).pack(side=tk.LEFT, padx=2)
        self.lbl_align_info = ttk.Label(align_sec, text="GS base shift: 0.0 eV", foreground="gray",
                                        font=("Arial", 8), wraplength=320)
        self.lbl_align_info.pack(anchor=tk.W, padx=2, fill=tk.X)

        # Options
        opt_sec = ttk.LabelFrame(scroll_frame, text="Display Options", padding=5)
        opt_sec.pack(fill=tk.X, pady=2)
        ttk.Label(opt_sec, text="FWHM:").pack(side=tk.LEFT)
        ent_fwhm = ttk.Entry(opt_sec, textvariable=self.fwhm_var, width=5)
        ent_fwhm.pack(side=tk.LEFT, padx=2)
        ent_fwhm.bind('<Return>', lambda e: self._update_plot())
        ttk.Checkbutton(opt_sec, text="Gauss", variable=self.is_gaussian_var, command=self._update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(opt_sec, text="Show XAS", variable=self.show_xas, command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(opt_sec, text="Normalize [0,1]", variable=self.normalize_var, command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(opt_sec, text="Split α/β", variable=self.split_spin_var, command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(opt_sec, text="GS Unocc Only", variable=self.unocc_only_gs, command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(opt_sec, text="d-Subplot Row", variable=self.show_d_subplot_var,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(opt_sec, text="Mirror β Spin (up/down)", variable=self.mirror_spin_var,
                        command=self._update_plot).pack(anchor=tk.W)
        self.show_legend_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_sec, text="Show Legend", variable=self.show_legend_var,
                        command=self._update_plot).pack(anchor=tk.W)

        # Edge
        edge_row = ttk.Frame(opt_sec)
        edge_row.pack(fill=tk.X)
        ttk.Label(edge_row, text="Edge:").pack(side=tk.LEFT)
        edge_cb = ttk.Combobox(edge_row, textvariable=self.edge_var, values=['K', 'L3'], state="readonly", width=3)
        edge_cb.pack(side=tk.LEFT, padx=3)
        edge_cb.bind("<<ComboboxSelected>>", lambda e: self._update_plot())

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
        dcomp_btn_row = ttk.Frame(dcomp_sec)
        dcomp_btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(dcomp_btn_row, text="Compute", command=self._compute_d_comp).pack(side=tk.LEFT, padx=2)
        self._dcomp_text = tk.Text(dcomp_sec, height=9, width=38, font=("Consolas", 8), wrap=tk.NONE, state="disabled")
        self._dcomp_text.pack(fill=tk.X, pady=2)

        # Status
        self.lbl_status = ttk.Label(scroll_frame, text="Load GS and ES PDOS datasets", foreground="gray",
                                    font=("Arial", 8), wraplength=320)
        self.lbl_status.pack(fill=tk.X, padx=5, pady=5)

        # Plot area
        self.fig = plt.Figure(figsize=(12, 15), dpi=100)
        self.gs_fig = GridSpec(3, 1, figure=self.fig, height_ratios=[1, 1, 1], hspace=0.40)
        self.ax_gs = self.fig.add_subplot(self.gs_fig[0])
        self.ax_es = self.fig.add_subplot(self.gs_fig[1])
        self.ax_cmp = self.fig.add_subplot(self.gs_fig[2])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame)
        self._init_labels()

    def _init_labels(self):
        for ax, t in [
            (self.ax_gs, "Ground-State PDOS (aligned to XAS E0)"),
            (self.ax_es, "Excited-State Core PDOS"),
            (self.ax_cmp, "GS vs ES Comparison"),
        ]:
            ax.set_title(t, fontsize=10)
            ax.set_xlabel("Energy rel. XAS E0 (eV)", fontsize=9)
            ax.set_ylabel("PDOS (norm)", fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='--')

    def _load_xas_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if fp:
            self._read_xas_csv(fp)

    def _read_xas_csv(self, filepath):
        e, y, n = es._robust_load_csv(filepath)
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
        self.lbl_xas.config(text=f"XAS: {os.path.basename(filepath)} (E0={self.xas_e0:.1f})", foreground="black")
        self._auto_align()
        self._update_plot()
        if hasattr(self.master, 'lift'):
            self.master.lift()

    def _find_peak(self, data_key, atom, group, unocc_only=False, e_range=None):
        data = self.gs_data if data_key == 'gs' else self.es_data
        if data is None or atom not in data['atoms']:
            return None
        ad = data['atoms'][atom]
        spin = 'alpha' if 'alpha' in ad else ('beta' if 'beta' in ad else None)
        if spin is None:
            return None
        e_src = ad[spin]['energy'].copy()
        dos_src = ad[spin]['groups'].get(group, np.zeros_like(e_src)).copy()
        if unocc_only:
            mask = ad[spin]['occupation'] < 0.5
            e_src = e_src[mask]
            dos_src = dos_src[mask]
        if e_range is not None:
            mask = (e_src >= e_range[0]) & (e_src <= e_range[1])
            if mask.sum() < 3:
                return None
            e_src = e_src[mask]
            dos_src = dos_src[mask]
        if len(e_src) < 3:
            return None
        fwhm = self.fwhm_var.get()
        is_g = self.is_gaussian_var.get()
        e_min, e_max = e_src.min(), e_src.max()
        pad = (e_max - e_min) * 0.1 or 2.0
        xg = np.linspace(e_min - pad, e_max + pad, 2000)
        dos = self._broaden_safe(e_src, dos_src, xg, fwhm, is_g)
        return xg[np.argmax(dos)]

    def _auto_align(self):
        mode = self.align_mode_var.get()
        if self.xas_e0 is None:
            self.lbl_align_info.config(text="Load XAS first to set alignment reference", foreground="red")
            return

        es_atom = 'Fex' if self.es_data and 'Fex' in self.es_data['atoms'] else \
                  ('Fe' if self.es_data and 'Fe' in self.es_data['atoms'] else None)
        gs_atom = 'Fe' if self.gs_data and 'Fe' in self.gs_data['atoms'] else None
        base_shift = self.xas_e0

        if mode == 'Fermi_E0':
            self.gs_base_shift = base_shift
            info = f"Fermi→E0: GS Fermi → XAS E0 (= {self.xas_e0:.2f} eV)"

        elif mode == 'Fe_p_Fe_p':
            if gs_atom is None or es_atom is None:
                self.lbl_align_info.config(text="Need Fe in GS + Fex in ES for p-peak alignment", foreground="red")
                return
            xanes_lo = self.xas_energy[0] if self.xas_energy is not None else self.xas_e0 - 25
            xanes_hi = self.xas_energy[-1] if self.xas_energy is not None else self.xas_e0 + 25
            gs_p = self._find_peak('gs', gs_atom, 'p', unocc_only=True)
            es_p_abs = self._find_peak('es', es_atom, 'p', e_range=(xanes_lo, xanes_hi))
            if gs_p is None or es_p_abs is None:
                self.lbl_align_info.config(text="Cannot find Fe-p peaks in XANES window", foreground="red")
                return
            base_shift = es_p_abs - gs_p
            self.gs_base_shift = base_shift
            es_p_rel = es_p_abs - self.xas_e0
            info = f"Fe-p→Fe-p: ΔE={base_shift:.2f} eV (GS p@+{gs_p:.2f} eV → ES p@{es_p_rel:+.2f} eV rel E0)"

        self.energy_shift_gs.set(0.0)
        self.lbl_align_info.config(text=info, foreground="black")
        self._update_plot()

    def _parse_dropped_files(self, event_data):
        pattern = r'\{.*?\}|\S+'
        matches = re.findall(pattern, event_data)
        return [m.strip('{}') for m in matches]

    def _on_pdos_drop(self, event, mode):
        files = self._parse_dropped_files(event.data)
        if not files:
            return
        pdos_files = [f for f in files if f.lower().endswith('.pdos')]
        dirs = [f for f in files if os.path.isdir(f)]
        if dirs:
            self._load_pdos(mode, directory=dirs[0])
        elif pdos_files:
            self._load_pdos_files(mode, files=pdos_files)

    def _load_pdos(self, mode, directory=None):
        if mode == 'gs':
            d = directory or filedialog.askdirectory(title="Select GS PDOS folder")
            if not d:
                return
            data = parse_gs_pdos(d)
            if data is None:
                self.lbl_gs.config(text="No GS PDOS found", foreground="red")
                return
            self.gs_data = data
            atoms = list(data['atoms'].keys())
            self.lbl_gs.config(text=f"GS: {', '.join(atoms)} | Fermi={data['fermi_ev']:.2f} eV", foreground="black")
            logger.info(f"GS PDOS: atoms={atoms}")
        else:
            d = directory or filedialog.askdirectory(title="Select ES PDOS folder")
            if not d:
                return
            data = parse_es_pdos(d)
            if data is None:
                self.lbl_es.config(text="No ES PDOS found", foreground="red")
                return
            self.es_data = data
            atoms = list(data['atoms'].keys())
            self.lbl_es.config(text=f"ES: {', '.join(atoms)} | Fermi={data['fermi_ev']:.2f} eV", foreground="black")
            logger.info(f"ES PDOS: atoms={atoms}")
        self._rebuild_orbital_controls(mode)
        self._update_status()
        self._update_plot()
        self.master.lift()

    def _load_pdos_files(self, mode, files=None):
        if mode == 'gs':
            fps = files or filedialog.askopenfilenames(title="Select *charge*.pdos files", filetypes=[("PDOS files", "*.pdos")])
            if not fps:
                return
            data = parse_gs_pdos_files(list(fps))
            if data is None:
                self.lbl_gs.config(text="No valid GS PDOS", foreground="red")
                return
            self.gs_data = data
            atoms = list(data['atoms'].keys())
            self.lbl_gs.config(text=f"GS: {', '.join(atoms)} | Fermi={data['fermi_ev']:.2f} eV", foreground="black")
        else:
            fps = files or filedialog.askopenfilenames(title="Select *-x-*.pdos files", filetypes=[("PDOS files", "*.pdos")])
            if not fps:
                return
            filtered = [f for f in fps if '-x-' in os.path.basename(f)]
            data = es.parse_pdos_file_list(filtered)
            if data is None:
                self.lbl_es.config(text="No valid ES PDOS", foreground="red")
                return
            self.es_data = data
            atoms = list(data['atoms'].keys())
            self.lbl_es.config(text=f"ES: {', '.join(atoms)} | Fermi={data['fermi_ev']:.2f} eV", foreground="black")
        self._rebuild_orbital_controls(mode)
        self._update_status()
        self._update_plot()
        self.master.lift()

    @staticmethod
    def _norm_atom(atom):
        if atom in ('Fex',): return 'Fe'
        if atom in ('Cos',): return 'Co'
        if atom in ('Nis',): return 'Ni'
        return atom

    def _rebuild_orbital_controls(self, mode):
        data = self.gs_data if mode == 'gs' else self.es_data
        vars_dict = self.gs_orbital_vars if mode == 'gs' else self.es_orbital_vars
        styles_dict = self.gs_orbital_styles if mode == 'gs' else self.es_orbital_styles
        frame = self.gs_orbital_frame if mode == 'gs' else self.es_orbital_frame
        show_d = (self.show_d_components_gs_var if mode == 'gs' else self.show_d_components_es_var).get()

        for w in frame.winfo_children():
            w.destroy()

        if not data:
            return

        palette_idx = 0
        needed = set()
        for atom, ad in data['atoms'].items():
            spin_key = 'alpha' if 'alpha' in ad else ('beta' if 'beta' in ad else None)
            if not spin_key:
                continue
            avail_groups = list(ad[spin_key]['groups'].keys())
            for g in avail_groups:
                if g in es._D_SUBGROUP_NAMES and not show_d:
                    continue
                key = f"{atom}_{g}"
                needed.add(key)

        # Remove stale keys
        for k in list(vars_dict.keys()):
            if k not in needed:
                del vars_dict[k]
        for k in list(styles_dict.keys()):
            if k not in needed:
                del styles_dict[k]

        for atom, ad in data['atoms'].items():
            spin_key = 'alpha' if 'alpha' in ad else ('beta' if 'beta' in ad else None)
            if not spin_key:
                continue
            avail_groups = list(ad[spin_key]['groups'].keys())
            for g in avail_groups:
                if g in es._D_SUBGROUP_NAMES and not show_d:
                    continue
                key = f"{atom}_{g}"
                default_on = (g == 'p') or (atom in ('Fe', 'Fex', 'Co', 'Ni', 'Mn', 'Cu', 'Ti', 'Zn') and g == 'd')
                if g in es._D_SUBGROUP_NAMES:
                    default_on = False

                if key not in vars_dict:
                    vars_dict[key] = tk.BooleanVar(value=default_on)

                if key not in styles_dict:
                    # Try matching the other side (GS→ES or ES→GS) for same-base color
                    base_key = f"{self._norm_atom(atom)}_{g}"
                    side = 'es' if mode == 'gs' else 'gs'
                    other_dict = self.es_orbital_styles if mode == 'gs' else self.gs_orbital_styles
                    matched_key = None
                    for ok in other_dict:
                        if self._norm_atom(ok.split('_')[0]) == self._norm_atom(atom) and ok.endswith(f'_{g}'):
                            matched_key = ok
                            break
                    if matched_key and matched_key in other_dict:
                        styles_dict[key] = dict(other_dict[matched_key])
                    else:
                        color, ls = es._CB_PALETTE[palette_idx % len(es._CB_PALETTE)]
                        lw = 1.5
                        styles_dict[key] = {
                            'color': color, 'linestyle': ls, 'linewidth': lw, 'alpha': 0.8,
                            'beta_linestyle': ls, 'beta_linewidth': lw,
                        }
                        palette_idx += 1

                # Also copy style from the other side if it exists as an inherited style
                # This already covered above

                st = styles_dict[key]
                row = ttk.Frame(frame)
                row.pack(fill=tk.X, pady=1)
                cb = ttk.Checkbutton(row, text=f"{atom}-{g}", variable=vars_dict[key],
                                     command=self._update_plot, width=8)
                cb.pack(side=tk.LEFT)
                btn_color = tk.Button(row, text='\u25a0', fg=st['color'],
                                      width=2, bd=1, relief='ridge',
                                      command=lambda k=key: self._pick_orbital_color(mode, k))
                btn_color.pack(side=tk.LEFT, padx=1)

                ttk.Label(row, text="\u03b1:", font=("Arial", 7)).pack(side=tk.LEFT)
                cb_ls_a = ttk.Combobox(row, values=['-', '--', '-.', ':'], width=3, state='readonly')
                cb_ls_a.set(st['linestyle'])
                cb_ls_a.bind('<<ComboboxSelected>>', lambda e, k=key: self._set_orbital_style(mode, k, 'linestyle', e.widget.get()))
                cb_ls_a.pack(side=tk.LEFT, padx=1)
                sb_lw_a = ttk.Spinbox(row, from_=0.5, to=5.0, increment=0.5, width=3)
                sb_lw_a.set(st['linewidth'])
                sb_lw_a.bind('<Return>', lambda e, k=key, w=sb_lw_a: self._set_orbital_style(mode, k, 'linewidth', float(w.get())))
                sb_lw_a.bind('<FocusOut>', lambda e, k=key, w=sb_lw_a: self._set_orbital_style(mode, k, 'linewidth', float(w.get())))
                sb_lw_a.pack(side=tk.LEFT, padx=1)

                ttk.Label(row, text="\u03b2:", font=("Arial", 7)).pack(side=tk.LEFT, padx=(4, 0))
                cb_ls_b = ttk.Combobox(row, values=['-', '--', '-.', ':'], width=3, state='readonly')
                cb_ls_b.set(st.get('beta_linestyle', '--'))
                cb_ls_b.bind('<<ComboboxSelected>>', lambda e, k=key: self._set_orbital_style(mode, k, 'beta_linestyle', e.widget.get()))
                cb_ls_b.pack(side=tk.LEFT, padx=1)
                sb_lw_b = ttk.Spinbox(row, from_=0.5, to=5.0, increment=0.5, width=3)
                sb_lw_b.set(st.get('beta_linewidth', 1.0))
                sb_lw_b.bind('<Return>', lambda e, k=key, w=sb_lw_b: self._set_orbital_style(mode, k, 'beta_linewidth', float(w.get())))
                sb_lw_b.bind('<FocusOut>', lambda e, k=key, w=sb_lw_b: self._set_orbital_style(mode, k, 'beta_linewidth', float(w.get())))
                sb_lw_b.pack(side=tk.LEFT, padx=1)

    def _toggle_d_components(self, mode):
        var = self.show_d_components_gs_var if mode == 'gs' else self.show_d_components_es_var
        btn = self.btn_d_gs if mode == 'gs' else self.btn_d_es
        new = not var.get()
        var.set(new)
        btn.config(text="\u25bc d components" if new else "\u25b6 d components")
        self._rebuild_orbital_controls(mode)
        self._update_plot()

    def _pick_orbital_color(self, mode, key):
        styles_dict = self.gs_orbital_styles if mode == 'gs' else self.es_orbital_styles
        curr = styles_dict.get(key, {}).get('color', '#000000')
        from tkinter import colorchooser
        res = colorchooser.askcolor(initialcolor=curr, title=f"Color for {key}")
        if res and res[1]:
            styles_dict[key]['color'] = res[1]
            self._update_plot()

    def _set_orbital_style(self, mode, key, attr, value):
        styles_dict = self.gs_orbital_styles if mode == 'gs' else self.es_orbital_styles
        if key not in styles_dict:
            styles_dict[key] = {}
        styles_dict[key][attr] = value
        self._update_plot()

    def _update_status(self):
        parts = []
        if self.gs_data:
            parts.append("GS loaded")
        if self.es_data:
            parts.append("ES loaded")
        if self.xas_e0:
            parts.append(f"XAS E0={self.xas_e0:.1f}")
        self.lbl_status.config(text=" | ".join(parts) if parts else "Load PDOS datasets", foreground="black" if parts else "gray")

    def _extract_pdos(self, data_key, atom, group, spin_filter=None, unocc_only=False, shift=0.0):
        data = self.gs_data if data_key == 'gs' else self.es_data
        if data is None or atom not in data['atoms']:
            return None, None
        ad = data['atoms'][atom]

        def _one_spin(sp):
            if sp not in ad:
                return None, None
            e_r = ad[sp]['energy'].copy()
            d_r = ad[sp]['groups'].get(group, np.zeros_like(e_r)).copy()
            if unocc_only:
                mask = ad[sp]['occupation'] < 0.5
                e_r = e_r[mask]
                d_r = d_r[mask]
            return e_r + shift, d_r

        if spin_filter is not None:
            do_spin = (spin_filter == 'alpha' and self.show_spin_alpha.get()) or \
                      (spin_filter == 'beta' and self.show_spin_beta.get())
            if not do_spin:
                return None, None
            return _one_spin(spin_filter)

        dos_total = None
        e_src = None
        for s in ['alpha', 'beta']:
            do_spin = (s == 'alpha' and self.show_spin_alpha.get()) or (s == 'beta' and self.show_spin_beta.get())
            if not do_spin or s not in ad:
                continue
            e, d = _one_spin(s)
            if e is None:
                continue
            if dos_total is None:
                dos_total = d
                e_src = e
            else:
                if len(e) != len(e_src) or not np.allclose(e_src, e, atol=0.01):
                    lo, hi = max(e_src.min(), e.min()), min(e_src.max(), e.max())
                    n = min(len(e_src), len(e))
                    en = np.linspace(lo, hi, n)
                    dos_total = np.interp(en, e_src, dos_total)
                    d = np.interp(en, e, d)
                    e_src = en
                dos_total += d
        return (e_src, dos_total) if dos_total is not None else (None, None)

    def _broaden_safe(self, energy, dos, x_grid, fwhm, is_gauss):
        if energy is None or len(energy) < 2:
            return np.zeros_like(x_grid)
        return es.broaden_pdos(energy, dos, x_grid, fwhm, is_gauss)

    def _update_plot(self, *args):
        if self.gs_data is None and self.es_data is None:
            return

        fwhm = self.fwhm_var.get()
        is_gauss = self.is_gaussian_var.get()
        do_norm = self.normalize_var.get()
        gs_shift = self.gs_base_shift + self.energy_shift_gs.get()
        es_shift = self.energy_shift_es.get()
        edge = self.edge_var.get()
        allowed_orb = _SELECTION_RULES.get(edge, {}).get('allowed', 'p')
        mirror = self.mirror_spin_var.get()

        self.fig.clear()
        self._plotted_curves = {'gs': [], 'es': [], 'cmp': [], 'dcomp': []}
        show_d = self.show_d_subplot_var.get()
        if show_d and mirror:
            n_rows, ratios = 7, [1, 1, 2.5, 0.5, 0.5, 0.5, 0.5]
        elif show_d:
            n_rows, ratios = 5, [1, 1, 2.5, 0.6, 0.6]
        else:
            n_rows, ratios = 3, [1, 1, 2.5]
        self.gs_fig = GridSpec(n_rows, 1, figure=self.fig, height_ratios=ratios, hspace=0.45)
        self.ax_gs = self.fig.add_subplot(self.gs_fig[0])
        self.ax_es = self.fig.add_subplot(self.gs_fig[1])
        self.ax_cmp = self.fig.add_subplot(self.gs_fig[2])
        self._init_labels()
        if mirror:
            self.ax_gs.set_ylabel("PDOS (α↑ / β↓)", fontsize=9)
            self.ax_es.set_ylabel("PDOS (α↑ / β↓)", fontsize=9)

        if self.xas_energy is not None and self.xas_e0 is not None:
            e_range = (self.xas_energy[0] - self.xas_e0, self.xas_energy[-1] - self.xas_e0)
        else:
            e_range = (-25, 20)

        # === PANEL 1: GS valence PDOS (Fermi=0) ===
        ax1 = self.ax_gs
        ax1.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
        if self.gs_data:
            for atom, ad in self.gs_data['atoms'].items():
                for g in ad.get('alpha', {}).get('groups', {}):
                    if g not in _ORBITAL_GROUPS and g not in es._D_SUBGROUP_NAMES:
                        continue
                    key = f"{atom}_{g}"
                    if not self.gs_orbital_vars.get(key, tk.BooleanVar()).get():
                        continue
                    orb_st = self.gs_orbital_styles.get(key, {})
                    color = orb_st.get('color', '#0072B2')
                    base_ls = orb_st.get('linestyle', ':')
                    base_lw = orb_st.get('linewidth', 1.5)

                    spin_specs = []
                    if self.split_spin_var.get():
                        for sn, sl in [('α', 'alpha'), ('β', 'beta')]:
                            ds = (sn == 'α' and self.show_spin_alpha.get()) or (sn == 'β' and self.show_spin_beta.get())
                            if ds and sl in ad:
                                sp_ls = '-' if sn == 'α' else orb_st.get('beta_linestyle', '--')
                                sp_lw = base_lw if sn == 'α' else orb_st.get('beta_linewidth', base_lw)
                                spin_specs.append((sl, sp_ls, sp_lw, f" ({sn})"))
                    else:
                        spin_specs.append((None, base_ls, base_lw, ''))

                    for sn, ls_mod, lw_mod, st_suf in spin_specs:
                        e, d = self._extract_pdos('gs', atom, g, spin_filter=sn, unocc_only=self.unocc_only_gs.get(), shift=gs_shift)
                        if e is None or len(e) < 2:
                            continue
                        if self.xas_e0:
                            e = e - self.xas_e0
                        xg = np.linspace(e_range[0], e_range[1], 2000)
                        dos = self._broaden_safe(e, d, xg, fwhm, is_gauss)
                        if do_norm and np.max(dos) > 0:
                            dos = dos / np.max(dos)
                        if sn == 'beta' and mirror:
                            dos = -dos
                        ax1.plot(xg, dos, color=color, ls=ls_mod, lw=lw_mod, alpha=0.8, label=key + st_suf)
                        self._plotted_curves['gs'].append({'label': key + st_suf, 'x': xg.copy(), 'y': dos.copy()})

        if self.show_legend_var.get():
            ax1.legend(loc='upper left', fontsize='x-small')
        ax1.autoscale(enable=True, axis='y', tight=False)
        ax1.set_xlim(e_range)

        # === PANEL 2: ES core PDOS (E0=0) ===
        ax2 = self.ax_es
        ax2.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
        if self.es_data:
            for atom, ad in self.es_data['atoms'].items():
                for g in ad.get('alpha', {}).get('groups', {}):
                    if g not in _ORBITAL_GROUPS and g not in es._D_SUBGROUP_NAMES:
                        continue
                    key = f"{atom}_{g}"
                    if not self.es_orbital_vars.get(key, tk.BooleanVar()).get():
                        continue
                    orb_st = self.es_orbital_styles.get(key, {})
                    color = orb_st.get('color', '#E69F00')
                    base_ls = orb_st.get('linestyle', ':')
                    base_lw = orb_st.get('linewidth', 1.5)

                    spin_specs = []
                    if self.split_spin_var.get():
                        for sn, sl in [('α', 'alpha'), ('β', 'beta')]:
                            ds = (sn == 'α' and self.show_spin_alpha.get()) or (sn == 'β' and self.show_spin_beta.get())
                            if ds and sl in ad:
                                sp_ls = '-' if sn == 'α' else orb_st.get('beta_linestyle', '--')
                                sp_lw = base_lw if sn == 'α' else orb_st.get('beta_linewidth', base_lw)
                                spin_specs.append((sl, sp_ls, sp_lw, f" ({sn})"))
                    else:
                        spin_specs.append((None, base_ls, base_lw, ''))

                    for sn, ls_mod, lw_mod, st_suf in spin_specs:
                        e_raw, d_raw = self._extract_pdos('es', atom, g, spin_filter=sn, unocc_only=False, shift=es_shift)
                        if e_raw is None or len(e_raw) < 2:
                            continue
                        if self.xas_e0:
                            e_rel = e_raw - self.xas_e0
                        else:
                            e_rel = e_raw - (e_raw.min() + e_raw.max()) / 2
                        xg = np.linspace(e_range[0], e_range[1], 2000)
                        dos = self._broaden_safe(e_rel, d_raw, xg, fwhm, is_gauss)
                        if do_norm and np.max(dos) > 0:
                            dos = dos / np.max(dos)
                        if sn == 'beta' and mirror:
                            dos = -dos
                        ax2.plot(xg, dos, color=color, ls=ls_mod, lw=lw_mod, alpha=0.8, label=key + st_suf)
                        self._plotted_curves['es'].append({'label': key + st_suf, 'x': xg.copy(), 'y': dos.copy()})

        # XAS overlay on panel 2
        if self.xas_energy is not None and self.xas_e0 is not None and self.show_xas.get():
            xas_rel = self.xas_energy - self.xas_e0
            y = self.xas_intensity
            if do_norm and np.max(y) > 0:
                y = y / np.max(y)
            ax2.plot(xas_rel, y, 'k-', lw=2, alpha=0.6, label='XAS')
            self._plotted_curves['es'].append({'label': 'XAS', 'x': xas_rel.copy(), 'y': y.copy()})
            if mirror:
                ax2.plot(xas_rel, -y, 'k-', lw=2, alpha=0.5, label='XAS (β mirror)')
                self._plotted_curves['es'].append({'label': 'XAS (mirror)', 'x': xas_rel.copy(), 'y': (-y).copy()})

        if self.show_legend_var.get():
            ax2.legend(loc='upper left', fontsize='x-small')
        ax2.autoscale(enable=True, axis='y', tight=False)
        ax2.set_xlim(e_range)

        # === PANEL 3: GS vs ES overlay (matched orbitals) ===
        ax3 = self.ax_cmp
        ax3.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)

        # Build spin specs for Panel 3: (spin_filter, y_sign, label_suffix, ls)
        if self.split_spin_var.get():
            p3_spins = []
            if self.show_spin_alpha.get():
                p3_spins.append(('alpha', 1.0, '', '-'))
            if self.show_spin_beta.get():
                yb = -1.0 if mirror else 1.0
                p3_spins.append(('beta', yb, ' β', '--'))
        else:
            p3_spins = [(None, 1.0, '', '-')]

        if self.gs_data and self.es_data:
            # Build ES-atom → GS-atom lookup by normalized name
            es_to_gs = {}
            for ea in self.es_data['atoms']:
                n = self._norm_atom(ea)
                for ga in self.gs_data['atoms']:
                    if self._norm_atom(ga) == n:
                        es_to_gs[ea] = ga
                        break

            # Iterate GS-enabled keys; find matching ES key (both sides enabled)
            for gs_key, gs_var in self.gs_orbital_vars.items():
                if not gs_var.get():
                    continue
                p = gs_key.rsplit('_', 1)
                if len(p) != 2:
                    continue
                ga, g = p
                if g not in _ORBITAL_GROUPS and g not in es._D_SUBGROUP_NAMES:
                    continue
                # find ES atom paired with ga
                ea = None
                for e_atom in es_to_gs:
                    if es_to_gs[e_atom] == ga:
                        ea = e_atom
                        break
                if ea is None:
                    continue
                es_key = f"{ea}_{g}"
                if not self.es_orbital_vars.get(es_key, tk.BooleanVar()).get():
                    continue

                gs_st = self.gs_orbital_styles.get(gs_key, {})
                color = gs_st.get('color', '#0072B2')
                base_lw = gs_st.get('linewidth', 1.5)

                for sf, y_s, sp_suf, sp_ls in p3_spins:
                    # GS (solid)
                    e_gs, d_gs = self._extract_pdos('gs', ga, g, spin_filter=sf,
                                                     unocc_only=self.unocc_only_gs.get(), shift=gs_shift)
                    if e_gs is not None and len(e_gs) >= 2:
                        if self.xas_e0:
                            e_gs = e_gs - self.xas_e0
                        xg = np.linspace(e_range[0], e_range[1], 2000)
                        dos_gs = self._broaden_safe(e_gs, d_gs, xg, fwhm, is_gauss)
                        if do_norm and np.max(dos_gs) > 0:
                            dos_gs = dos_gs / np.max(dos_gs)
                        lbl = f'GS {ga} {g}{sp_suf}'
                        ax3.plot(xg, dos_gs * y_s, color=color, ls='-', lw=base_lw,
                                 alpha=0.9, label=lbl)
                        self._plotted_curves['cmp'].append({'label': lbl, 'x': xg.copy(), 'y': (dos_gs * y_s).copy()})

                    # ES (dashed, same color)
                    e_es_raw, d_es = self._extract_pdos('es', ea, g, spin_filter=sf,
                                                         unocc_only=False, shift=es_shift)
                    if e_es_raw is not None and len(e_es_raw) >= 2:
                        if self.xas_e0:
                            e_es = e_es_raw - self.xas_e0
                        else:
                            e_es = e_es_raw - (e_es_raw.min() + e_es_raw.max()) / 2
                        xg = np.linspace(e_range[0], e_range[1], 2000)
                        dos_es = self._broaden_safe(e_es, d_es, xg, fwhm, is_gauss)
                        if do_norm and np.max(dos_es) > 0:
                            dos_es = dos_es / np.max(dos_es)
                        lbl = f'ES {ea} {g}{sp_suf}'
                        ax3.plot(xg, dos_es * y_s, color=color, ls='--', lw=base_lw,
                                 alpha=0.9, label=lbl)
                        self._plotted_curves['cmp'].append({'label': lbl, 'x': xg.copy(), 'y': (dos_es * y_s).copy()})

        if self.xas_energy is not None and self.xas_e0 is not None and self.show_xas.get():
            xas_rel = self.xas_energy - self.xas_e0
            y = self.xas_intensity
            if do_norm and np.max(y) > 0:
                y = y / np.max(y)
            ax3.plot(xas_rel, y, 'k-', lw=2, alpha=0.4, label='XAS')
            self._plotted_curves['cmp'].append({'label': 'XAS', 'x': xas_rel.copy(), 'y': y.copy()})
            if mirror:
                ax3.plot(xas_rel, -y, 'k-', lw=2, alpha=0.3, label='XAS (β mirror)')
                self._plotted_curves['cmp'].append({'label': 'XAS (mirror)', 'x': xas_rel.copy(), 'y': (-y).copy()})

        ax3.legend(loc='upper left', fontsize='x-small')
        ax3.autoscale(enable=True, axis='y', tight=False)
        ax3.set_xlim(e_range)

        # Peak shift info
        info_parts = []
        if self.gs_data and self.es_data:
            # find GS Fe ↔ ES Fex pair
            ga = 'Fe' if 'Fe' in self.gs_data['atoms'] else None
            ea = None
            if ga:
                for a in self.es_data['atoms']:
                    if self._norm_atom(a) == self._norm_atom(ga):
                        ea = a
                        break
            if ga and ea:
                for g, lab in [('p', 'Fe-p'), ('d', 'Fe-d')]:
                    if ga in self.gs_data['atoms'] and ea in self.es_data['atoms']:
                        e_gs, d_gs = self._extract_pdos('gs', ga, g, unocc_only=self.unocc_only_gs.get(), shift=gs_shift)
                        e_es_raw, d_es = self._extract_pdos('es', ea, g, unocc_only=False, shift=es_shift)
                        if e_gs is not None and len(e_gs) > 0 and e_es_raw is not None and len(e_es_raw) > 0:
                            if self.xas_e0:
                                e_gs = e_gs - self.xas_e0
                            xg = np.linspace(e_range[0], e_range[1], 2000)
                            dg = self._broaden_safe(e_gs, d_gs, xg, fwhm, is_gauss)
                            e_es = e_es_raw - (self.xas_e0 if self.xas_e0 else 0)
                            de = self._broaden_safe(e_es, d_es, xg, fwhm, is_gauss)
                        if np.max(dg) > 0 and np.max(de) > 0:
                            p_gs = xg[np.argmax(dg)]
                            p_es = xg[np.argmax(de)]
                            info_parts.append(f"{lab}: GS={p_gs:.1f} ES={p_es:.1f} shift={p_es - p_gs:+.1f} eV")

        if info_parts:
            self.lbl_status.config(text=" | ".join(info_parts), foreground="black")

        # ---- d-Subplot Rows ----
        if self.show_d_subplot_var.get():
            gs_atom_es = 'Fex' if self.es_data and 'Fex' in self.es_data['atoms'] else None
            gs_atom_gs = 'Fe' if self.gs_data and 'Fe' in self.gs_data['atoms'] else None

            def _draw_d_components(ax, atom, side, spin_filter=None, unocc=False, y_sign=1.0):
                if atom is None or ax is None:
                    return
                ad = (self.gs_data if side == 'gs' else self.es_data)['atoms'][atom]
                ovars = self.gs_orbital_vars if side == 'gs' else self.es_orbital_vars
                ostyles = self.gs_orbital_styles if side == 'gs' else self.es_orbital_styles
                shift = gs_shift if side == 'gs' else es_shift
                for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
                    if not ovars.get(f"{atom}_{sub}", tk.BooleanVar()).get():
                        continue
                    e, d_raw = self._extract_pdos(side, atom, sub,
                                                  unocc_only=unocc and self.unocc_only_gs.get(),
                                                  shift=shift, spin_filter=spin_filter)
                    if e is None or len(e) < 2:
                        continue
                    if self.xas_e0:
                        e = e - self.xas_e0
                    else:
                        if side != 'gs':
                            e = e - (e.min() + e.max()) / 2
                    xg = np.linspace(e_range[0], e_range[1], 2000)
                    dos = self._broaden_safe(e, d_raw, xg, fwhm, is_gauss)
                    if do_norm and np.max(dos) > 0:
                        dos = dos / np.max(dos)
                    ost = ostyles.get(f"{atom}_{sub}", {})
                    color = ost.get('color', es._CB_PALETTE[('d_xy','d_yz','d_z2','d_xz','d_x2y2').index(sub)][0])
                    ls = ost.get('linestyle', es._CB_PALETTE[('d_xy','d_yz','d_z2','d_xz','d_x2y2').index(sub)][1])
                    ax.plot(xg, dos * y_sign, color=color, linestyle=ls,
                            linewidth=1.5, alpha=0.8, label=sub)
                    lbl_suffix = ''
                    if spin_filter == 'beta':
                        lbl_suffix = ' beta'
                    self._plotted_curves['dcomp'].append({
                        'label': f'{side.upper()} {atom} {sub}{lbl_suffix}',
                        'x': xg.copy(),
                        'y': (dos * y_sign).copy(),
                    })
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(loc='upper right', fontsize='x-small', ncol=2)
                ax.set_xlim(e_range)

            if mirror:
                if gs_atom_gs:
                    ax_d = self.fig.add_subplot(self.gs_fig[3])
                    ax_d.set_title('GS Fe d-orbital components (alpha)', fontsize=9)
                    ax_d.grid(True, alpha=0.2, linestyle='--')
                    ax_d.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
                    _draw_d_components(ax_d, gs_atom_gs, 'gs', spin_filter='alpha', unocc=True, y_sign=1.0)
                if gs_atom_gs:
                    ax_d = self.fig.add_subplot(self.gs_fig[4])
                    ax_d.set_title('GS Fe d-orbital components (beta)', fontsize=9)
                    ax_d.grid(True, alpha=0.2, linestyle='--')
                    ax_d.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
                    _draw_d_components(ax_d, gs_atom_gs, 'gs', spin_filter='beta', unocc=True, y_sign=-1.0)
                if gs_atom_es:
                    ax_d = self.fig.add_subplot(self.gs_fig[5])
                    ax_d.set_title('ES Fe d-orbital components (alpha)', fontsize=9)
                    ax_d.grid(True, alpha=0.2, linestyle='--')
                    ax_d.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
                    _draw_d_components(ax_d, gs_atom_es, 'es', spin_filter='alpha', unocc=False, y_sign=1.0)
                if gs_atom_es:
                    ax_d = self.fig.add_subplot(self.gs_fig[6])
                    ax_d.set_title('ES Fe d-orbital components (beta)', fontsize=9)
                    ax_d.grid(True, alpha=0.2, linestyle='--')
                    ax_d.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
                    _draw_d_components(ax_d, gs_atom_es, 'es', spin_filter='beta', unocc=False, y_sign=-1.0)
            else:
                if gs_atom_gs:
                    ax_d_gs = self.fig.add_subplot(self.gs_fig[3])
                    ax_d_gs.set_title('GS Fe d-orbital components', fontsize=9)
                    ax_d_gs.grid(True, alpha=0.2, linestyle='--')
                    ax_d_gs.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
                    _draw_d_components(ax_d_gs, gs_atom_gs, 'gs', spin_filter=None, unocc=True, y_sign=1.0)
                if gs_atom_es:
                    ax_d_es = self.fig.add_subplot(self.gs_fig[4])
                    ax_d_es.set_title('ES Fe d-orbital components', fontsize=9)
                    ax_d_es.grid(True, alpha=0.2, linestyle='--')
                    ax_d_es.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
                    _draw_d_components(ax_d_es, gs_atom_es, 'es', spin_filter=None, unocc=False, y_sign=1.0)
        self.canvas.draw_idle()

    def _export_csv(self):
        if not any(self._plotted_curves.values()):
            messagebox.showwarning("No Data", "No curves to export.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                          title="Export CSV (base name for multi-panel)")
        if not fp:
            return
        md = xport.build_metadata('pdos_gs_vs_es',
                                  fwhm=self.fwhm_var.get(), is_gaussian=self.is_gaussian_var.get(),
                                  shift=f"GS={self.energy_shift_gs.get():.3f} ES={self.energy_shift_es.get():.3f}",
                                  edge=self.edge_var.get(), normalized=self.normalize_var.get(),
                                  split_spin=self.split_spin_var.get(),
                                  gs_unocc_only=self.unocc_only_gs.get())
        base, ext = os.path.splitext(fp)
        panels = [('gs', 'GS Valence PDOS'), ('es', 'ES Core PDOS'), ('cmp', 'GS vs ES Comparison'), ('dcomp', 'd-Orbital Components')]
        for key, pname in panels:
            if self._plotted_curves.get(key):
                xport.write_panel_csv(f"{base}_{key}{ext}", pname, self._plotted_curves[key], metadata=md)
        self._write_dcomp_csv(base)
        messagebox.showinfo("Done", f"CSV exported:\n  {base}_[gs,es,cmp,dcomp]{ext} + dcomp")

    def _export_image(self):
        fp = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                          title="Export Image")
        if not fp:
            return
        xport.save_figure_png(self.fig, fp)
        messagebox.showinfo("Done", f"Image saved to:\n{fp}")

    def _write_dcomp_csv(self, base_path):
        dcomp_all = []
        for label, data, key_atom in [("GS", self.gs_data, 'Fe'), ("ES", self.es_data, 'Fex')]:
            if data is None or key_atom not in data['atoms']:
                continue
            atom = key_atom
            comp = es._compute_d_composition(data, atom)
            if comp is None:
                continue
            dcomp_all.append((label, atom, comp))
        if not dcomp_all:
            return
        fp = f"{base_path}_dcomp.csv"
        with open(fp, 'w') as f:
            for label, atom, comp in dcomp_all:
                f.write(f"# {label} {atom} d-orbital composition (occ>0.5)\n")
                f.write("component,alpha_percent,beta_percent,total_percent\n")
                for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
                    c = comp[sub]
                    f.write(f"{sub},{c['alpha']:.1f},{c['beta']:.1f},{c['total']:.1f}\n")
                f.write("\n")

    def _compute_d_comp(self):
        gs_lines = []
        es_lines = []
        ncols = 0

        for label, data, key_atom, out in [("GS (occ>0.5)", self.gs_data, 'Fe', gs_lines),
                                             ("ES (all MOs)", self.es_data, 'Fex' if self.es_data and 'Fex' in self.es_data['atoms'] else 'Fe', es_lines)]:
            if data is None or key_atom not in data['atoms']:
                out.append(f"{label}: No data")
                continue
            comp = es._compute_d_composition(data, key_atom)
            if comp is None:
                out.append(f"{label}: No d character")
                continue
            out.append(f"  {label}")
            out.append(f"                  alpha  beta  total")
            for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
                c = comp[sub]
                a_str = f"{c['alpha']:5.1f}" if c['alpha'] > 0 else "   — "
                b_str = f"{c['beta']:5.1f}" if c['beta'] > 0 else "   — "
                t_str = f"{c['total']:5.1f}"
                out.append(f"  {es._D_SUBGROUP_LABELS[sub]:20s} {a_str} {b_str} {t_str}")
            if ncols == 0:
                ncols = len(out) - 1

        # Merge side by side
        merged = []
        gs_lines = gs_lines or [""]
        es_lines = es_lines or [""]
        max_rows = max(len(gs_lines), len(es_lines))
        for i in range(max_rows):
            g = gs_lines[i] if i < len(gs_lines) else ""
            e = es_lines[i] if i < len(es_lines) else ""
            merged.append(f"{g:45s}  {e}")
        self._dcomp_text.config(state="normal")
        self._dcomp_text.delete("1.0", tk.END)
        self._dcomp_text.insert(tk.END, "\n".join(merged))
        self._dcomp_text.config(state="disabled")
