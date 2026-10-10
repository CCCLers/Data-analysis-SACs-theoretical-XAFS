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
import export_utils as xport

logger = logging.getLogger(__name__)

_ORBITAL_GROUPS = es._ORBITAL_GROUPS
_SELECTION_RULES = es._SELECTION_RULES
_EDGE_OPTIONS = es._EDGE_OPTIONS


def parse_gs_pdos(directory):
    files = sorted(glob.glob(os.path.join(directory, '*charge*.pdos')))
    return es.parse_pdos_file_list(files)


def parse_gs_pdos_files(file_paths):
    filtered = [f for f in file_paths if 'charge' in os.path.basename(f)]
    return es.parse_pdos_file_list(filtered)


class PDOSGSXASWindow:
    def __init__(self, master, directory=None):
        self.master = master
        master.title("Ground-State PDOS + XAS Analysis")
        master.geometry("1400x900")

        self.pdos_data = None
        self.xas_energy = None
        self.xas_intensity = None
        self.xas_e0 = None
        self.xas_filepath = None
        self.pdos_dir = directory or ''
        self._align_clicks = []
        self._align_mode_active = False

        self.edge_var = tk.StringVar(value='K')
        self.fwhm_var = tk.DoubleVar(value=0.8)
        self.is_gaussian_var = tk.BooleanVar(value=False)
        self.energy_shift_var = tk.DoubleVar(value=0.0)
        self.show_spin_alpha = tk.BooleanVar(value=True)
        self.show_spin_beta = tk.BooleanVar(value=True)
        self.show_xas = tk.BooleanVar(value=True)
        self.unocc_only_var = tk.BooleanVar(value=True)
        self.normalize_var = tk.BooleanVar(value=True)
        self.split_spin_var = tk.BooleanVar(value=True)
        self.show_d_subplot_var = tk.BooleanVar(value=False)
        self.show_d_components_var = tk.BooleanVar(value=False)
        self.mirror_spin_var = tk.BooleanVar(value=False)
        self.align_mode_var = tk.StringVar(value='Fermi')

        self.orbital_vars = {}
        self.orbital_styles = {}
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

        xas_section = ttk.LabelFrame(scroll_frame, text="XAS Spectrum", padding=5)
        xas_section.pack(fill=tk.X, pady=2)
        self.lbl_xas = ttk.Label(xas_section, text="No file loaded", foreground="gray")
        self.lbl_xas.pack(fill=tk.X)
        xas_btn_frame = ttk.Frame(xas_section)
        xas_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(xas_btn_frame, text="Load CSV", command=self._load_xas_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(xas_btn_frame, text="Load Spectrum", command=self._load_xas_spectrum).pack(side=tk.LEFT, padx=2)
        self.xas_drop = tk.Text(xas_section, height=2, relief="solid", bd=1)
        self.xas_drop.pack(fill=tk.X, pady=2)
        self.xas_drop.insert(tk.END, "\n  Drop XAS CSV here")
        self.xas_drop.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.xas_drop.dnd_bind('<<Drop>>', self._on_xas_drop)  # type: ignore[attr-defined]
        self.xas_drop.config(state="disabled")

        pdos_section = ttk.LabelFrame(scroll_frame, text="GS PDOS Data", padding=5)
        pdos_section.pack(fill=tk.X, pady=2)
        self.lbl_pdos = ttk.Label(pdos_section, text="No GS PDOS loaded", foreground="gray")
        self.lbl_pdos.pack(fill=tk.X)
        pdos_btn_frame = ttk.Frame(pdos_section)
        pdos_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(pdos_btn_frame, text="Load Directory", command=self._load_pdos_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(pdos_btn_frame, text="Load .pdos Files", command=self._load_pdos_files).pack(side=tk.LEFT, padx=2)
        self.pdos_drop = tk.Text(pdos_section, height=2, relief="solid", bd=1)
        self.pdos_drop.pack(fill=tk.X, pady=2)
        self.pdos_drop.insert(tk.END, "\n  Drop .pdos files/folder here")
        self.pdos_drop.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.pdos_drop.dnd_bind('<<Drop>>', self._on_pdos_drop)  # type: ignore[attr-defined]
        self.pdos_drop.config(state="disabled")

        align_section = ttk.LabelFrame(scroll_frame, text="Energy Alignment", padding=5)
        align_section.pack(fill=tk.X, pady=2)

        align_row1 = ttk.Frame(align_section)
        align_row1.pack(fill=tk.X)
        ttk.Label(align_row1, text="Edge:").pack(side=tk.LEFT)
        edge_cb = ttk.Combobox(align_row1, textvariable=self.edge_var, values=_EDGE_OPTIONS, state="readonly", width=5)
        edge_cb.pack(side=tk.LEFT, padx=5)
        edge_cb.bind("<<ComboboxSelected>>", lambda e: (self._update_rule_label(), self._update_plot()))
        ttk.Label(align_row1, text="Selection Rule:").pack(side=tk.LEFT, padx=5)
        self.lbl_rule = ttk.Label(align_row1, text="", foreground="blue")
        self.lbl_rule.pack(side=tk.LEFT)
        self._update_rule_label()

        gs_align_modes = [
            ('Fermi', 'Fermi → E0'),
            ('Pre-edge', 'Pre-edge → Fe-d'),
            ('Main-edge', 'Main-edge → Fe-p'),
        ]
        for val, txt in gs_align_modes:
            rb = ttk.Radiobutton(align_section, text=txt, variable=self.align_mode_var,
                                 value=val, command=self._auto_align)
            rb.pack(anchor=tk.W, padx=10)
        ttk.Button(align_section, text="Auto Align", command=self._auto_align).pack(pady=2)

        shift_row = ttk.Frame(align_section)
        shift_row.pack(fill=tk.X)
        ttk.Label(shift_row, text="Fine Shift:").pack(side=tk.LEFT)
        self.ent_shift = ttk.Entry(shift_row, textvariable=self.energy_shift_var, width=6)
        self.ent_shift.pack(side=tk.LEFT, padx=1)
        ttk.Label(shift_row, text="eV").pack(side=tk.LEFT)
        self.ent_shift.bind('<Return>', lambda e: self._update_plot())
        ttk.Button(shift_row, text='◀◀', width=3, command=lambda: self._nudge_shift(-2.0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(shift_row, text='◀', width=3, command=lambda: self._nudge_shift(-0.5)).pack(side=tk.LEFT, padx=1)
        ttk.Button(shift_row, text='▶', width=3, command=lambda: self._nudge_shift(0.5)).pack(side=tk.LEFT, padx=1)
        ttk.Button(shift_row, text='▶▶', width=3, command=lambda: self._nudge_shift(2.0)).pack(side=tk.LEFT, padx=1)

        click_row = ttk.Frame(align_section)
        click_row.pack(fill=tk.X, pady=1)
        self.btn_align_click = ttk.Button(click_row, text="Align by Click",
                                          command=self._toggle_align_click_mode)
        self.btn_align_click.pack(side=tk.LEFT)
        self.lbl_click_status = ttk.Label(click_row, text="", foreground="red", font=("Arial", 8))
        self.lbl_click_status.pack(side=tk.LEFT, padx=5)

        self.lbl_align_info = ttk.Label(align_section, text="Align: Fermi → E0", foreground="gray", font=("Arial", 8), wraplength=300)
        self.lbl_align_info.pack(anchor=tk.W, padx=2, fill=tk.X)

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

        orb_section = ttk.LabelFrame(scroll_frame, text="Orbitals to Display", padding=5)
        orb_section.pack(fill=tk.X, pady=2)
        d_toggle_row = ttk.Frame(orb_section)
        d_toggle_row.pack(fill=tk.X, pady=(0, 2))
        self.btn_split_d = ttk.Button(d_toggle_row, text="\u25b6 d components",
                                      command=self._toggle_d_components, width=16)
        self.btn_split_d.pack(side=tk.LEFT)
        self.orbital_frame = ttk.Frame(orb_section)
        self.orbital_frame.pack(fill=tk.X)

        display_section = ttk.LabelFrame(scroll_frame, text="Display Options", padding=5)
        display_section.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(display_section, text="Show XAS", variable=self.show_xas,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="Occupied + Unoccupied", variable=self.unocc_only_var,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="Normalize [0,1]", variable=self.normalize_var,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="Split α/β Spins", variable=self.split_spin_var,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="d-Subplot Row", variable=self.show_d_subplot_var,
                        command=self._update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_section, text="Mirror β Spin (up/down)", variable=self.mirror_spin_var,
                        command=self._update_plot).pack(anchor=tk.W)
        spin_row = ttk.Frame(display_section)
        spin_row.pack(fill=tk.X)
        ttk.Checkbutton(spin_row, text="α", variable=self.show_spin_alpha,
                        command=self._update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(spin_row, text="β", variable=self.show_spin_beta,
                        command=self._update_plot).pack(side=tk.LEFT, padx=5)

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

        self.fig = plt.Figure(figsize=(12, 10), dpi=100)
        self.gs = GridSpec(2, 1, figure=self.fig, height_ratios=[1, 1], hspace=0.35)
        self.ax_top = self.fig.add_subplot(self.gs[0])
        self.ax_hyb = self.fig.add_subplot(self.gs[1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_align_click)
        NavigationToolbar2Tk(self.canvas, plot_frame)
        self._init_plot_labels()

    def _update_rule_label(self):
        edge = self.edge_var.get()
        rule = _SELECTION_RULES.get(edge, {})
        self.lbl_rule.config(text=rule.get('desc', ''))

    def _init_plot_labels(self):
        self.ax_top.set_title("XAS + Ground-State PDOS", fontsize=11)
        self.ax_top.set_xlabel("Energy (eV)", fontsize=10)
        self.ax_top.set_ylabel("Intensity", fontsize=10)
        self.ax_top.grid(True, alpha=0.25, linestyle='--')
        self.ax_hyb.set_title("GS Orbital Hybridization", fontsize=11)
        self.ax_hyb.set_xlabel("Energy (eV)", fontsize=10)
        self.ax_hyb.set_ylabel("PDOS (states/eV)", fontsize=10)
        self.ax_hyb.grid(True, alpha=0.25, linestyle='--')

    def _get_default_orbital_style(self, atom, group):
        color, ls = es._CB_PALETTE[0]
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
            for g in ad[spin_key]['groups']:
                key = f"{atom}_{g}"
                default = g == 'p'
                if g in es._D_SUBGROUP_NAMES:
                    default = False
                self.orbital_vars[key] = old_vars.get(key, tk.BooleanVar(value=default))

                if old_styles.get(key):
                    style = old_styles[key]
                else:
                    color, ls = es._CB_PALETTE[palette_idx % len(es._CB_PALETTE)]
                    lw = 1.5
                    style = {
                        'color': color, 'linestyle': ls, 'linewidth': lw, 'alpha': 0.8,
                        'beta_linestyle': ls, 'beta_linewidth': lw,
                    }
                    palette_idx += 1
                self.orbital_styles[key] = style

                if g in es._D_SUBGROUP_NAMES and not show_d_comp:
                    continue

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

    # --- XAS loading (reuses from pdos_tool) ---
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
        pdos_files = [f for f in files if f.lower().endswith('.pdos') and 'charge' in os.path.basename(f).lower()]
        dirs = [f for f in files if os.path.isdir(f)]
        if dirs:
            self._load_pdos(dirs[0])
        elif pdos_files:
            self._load_pdos_from_files(pdos_files)

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

    # --- PDOS loading ---
    def _load_pdos_dir(self):
        initial = self.pdos_dir if self.pdos_dir else (os.path.dirname(self.xas_filepath) if self.xas_filepath else '.')
        d = filedialog.askdirectory(title="Select folder with *charge*.pdos files", initialdir=initial)
        if d:
            self._load_pdos(d)

    def _load_pdos_files(self):
        initial = self.pdos_dir if self.pdos_dir else (os.path.dirname(self.xas_filepath) if self.xas_filepath else '.')
        fps = filedialog.askopenfilenames(title="Select *charge*.pdos Files", filetypes=[("PDOS files", "*.pdos")], initialdir=initial)
        if fps:
            self._load_pdos_from_files(list(fps))

    def _load_pdos_from_files(self, file_paths):
        data = parse_gs_pdos_files(file_paths)
        if data is None:
            self.lbl_pdos.config(text="No valid GS PDOS files found", foreground="red")
            return
        self.pdos_data = data
        self.pdos_dir = os.path.dirname(file_paths[0])
        atoms = list(data['atoms'].keys())
        self.lbl_pdos.config(text=f"GS PDOS: {', '.join(atoms)} | Fermi={data['fermi_ev']:.2f}eV", foreground="black")
        logger.info(f"Loaded GS PDOS: atoms={atoms}, Fermi={data['fermi_ev']:.2f}eV")
        self._rebuild_orbital_controls()
        lumo = self._compute_lumo()
        if lumo is not None:
            self.lbl_align_info.config(text=f"GS LUMO: {lumo:.2f} eV", foreground="gray")
        self._auto_align()
        self._update_plot()
        self.master.lift()

    def _load_pdos(self, directory):
        data = parse_gs_pdos(directory)
        if data is None:
            self.lbl_pdos.config(text="No *charge*.pdos files found", foreground="red")
            return
        self.pdos_data = data
        self.pdos_dir = directory
        atoms = list(data['atoms'].keys())
        self.lbl_pdos.config(text=f"GS PDOS: {', '.join(atoms)} | Fermi={data['fermi_ev']:.2f}eV", foreground="black")
        logger.info(f"Loaded GS PDOS: atoms={atoms}, Fermi={data['fermi_ev']:.2f}eV")
        self._rebuild_orbital_controls()
        lumo = self._compute_lumo()
        if lumo is not None:
            self.lbl_align_info.config(text=f"GS LUMO: {lumo:.2f} eV", foreground="gray")
        self._auto_align()
        self._update_plot()
        self.master.lift()

    def _compute_lumo(self):
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
        if self.xas_energy is None or self.xas_e0 is None:
            return self.xas_e0 or 0
        e, y = self.xas_energy, self.xas_intensity
        e0 = self.xas_e0
        mask = (e < e0 - 0.5) & (e > e.min() + 2)
        if mask.sum() < 3:
            return e[e < e0][-1] if (e < e0).any() else e0 - 2
        ye, ee = y[mask], e[mask]
        d = np.diff(ye)
        peaks = []
        for i in range(1, len(d)):
            if d[i-1] > 0 and d[i] <= 0:
                peaks.append((ee[i], ye[i]))
        if not peaks:
            return ee[-1]
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[0][0]

    def _detect_xas_mainedge_peak(self):
        if self.xas_energy is None:
            return self.xas_e0 or 0
        return self.xas_energy[np.argmax(self.xas_intensity)]

    def _detect_gs_pdos_first_peak(self, atom, group):
        if not self.pdos_data or atom not in self.pdos_data['atoms']:
            return 0
        ad = self.pdos_data['atoms'][atom]
        e_src, dos_src = self._extract_pdos(atom, group, ad)
        if e_src is None or len(e_src) < 3:
            return 0
        fwhm = self.fwhm_var.get()
        is_g = self.is_gaussian_var.get()
        xg = np.linspace(-10, 25, 2000)
        dos = self._broaden_safe(e_src, dos_src, xg, fwhm, is_g)
        thresh = np.max(dos) * 0.05
        d = np.diff(dos)
        for i in range(1, len(d)):
            if d[i-1] > 0 and d[i] <= 0 and dos[i] > thresh:
                return xg[i]
        return xg[np.argmax(dos)]

    def _detect_gs_pdos_main_peak(self, atom, group):
        if not self.pdos_data or atom not in self.pdos_data['atoms']:
            return 0
        ad = self.pdos_data['atoms'][atom]
        e_src, dos_src = self._extract_pdos(atom, group, ad)
        if e_src is None or len(e_src) < 3:
            return 0
        fwhm = self.fwhm_var.get()
        is_g = self.is_gaussian_var.get()
        xg = np.linspace(-10, 25, 2000)
        dos = self._broaden_safe(e_src, dos_src, xg, fwhm, is_g)
        return xg[np.argmax(dos)]

    def _auto_align(self):
        if self.pdos_data is None:
            return
        mode = self.align_mode_var.get()
        info_parts = []

        if mode == 'Fermi':
            shift = 0.0
            info_parts.append("Fermi = E0 → shift=0")

        elif mode == 'Pre-edge':
            if self.xas_e0 is None:
                self.lbl_align_info.config(text="XAS not loaded", foreground="red")
                return
            xas_peak = self._detect_xas_preedge_peak()
            pdos_peak = self._detect_gs_pdos_first_peak('Fe', 'd')
            shift = (xas_peak - self.xas_e0) - pdos_peak
            info_parts.append(f"XAS pre-edge @ {xas_peak:.2f} eV")
            info_parts.append(f"GS Fe-d 1st peak @ {pdos_peak:.2f} eV")

        elif mode == 'Main-edge':
            if self.xas_e0 is None:
                self.lbl_align_info.config(text="XAS not loaded", foreground="red")
                return
            xas_peak = self._detect_xas_mainedge_peak()
            pdos_peak = self._detect_gs_pdos_main_peak('Fe', 'p')
            shift = (xas_peak - self.xas_e0) - pdos_peak
            info_parts.append(f"XAS white line @ {xas_peak:.2f} eV")
            info_parts.append(f"GS Fe-p peak @ {pdos_peak:.2f} eV")

        else:
            return

        self.energy_shift_var.set(shift)
        self.ent_shift.update()
        info_parts.append(f"shift={shift:.2f} eV")
        self.lbl_align_info.config(text=" | ".join(info_parts), foreground="black")
        self._update_plot()

    def _nudge_shift(self, delta):
        curr = self.energy_shift_var.get()
        self.energy_shift_var.set(round(curr + delta, 3))
        self.ent_shift.update()
        self._update_plot()

    def _toggle_align_click_mode(self):
        if self._align_mode_active:
            self._align_mode_active = False
            self._align_clicks = []
            self.lbl_click_status.config(text="")
            self.btn_align_click.config(text="Align by Click")
            if hasattr(self, '_click_marker'):
                for m in self._click_marker:
                    try:
                        m.remove()
                    except:
                        pass
                self._click_marker = []
            self.canvas.draw_idle()
            return
        self._align_mode_active = True
        self.lbl_click_status.config(text="Click: select PDOS peak")
        self.btn_align_click.config(text="Cancel")
        self._align_clicks = []
        self._click_marker = []

    def _on_align_click(self, event):
        if not self._align_mode_active or event.inaxes not in (self.ax_top, self.ax_hyb):
            return
        if len(self._align_clicks) == 0:
            self._align_clicks.append(event.xdata)
            self._click_marker.append(self.ax_top.axvline(event.xdata, color='r', ls='--', lw=1.5, alpha=0.8))
            self.lbl_click_status.config(text="Now click corresponding XAS feature")
            self.canvas.draw_idle()
        elif len(self._align_clicks) == 1:
            x2 = event.xdata
            x1 = self._align_clicks[0]
            curr = self.energy_shift_var.get()
            new_shift = round(curr + (x2 - x1), 3)
            self.energy_shift_var.set(new_shift)
            self.ent_shift.update()
            self._align_clicks = []
            self.lbl_click_status.config(text=f"Aligned: shift={new_shift:+.3f} eV")
            self.btn_align_click.config(text="Align by Click")
            for m in self._click_marker:
                try:
                    m.remove()
                except:
                    pass
            self._click_marker = []
            self._update_plot()

    def _broaden_safe(self, energy, dos, x_grid, fwhm, is_gauss):
        if energy is None or len(energy) < 2:
            return np.zeros_like(x_grid)
        return es.broaden_pdos(energy, dos, x_grid, fwhm, is_gauss)

    def _export_csv(self):
        if not any(self._plotted_curves.values()):
            messagebox.showwarning("No Data", "No curves to export. Load data first.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                          title="Export CSV (base name for multi-panel)")
        if not fp:
            return
        md = xport.build_metadata('pdos_gs',
                                  fwhm=self.fwhm_var.get(), is_gaussian=self.is_gaussian_var.get(),
                                  shift=self.energy_shift_var.get(), edge=self.edge_var.get(),
                                  normalized=self.normalize_var.get(), split_spin=self.split_spin_var.get())
        base, ext = os.path.splitext(fp)
        panels = [('top', 'XAS+PDOS'), ('hyb', 'Hybridization'), ('dcomp', 'd-Orbital Components')]
        for key, pname in panels:
            if self._plotted_curves.get(key):
                xport.write_panel_csv(f"{base}_{key}{ext}", pname, self._plotted_curves[key], metadata=md)
        self._write_dcomp_csv(base)
        messagebox.showinfo("Done", f"CSV exported:\n  {base}_[top,hyb,dcomp]{ext} + dcomp")

    def _export_image(self):
        fp = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                          title="Export Image")
        if not fp:
            return
        xport.save_figure_png(self.fig, fp)
        messagebox.showinfo("Done", f"Image saved to:\n{fp}")

    def _write_dcomp_csv(self, base_path):
        if self.pdos_data is None:
            return
        atom = 'Fe' if 'Fe' in self.pdos_data['atoms'] else None
        if atom is None:
            return
        comp = es._compute_d_composition(self.pdos_data, atom)
        if comp is None:
            return
        fp = f"{base_path}_dcomp.csv"
        with open(fp, 'w') as f:
            f.write("# d-orbital composition (occ>0.5)\n")
            f.write("component,alpha_percent,beta_percent,total_percent\n")
            for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
                c = comp[sub]
                f.write(f"{sub},{c['alpha']:.1f},{c['beta']:.1f},{c['total']:.1f}\n")

    def _compute_d_comp(self):
        if self.pdos_data is None:
            self._dcomp_text.config(state="normal")
            self._dcomp_text.delete("1.0", tk.END)
            self._dcomp_text.insert(tk.END, "Load GS PDOS first")
            self._dcomp_text.config(state="disabled")
            return
        atom = 'Fe' if 'Fe' in self.pdos_data['atoms'] else None
        if atom is None:
            self._dcomp_text.config(state="normal")
            self._dcomp_text.delete("1.0", tk.END)
            self._dcomp_text.insert(tk.END, "No Fe atom in PDOS")
            self._dcomp_text.config(state="disabled")
            return
        comp = es._compute_d_composition(self.pdos_data, atom)
        if comp is None:
            self._dcomp_text.config(state="normal")
            self._dcomp_text.delete("1.0", tk.END)
            self._dcomp_text.insert(tk.END, "No d character found")
            self._dcomp_text.config(state="disabled")
            return
        lines = [f"Fe d-orbital composition (occupied MOs):", "",
                 f"                  alpha  beta  total",
                 f"  {'─'*30}"]
        for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
            c = comp[sub]
            a_str = f"{c['alpha']:5.1f}" if c['alpha'] > 0 else "   — "
            b_str = f"{c['beta']:5.1f}" if c['beta'] > 0 else "   — "
            t_str = f"{c['total']:5.1f}"
            lines.append(f"  {es._D_SUBGROUP_LABELS[sub]:20s} {a_str} {b_str} {t_str}")
        self._dcomp_text.config(state="normal")
        self._dcomp_text.delete("1.0", tk.END)
        self._dcomp_text.insert(tk.END, "\n".join(lines))
        self._dcomp_text.config(state="disabled")

    # --- Plotting ---
    def _update_plot(self, *args):
        if self.pdos_data is None and self.xas_energy is None:
            return

        fine_shift = self.energy_shift_var.get()
        fwhm = self.fwhm_var.get()
        is_gauss = self.is_gaussian_var.get()
        do_norm = self.normalize_var.get()
        edge = self.edge_var.get()
        allowed_orb = _SELECTION_RULES.get(edge, {}).get('allowed', 'p')
        mirror = self.mirror_spin_var.get()

        self.fig.clear()
        self._plotted_curves = {'top': [], 'hyb': [], 'dcomp': []}
        show_d = self.show_d_subplot_var.get()
        if show_d and mirror:
            n_rows, ratios = 4, [1, 1, 0.6, 0.6]
        elif show_d:
            n_rows, ratios = 3, [1, 1, 0.8]
        else:
            n_rows, ratios = 2, [1, 1]
        self.gs = GridSpec(n_rows, 1, figure=self.fig, height_ratios=ratios, hspace=0.40)
        self.ax_top = self.fig.add_subplot(self.gs[0])
        self.ax_hyb = self.fig.add_subplot(self.gs[1])
        self._init_plot_labels()

        if self.xas_energy is not None and self.xas_e0 is not None:
            e_rel_min = self.xas_energy[0] - self.xas_e0 - 2
            e_rel_max = self.xas_energy[-1] - self.xas_e0 + 2
        else:
            e_rel_min, e_rel_max = -10, 25

        if self.pdos_data:
            for atom, ad in self.pdos_data['atoms'].items():
                for spin in ['alpha', 'beta']:
                    if spin not in ad:
                        continue
                    e_fermi = ad[spin]['energy'][0] - ad[spin]['energy'][0]
                    break

        # ---- Ax Top: XAS + GS PDOS (dual Y) ----
        ax1 = self.ax_top
        ax1.clear()
        ax1.set_title("XAS + Ground-State PDOS Comparison", fontsize=11)
        ax1.set_xlabel("Energy (eV)", fontsize=10)
        ax1.set_ylabel("XAS Intensity (±)" if mirror else "XAS Intensity", fontsize=10, color='black')
        ax1.grid(True, alpha=0.25, linestyle='--')
        ax1.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax1_pdos = ax1.twinx()
        ax1_pdos.set_ylabel("PDOS (α↑ / β↓)" if mirror else ("PDOS (norm)" if do_norm else "PDOS (states/eV)"),
                            fontsize=10, color='tab:red')

        # XAS on relative energy axis
        if self.xas_energy is not None and self.xas_e0 is not None and self.show_xas.get():
            xas_rel = self.xas_energy - self.xas_e0
            y = self.xas_intensity
            if do_norm and np.max(y) > 0:
                y = y / np.max(y)
            ax1.plot(xas_rel, y, 'k-', linewidth=2.0, label="XAS")
            self._plotted_curves['top'].append({'label': 'XAS', 'x': xas_rel.copy(), 'y': y.copy()})
            if mirror:
                ax1.plot(xas_rel, -y, 'k-', linewidth=2.0, alpha=0.5, label="XAS (β mirror)")
                self._plotted_curves['top'].append({'label': 'XAS (mirror)', 'x': xas_rel.copy(), 'y': (-y).copy()})
            ax1.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

        # GS PDOS on relative energy axis (0 = Fermi)
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
                        e_src, dos_src = self._extract_pdos(atom, group, ad, spin=spin_name)
                        if e_src is None or len(e_src) < 2:
                            continue
                        e_src = e_src + fine_shift
                        x_grid = np.linspace(e_rel_min, e_rel_max, 2000)
                        dos_b = self._broaden_safe(e_src, dos_src, x_grid, fwhm, is_gauss)
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
                        self._plotted_curves['top'].append({'label': lbl, 'x': x_grid.copy(), 'y': dos_plot.copy()})

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_pdos.get_legend_handles_labels()
            if lines1 or lines2:
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')

        ax1.set_xlim(e_rel_min, e_rel_max)

        # ---- Ax Bottom: Hybridization ----
        ax3 = self.ax_hyb
        ax3.clear()
        ax3.set_title("GS Orbital Hybridization", fontsize=11)
        ax3.set_xlabel("Energy (eV)", fontsize=10)
        ax3.set_ylabel("PDOS (α↑ / β↓)" if mirror else ("PDOS (norm)" if do_norm else "PDOS (states/eV)"), fontsize=10)
        ax3.grid(True, alpha=0.25, linestyle='--')
        ax3.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Fermi')

        if self.pdos_data:
            for key, var in self.orbital_vars.items():
                if not var.get():
                    continue
                parts = key.split('_', 1)
                atom, group = parts[0], parts[1] if len(parts) > 1 else ''
                if atom not in self.pdos_data['atoms'] or group not in es.VALID_HYB_GROUPS:
                    continue
                ad = self.pdos_data['atoms'][atom]
                style = self.orbital_styles.get(key, {'color': '#7f7f7f', 'alpha': 0.7})

                spin_specs = []
                if self.split_spin_var.get():
                    for sn, sl in [('α', 'alpha'), ('β', 'beta')]:
                        do_spin = (sn == 'α' and self.show_spin_alpha.get()) or (sn == 'β' and self.show_spin_beta.get())
                        if do_spin and sl in ad:
                            spin_specs.append((sl, '', f" ({sn})"))
                else:
                    spin_specs.append((None, style['linestyle'], ''))

                for spin_name, ls_mod, spin_tag in spin_specs:
                    e_src, dos_src = self._extract_pdos(atom, group, ad, spin=spin_name)
                    if e_src is None or len(e_src) < 2:
                        continue
                    e_src = e_src + fine_shift
                    x_grid = np.linspace(e_rel_min, e_rel_max, 2000)
                    dos_b = self._broaden_safe(e_src, dos_src, x_grid, fwhm, is_gauss)
                    if do_norm and np.max(dos_b) > 0:
                        dos_b = dos_b / np.max(dos_b)
                    if spin_name == 'beta':
                        if mirror:
                            dos_plot = -dos_b
                        else:
                            dos_plot = dos_b
                        ls_eff = style.get('beta_linestyle', '--')
                        lw_eff = style.get('beta_linewidth', style['linewidth'])
                        lbl = f"{key} (β↓)" if mirror else f"{key} (β)"
                    elif spin_name == 'alpha':
                        dos_plot = dos_b
                        ls_eff = style['linestyle']
                        lw_eff = style['linewidth']
                        lbl = f"{key} (α↑)" if mirror else f"{key} (α)"
                    else:
                        dos_plot = dos_b
                        ls_eff = style['linestyle']
                        lw_eff = style['linewidth']
                        lbl = key + spin_tag
                    ax3.plot(x_grid, dos_plot, linewidth=lw_eff, linestyle=ls_eff, color=style['color'], label=lbl)
                    self._plotted_curves['hyb'].append({'label': lbl, 'x': x_grid.copy(), 'y': dos_plot.copy()})
                    ax3.fill_between(x_grid, 0, dos_b, color=style['color'], alpha=0.08)

            if self.xas_e0:
                ax3.axvline(0, color='gray', linestyle=':', alpha=0.5, label=f'0')
            ax3.legend(loc='upper left', fontsize='small')
            ax3.set_xlim(e_rel_min, e_rel_max)

        # ---- d-Subplot Rows ----
        if self.show_d_subplot_var.get() and self.pdos_data and 'Fe' in self.pdos_data['atoms']:
            ad = self.pdos_data['atoms']['Fe']
            if mirror:
                ax_da = self.fig.add_subplot(self.gs[2])
                ax_da.set_title("Fe d-orbital components (α↑)", fontsize=10)
                ax_da.grid(True, alpha=0.25, linestyle='--')
                ax_da.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
                ax_db = self.fig.add_subplot(self.gs[3])
                ax_db.set_title("Fe d-orbital components (β↓)", fontsize=10)
                ax_db.set_xlabel("Energy (eV)", fontsize=9)
                ax_db.grid(True, alpha=0.25, linestyle='--')
                ax_db.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
                for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
                    st_key = f"Fe_{sub}"
                    sty = self.orbital_styles.get(st_key, {})
                    for spin_name, ax_d, sign, ls_attr, lw_attr in \
                        [('alpha', ax_da, 1, 'linestyle', 'linewidth'),
                         ('beta', ax_db, -1, 'beta_linestyle', 'beta_linewidth')]:
                        e_src, dos_src = self._extract_pdos('Fe', sub, ad, spin=spin_name)
                        if e_src is None or len(e_src) < 2:
                            continue
                        e_src = e_src + fine_shift
                        x_grid = np.linspace(e_rel_min, e_rel_max, 2000)
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
                ax_da.set_xlim(e_rel_min, e_rel_max)
                ax_db.legend(loc='upper right', fontsize='x-small', ncol=2)
                ax_db.set_xlim(e_rel_min, e_rel_max)
            else:
                ax_d = self.fig.add_subplot(self.gs[2])
                ax_d.set_title("Fe d-orbital components", fontsize=10)
                ax_d.set_xlabel("Energy (eV)", fontsize=9)
                ax_d.grid(True, alpha=0.25, linestyle='--')
                ax_d.axvline(0, color='gray', ls=':', alpha=0.5, lw=1)
                for sub in ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2'):
                    e_src, dos_src = self._extract_pdos('Fe', sub, ad)
                    if e_src is None or len(e_src) < 2:
                        continue
                    e_src = e_src + fine_shift
                    x_grid = np.linspace(e_rel_min, e_rel_max, 2000)
                    dos_b = self._broaden_safe(e_src, dos_src, x_grid, fwhm, is_gauss)
                    if do_norm and np.max(dos_b) > 0:
                        dos_b = dos_b / np.max(dos_b)
                    st_key = f"Fe_{sub}"
                    sty = self.orbital_styles.get(st_key, {})
                    col = sty.get('color', '#7f7f7f')
                    ls = sty.get('linestyle', '-')
                    lw = sty.get('linewidth', 1.5)
                    ax_d.plot(x_grid, dos_b, color=col, linestyle=ls,
                              linewidth=lw, alpha=0.8, label=sub)
                    self._plotted_curves['dcomp'].append({'label': sub, 'x': x_grid.copy(), 'y': dos_b.copy()})
                ax_d.legend(loc='upper right', fontsize='x-small', ncol=2)
                ax_d.set_xlim(e_rel_min, e_rel_max)

        self.canvas.draw_idle()

    def _extract_pdos(self, atom, group, ad, spin=None):
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
            return e_raw, dos_raw

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
            e = e_raw
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
