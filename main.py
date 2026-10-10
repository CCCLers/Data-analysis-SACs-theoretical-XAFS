import sys
import os
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from tkinterdnd2 import DND_FILES, TkinterDnD as TkDnD
import json
import csv
import numpy as np 
import matplotlib
matplotlib.use('Agg') 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# 导入自定义模块
import process_xas 
from plot_xas import InteractiveXASPlotter
from presets_manager import PresetsManager
import mixer_tool
import pdos_tool
import pdos_gs_tool
import pdos_gs_es_compare
import pdos_region_analysis

# =========================================================================
# 修复核心：E0SelectionDialog (彻底解决双轴重影问题)
# =========================================================================
class E0SelectionDialog(tk.Toplevel):
    def __init__(self, parent, raw_data, config):
        super().__init__(parent)
        self.title("Interactive E0 Selection")
        self.geometry("1200x850")
        self.raw_data = raw_data
        self.config = config
        self.selected_e0 = None 
        
        # --- 变量 ---
        default_fwhm = self._compute_preview_fwhm()
        self.fwhm_var = tk.StringVar(value=default_fwhm)
        
        # 可视化控制变量
        self.show_deriv1 = tk.BooleanVar(value=True)  # 默认显示一阶导
        self.show_deriv2 = tk.BooleanVar(value=False) # 默认不显示二阶导
        self.show_sticks = tk.BooleanVar(value=False) # 默认不显示原始激发
        
        self.selected_method = tk.StringVar(value="std3") 
        
        # Custom 专用变量
        self.custom_val = tk.DoubleVar(value=0.0)
        self.user_has_interacted = False
        
        # 缓存数据
        self.candidates = {} 
        self.current_x = None
        self.current_y = None
        self.current_dydx = None
        self.current_d2ydx2 = None 
        self._ax2 = None
        self._layer_state = None
        
        # 布局
        self._create_plot_area()
        self._create_controls()
        
        # 初始计算
        self._recalc_data()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # 键盘快捷键
        self.bind('<Return>', lambda e: self._on_confirm())
        self.bind('<KP_Enter>', lambda e: self._on_confirm())
        self.bind('<Escape>', lambda e: self._on_cancel())

    def _compute_preview_fwhm(self):
        try:
            cfg = self.config
            regions_str = cfg.get('broadening_regions_lorentz') or cfg.get('broadening_regions_gauss') or ""
            if not regions_str:
                return "1.5"
            if cfg.get('auto_split', False):
                parts = regions_str.replace(";", ",").split(",")
                pre = float(parts[0])
                post = float(parts[1]) if len(parts) > 1 else pre
                avg = (pre + post) / 2
            else:
                first = regions_str.split(";")[0].split(",")
                avg = float(first[-1])
            return f"{avg:.1f}"
        except:
            return "1.5"

    def _create_plot_area(self):
        # 只创建 Figure，不急着创建 Axes，留给 refresh_plot 统一管理
        self.fig = Figure(figsize=(10, 6), dpi=100)
        
        # 初始创建一个占位 ax，防止报错，但在 refresh 时会被覆盖
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 绑定点击
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

    def _create_controls(self):
        ctrl_frame = ttk.LabelFrame(self, text="Controls")
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # --- 区域1：展宽参数 ---
        fwhm_frame = ttk.Frame(ctrl_frame)
        fwhm_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Label(fwhm_frame, text="Preview FWHM:").pack(anchor=tk.W)
        fwhm_box = ttk.Frame(fwhm_frame)
        fwhm_box.pack(fill=tk.X, pady=2)
        ent = ttk.Entry(fwhm_box, textvariable=self.fwhm_var, width=6)
        ent.pack(side=tk.LEFT)
        ent.bind('<Return>', lambda e: self._recalc_data()) 
        ttk.Button(fwhm_box, text="Set", command=self._recalc_data, width=4).pack(side=tk.LEFT, padx=2)

        # --- 区域2：可视化层 (复选框) ---
        vis_frame = ttk.LabelFrame(ctrl_frame, text="Visualization Layers")
        vis_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # 这里使用了 command=self._refresh_plot，点击勾选框立即刷新
        ttk.Checkbutton(vis_frame, text="1st Deriv (Slope)", variable=self.show_deriv1, 
                        command=self._refresh_plot).pack(anchor=tk.W, padx=5)
        ttk.Checkbutton(vis_frame, text="2nd Deriv (Curvature)", variable=self.show_deriv2, 
                        command=self._refresh_plot).pack(anchor=tk.W, padx=5)
        ttk.Checkbutton(vis_frame, text="Raw Sticks (Original)", variable=self.show_sticks, 
                        command=self._refresh_plot).pack(anchor=tk.W, padx=5)

        # --- 区域3：E0 选择策略 ---
        self.radio_frame = ttk.LabelFrame(ctrl_frame, text="Select E0 Strategy")
        self.radio_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        col1 = ttk.Frame(self.radio_frame)
        col1.pack(side=tk.LEFT, padx=5)
        self.rb_std1 = ttk.Radiobutton(col1, text="Std 1", variable=self.selected_method, value="std1", command=self._refresh_plot)
        self.rb_std1.pack(anchor=tk.W)
        self.rb_std2 = ttk.Radiobutton(col1, text="Std 2", variable=self.selected_method, value="std2", command=self._refresh_plot)
        self.rb_std2.pack(anchor=tk.W)
        
        col2 = ttk.Frame(self.radio_frame)
        col2.pack(side=tk.LEFT, padx=5)
        self.rb_std3 = ttk.Radiobutton(col2, text="Std 3", variable=self.selected_method, value="std3", command=self._refresh_plot)
        self.rb_std3.pack(anchor=tk.W)
        
        custom_row = ttk.Frame(col2)
        custom_row.pack(anchor=tk.W)
        self.rb_custom = ttk.Radiobutton(custom_row, text="Custom:", variable=self.selected_method, value="custom", command=self._refresh_plot)
        self.rb_custom.pack(side=tk.LEFT)
        
        self.ent_custom = ttk.Entry(custom_row, textvariable=self.custom_val, width=8)
        self.ent_custom.pack(side=tk.LEFT, padx=2)
        self.ent_custom.bind('<Return>', self._on_manual_entry)

        # --- 区域4：确认 ---
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(side=tk.RIGHT, padx=10)
        
        self.lbl_final_val = ttk.Label(btn_frame, text="Selected: - eV", font=("Arial", 12, "bold"), foreground="blue")
        self.lbl_final_val.pack(side=tk.TOP, pady=5)
        
        ttk.Button(btn_frame, text="Confirm", command=self._on_confirm).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(side=tk.TOP, fill=tk.X)

    def _recalc_data(self):
        try: fwhm = float(self.fwhm_var.get())
        except: return
            
        emin = float(self.config.get('emin', 700))
        emax = float(self.config.get('emax', 750))
        npts = int(self.config.get('npts', 3000))
        self.current_x = np.linspace(emin, emax, npts)
        
        regions = [(emin, emax, fwhm)]
        self.current_y = process_xas.calculate_y(self.raw_data['energy'], self.raw_data['fdip'], self.current_x, regions, is_gaussian=False)
        self.current_dydx = np.gradient(self.current_y, self.current_x)
        self.current_d2ydx2 = np.gradient(self.current_dydx, self.current_x)
        
        self.candidates = process_xas.get_e0_candidates(self.current_x, self.current_y)
        
        self.rb_std1.config(text=f"Std1: {self.candidates.get('std1',(0,''))[0]:.2f}")
        self.rb_std2.config(text=f"Std2: {self.candidates.get('std2',(0,''))[0]:.2f}")
        self.rb_std3.config(text=f"Std3: {self.candidates.get('std3',(0,''))[0]:.2f}")
        
        if not self.user_has_interacted:
            self.custom_val.set(self.candidates.get('std3',(0,''))[0])
        
        self._refresh_plot()

    def _on_plot_click(self, event):
        if event.inaxes != self.ax: return
        if self.toolbar.mode != "": return 

        if event.xdata:
            self.custom_val.set(float(event.xdata))
            self.user_has_interacted = True
            self.selected_method.set("custom")
            self._refresh_plot()
            
    def _on_manual_entry(self, event=None):
        self.user_has_interacted = True
        self.selected_method.set("custom")
        self._refresh_plot()

    def _refresh_plot(self):
        if self.current_x is None or self.current_y is None or self.current_dydx is None or self.current_d2ydx2 is None: return

        current_layer = (self.show_deriv1.get(), self.show_deriv2.get(), self.show_sticks.get())
        needs_rebuild = self._layer_state != current_layer

        if needs_rebuild or self._ax2 is None:
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self._ax2 = None
            self._layer_state = current_layer

        self.ax.clear()

        # 1. Raw Sticks
        if self.show_sticks.get():
            max_y = np.max(self.current_y)
            max_f = np.max(self.raw_data['fdip'])
            scale = (max_y / max_f) * 0.5 if max_f > 0 else 1
            emin, emax = self.current_x[0], self.current_x[-1]
            valid = (self.raw_data['energy'] >= emin) & (self.raw_data['energy'] <= emax)
            self.ax.vlines(self.raw_data['energy'][valid], 0,
                           self.raw_data['fdip'][valid] * scale,
                           color='gray', alpha=0.5, linewidth=1, label="Raw Sticks")

        # 2. 主光谱
        self.ax.plot(self.current_x, self.current_y, 'k-', linewidth=2, label="Spectrum")

        # 3. 导数 (仅在图层变化时重建副轴)
        if self.show_deriv1.get() or self.show_deriv2.get():
            if needs_rebuild or self._ax2 is None:
                self._ax2 = self.ax.twinx()
                self._ax2.get_yaxis().set_visible(False)
            else:
                self._ax2.clear()
                self._ax2.get_yaxis().set_visible(False)

            if self.show_deriv1.get():
                self._ax2.plot(self.current_x, self.current_dydx,
                               color='tab:red', linestyle='--', alpha=0.6, label="1st Deriv")
            if self.show_deriv2.get():
                self._ax2.plot(self.current_x, self.current_d2ydx2,
                               color='tab:purple', linestyle=':', alpha=0.8, label="2nd Deriv")
        else:
            self._ax2 = None

        # 4. 候选线
        colors = {'std1': 'green', 'std2': 'orange', 'std3': 'blue'}
        ym = np.max(self.current_y)
        for k, (v, _) in self.candidates.items():
            self.ax.axvline(v, c=colors[k], ls=':', alpha=0.6)
            self.ax.text(v, ym * 1.01, k.upper(), color=colors[k], fontsize=8, rotation=90)

        # 5. 最终红线
        method = self.selected_method.get()
        final = self.custom_val.get() if method == 'custom' else self.candidates.get(method, self.candidates.get('std3', (0, 0)))[0]
        self.ax.axvline(final, c='red', lw=2, label="Selected")
        self.lbl_final_val.config(text=f"Selected: {final:.2f} eV")

        self.ax.set_title("Interactive Selection: Click plot to set Custom E0", fontsize=12)

        # 合并图例
        h1, l1 = self.ax.get_legend_handles_labels()
        if self._ax2 is not None:
            h2, l2 = self._ax2.get_legend_handles_labels()
            h1, l1 = h1 + h2, l1 + l2
        self.ax.legend(h1, l1, loc='upper left')

        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.canvas.draw()

    def _on_confirm(self):
        method = self.selected_method.get()
        if method == 'custom':
            self.selected_e0 = self.custom_val.get()
        elif method in self.candidates:
            self.selected_e0 = self.candidates[method][0]
        else:
            self.selected_e0 = self.candidates.get('std3', (0,0))[0]
        self.destroy()

    def _on_cancel(self):
        self.selected_e0 = None
        self.destroy()


# =========================================================================
# 主程序
# =========================================================================
class XASProcessorGUI:
    def __init__(self, master):
        self.master = master
        master.title("XAS Data Processor")
        master.geometry("1200x850") 
        master.resizable(True, True)

        self.preset_manager = PresetsManager()
        self.settings_file = self._get_settings_file_path()
        self.settings = self._load_settings()

        self.spectrum_output_var = tk.StringVar(value=self.settings.get('spectrum_output_choice', "spectrum_dir"))
        self.spectrum_custom_path = tk.StringVar(value=self.settings.get('spectrum_custom_path', ""))
        self.plot_output_var = tk.StringVar(value=self.settings.get('plot_output_choice', "current_dir"))
        self.plot_custom_path = tk.StringVar(value=self.settings.get('plot_custom_path', ""))
        
        self.broadening_method_var = tk.StringVar(value=self.settings.get('broadening_method', "Lorentzian"))
        self.current_preset_var = tk.StringVar()

        self.auto_split_var = tk.BooleanVar(value=False)
        self.split_strategy_var = tk.StringVar(value="std3") 

        self.preview_window = None
        self.plotter = None 

        self._create_menubar()
        self._create_widgets()
        
        self._refresh_preset_list()
        last_preset = self.settings.get('last_preset')
        names = self.preset_manager.get_preset_names()
        if last_preset and last_preset in names:
            self.current_preset_var.set(last_preset)
        elif names:
            self.current_preset_var.set(names[0])
        
        self._load_selected_preset_to_gui(None)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_menubar(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.on_closing)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        tools_menu.add_command(label="Spectrum Mixer", command=self.open_mixer_tool)
        tools_menu.add_separator()
        tools_menu.add_command(label="PDOS-XAS Analysis (ES)", command=self.open_pdos_tool)
        tools_menu.add_command(label="PDOS-XAS Analysis (GS)", command=self.open_pdos_gs_tool)
        tools_menu.add_separator()
        tools_menu.add_command(label="GS vs ES PDOS Compare", command=self.open_gs_vs_es_tool)
        tools_menu.add_separator()
        tools_menu.add_command(label="Region Integration (Centroid & Area)",
                               command=self.open_region_tool)
        tools_menu.add_separator()
        tools_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Data-analysis-SACs-theoretical-XAFS\n"
                                                                                  "CP2K Theoretical XAFS Data Analysis Toolkit\n\n"
                                                                                  "Source: https://github.com/CCCLers/Data-analysis-SACs-therotical-XAFS"))

    def open_mixer_tool(self):
        mixer_win = tk.Toplevel(self.master)
        app = mixer_tool.MixerWindow(mixer_win)

    def open_pdos_tool(self):
        pdos_win = tk.Toplevel(self.master)
        app = pdos_tool.PDOSXASWindow(pdos_win)

    def open_pdos_gs_tool(self):
        pdos_win = tk.Toplevel(self.master)
        app = pdos_gs_tool.PDOSGSXASWindow(pdos_win)

    def open_gs_vs_es_tool(self):
        cmp_win = tk.Toplevel(self.master)
        app = pdos_gs_es_compare.GSvsESWindow(cmp_win)

    def open_region_tool(self):
        reg_win = tk.Toplevel(self.master)
        app = pdos_region_analysis.PDOSRegionWindow(reg_win)

    def _get_unique_filepath(self, directory, filename):
        base, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        full_path = os.path.join(directory, new_filename)
        while os.path.exists(full_path):
            new_filename = f"{base}_{counter}{ext}"
            full_path = os.path.join(directory, new_filename)
            counter += 1
        return full_path

    def _get_settings_file_path(self):
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        settings_dir = os.path.join(base_dir, 'settings')
        os.makedirs(settings_dir, exist_ok=True)
        return os.path.join(settings_dir, 'app_settings.json')

    def _load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def _save_settings(self):
        self.settings.update({
            'spectrum_output_choice': self.spectrum_output_var.get(),
            'spectrum_custom_path': self.spectrum_custom_path.get(),
            'plot_output_choice': self.plot_output_var.get(),
            'plot_custom_path': self.plot_custom_path.get(),
            'broadening_method': self.broadening_method_var.get(),
            'last_preset': self.current_preset_var.get()
        })
        try:
            with open(self.settings_file, 'w') as f: json.dump(self.settings, f, indent=4)
        except Exception as e: logger.error(f"Save error: {e}")

    def _create_widgets(self):
        preset_frame = ttk.LabelFrame(self.master, text="Presets")
        preset_frame.pack(pady=5, padx=10, fill="x")

        ttk.Label(preset_frame, text="Select Preset:").pack(side=tk.LEFT, padx=5)
        self.preset_combo = ttk.Combobox(preset_frame, textvariable=self.current_preset_var, state="readonly", width=30)
        self.preset_combo.pack(side=tk.LEFT, padx=5)
        self.preset_combo.bind("<<ComboboxSelected>>", self._load_selected_preset_to_gui)

        ttk.Button(preset_frame, text="Save Preset", command=self._save_new_preset).pack(side=tk.LEFT, padx=10)
        ttk.Button(preset_frame, text="Delete", command=self._delete_current_preset).pack(side=tk.LEFT, padx=5)

        config_frame = ttk.LabelFrame(self.master, text="Calculation Parameters")
        config_frame.pack(pady=5, padx=10, fill="x")

        row1 = ttk.Frame(config_frame)
        row1.pack(fill="x", pady=5)
        self.entries = {} 

        def add_entry(parent, label, key, width=10):
            ttk.Label(parent, text=label).pack(side=tk.LEFT, padx=(10, 2))
            entry = ttk.Entry(parent, width=width)
            entry.pack(side=tk.LEFT)
            self.entries[key] = entry
            return entry

        add_entry(row1, "Atom:", "atom_kind", 5)
        add_entry(row1, "Donor:", "donor_type", 5)
        add_entry(row1, "Excit:", "excitation_type", 5)
        add_entry(row1, "E min:", "emin", 8)
        add_entry(row1, "E max:", "emax", 8)
        add_entry(row1, "N pts:", "npts", 8)

        row_control = ttk.Frame(config_frame)
        row_control.pack(fill="x", pady=5, padx=10)
        
        self.auto_check = ttk.Checkbutton(row_control, text="Enable Auto-Split (Interactive E0 Selection)", 
                                          variable=self.auto_split_var, command=self._on_auto_split_toggle)
        self.auto_check.pack(side=tk.LEFT)
        
        self.lbl_strategy_hint = ttk.Label(row_control, text=" (Check to open interactive E0 picker)", foreground="blue", state="disabled")
        self.lbl_strategy_hint.pack(side=tk.LEFT, padx=10)

        row2 = ttk.Frame(config_frame)
        row2.pack(fill="x", pady=5)
        self.lbl_lorentz = ttk.Label(row2, text="Lorentzian Rules:", foreground="blue")
        self.lbl_lorentz.pack(side=tk.LEFT, padx=10)
        self.entries["broadening_regions_lorentz"] = ttk.Entry(row2)
        self.entries["broadening_regions_lorentz"].pack(side=tk.LEFT, fill="x", expand=True, padx=5)

        row3 = ttk.Frame(config_frame)
        row3.pack(fill="x", pady=5)
        self.lbl_gauss = ttk.Label(row3, text="Gaussian Rules:", foreground="green")
        self.lbl_gauss.pack(side=tk.LEFT, padx=10)
        self.entries["broadening_regions_gauss"] = ttk.Entry(row3)
        self.entries["broadening_regions_gauss"].pack(side=tk.LEFT, fill="x", expand=True, padx=5)

        self._on_auto_split_toggle()

        main_op_frame = ttk.Frame(self.master)
        main_op_frame.pack(pady=5, padx=10, fill="both", expand=True)
        main_op_frame.columnconfigure(0, weight=1)
        main_op_frame.columnconfigure(1, weight=1)

        left_frame = ttk.LabelFrame(main_op_frame, text="Data Generation (.spectrum -> .csv)")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5)

        method_frame = ttk.Frame(left_frame)
        method_frame.pack(pady=5)
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        ttk.Combobox(method_frame, textvariable=self.broadening_method_var, 
                     values=["Lorentzian", "Gaussian"], state="readonly", width=12).pack(side=tk.LEFT, padx=5)

        self.process_dnd_text = tk.Text(left_frame, height=6, relief="solid", bd=1)
        self.process_dnd_text.pack(pady=5, padx=5, fill="both", expand=True)
        self.process_dnd_text.insert(tk.END, "\n      Drag & Drop .spectrum files here")
        self.process_dnd_text.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.process_dnd_text.dnd_bind('<<Drop>>', self.process_spectrum_dnd)  # type: ignore[attr-defined]
        self.process_dnd_text.config(state="disabled")

        ttk.Button(left_frame, text="Or Select Files", command=self.select_spectrum_files).pack(pady=5)
        
        path_frame = ttk.Frame(left_frame)
        path_frame.pack(fill="x", pady=5)
        ttk.Label(path_frame, text="Output:").pack(side=tk.LEFT, padx=5)
        self.spectrum_out_cb = ttk.Combobox(path_frame, textvariable=self.spectrum_output_var, 
                     values=["spectrum_dir", "custom_dir"], state="readonly", width=15)
        self.spectrum_out_cb.pack(side=tk.LEFT)
        self.spectrum_out_cb.bind("<<ComboboxSelected>>", self._on_spectrum_output_change)

        right_frame = ttk.LabelFrame(main_op_frame, text="Visualization (CSV Compare)")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        self.plot_dnd_text = tk.Text(right_frame, height=6, relief="solid", bd=1)
        self.plot_dnd_text.pack(pady=5, padx=5, fill="both", expand=True)
        self.plot_dnd_text.insert(tk.END, "\n      Drag & Drop .csv files here")
        self.plot_dnd_text.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.plot_dnd_text.dnd_bind('<<Drop>>', self.plot_csvs_dnd)  # type: ignore[attr-defined]
        self.plot_dnd_text.config(state="disabled")

        ttk.Button(right_frame, text="Or Select CSVs", command=self.select_csv_files).pack(pady=5)

        self.status_label = ttk.Label(self.master, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_bar = ttk.Progressbar(self.master, mode='determinate')
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, before=self.status_label)
        self.progress_bar.pack_forget()

    def update_status(self, msg):
        try:
            self.status_label.config(text=msg)
            self.master.update_idletasks()
        except tk.TclError:
            pass

    def _on_auto_split_toggle(self):
        is_auto = self.auto_split_var.get()
        if is_auto:
            self.lbl_lorentz.config(text="Lorentzian (Pre-E0 FWHM, Post-E0 FWHM):")
            self.lbl_gauss.config(text="Gaussian (Pre-E0 FWHM, Post-E0 FWHM):")
            self.lbl_strategy_hint.config(state="normal")
        else:
            self.lbl_lorentz.config(text="Lorentzian Regions (start,end,fwhm;...):")
            self.lbl_gauss.config(text="Gaussian Regions (start,end,fwhm;...):")
            self.lbl_strategy_hint.config(state="disabled")

    def _refresh_preset_list(self):
        names = self.preset_manager.get_preset_names()
        self.preset_combo['values'] = names
        if names:
            last_used = self.settings.get('last_preset')
            current = self.current_preset_var.get()
            if not current or current not in names:
                if last_used and last_used in names:
                    self.current_preset_var.set(last_used)
                else:
                    self.current_preset_var.set(names[0])

    def _load_selected_preset_to_gui(self, event):
        name = self.current_preset_var.get()
        if not name: return
        data = self.preset_manager.get_preset_data(name)
        for key, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, data.get(key, ""))
        self.update_status(f"Loaded preset: {name}")

    def _gather_config_from_gui(self):
        config = {}
        for key, entry in self.entries.items():
            config[key] = entry.get().strip()
        config['auto_split'] = self.auto_split_var.get()
        config['split_strategy'] = self.split_strategy_var.get()
        return config

    def _save_new_preset(self):
        current_data = self._gather_config_from_gui()
        if 'auto_split' in current_data: del current_data['auto_split']
        if 'split_strategy' in current_data: del current_data['split_strategy']
        name = simpledialog.askstring("Save Preset", "Enter preset name:")
        if name:
            self.preset_manager.add_preset(name, current_data)
            self._refresh_preset_list()
            self.current_preset_var.set(name)

    def _delete_current_preset(self):
        name = self.current_preset_var.get()
        if name:
            self.preset_manager.delete_preset(name)
            self._refresh_preset_list()
            if self.preset_combo['values']: self.current_preset_var.set(self.preset_combo['values'][0])
            self._load_selected_preset_to_gui(None)

    def _on_spectrum_output_change(self, event):
        if self.spectrum_output_var.get() == "custom_dir":
            d = filedialog.askdirectory(title="Select Output Dir")
            if d: self.spectrum_custom_path.set(d)
    
    def process_spectrum_dnd(self, event):
        files = self.master.splitlist(event.data)
        spectrum_files = [f for f in files if f.lower().endswith('.spectrum')]
        if not spectrum_files:
            return
        if len(spectrum_files) == 1:
            self.run_processing(spectrum_files[0])
        else:
            self.batch_process(spectrum_files)

    def select_spectrum_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Spectrum files", "*.spectrum")])
        if not files:
            return
        if len(files) == 1:
            self.run_processing(files[0])
        else:
            self.batch_process(list(files))

    def run_processing(self, spectrum_file):
        """单文件处理：交互式 E0 选择 + 预览。"""
        if not spectrum_file.lower().endswith('.spectrum'):
            return
        config_dict = self._gather_config_from_gui()
        out_dir = self._get_output_dir(spectrum_file)
        raw_data = process_xas.parse_spectrum_data(spectrum_file, config_dict)
        if not raw_data:
            messagebox.showerror("Error", "Could not parse file. Check Atom/Donor/Excit settings.")
            return

        if self.auto_split_var.get():
            self.update_status("Waiting for E0 selection...")
            dialog = E0SelectionDialog(self.master, raw_data, config_dict)
            self.master.wait_window(dialog)
            if dialog.selected_e0 is None:
                self.update_status("Cancelled.")
                return
            config_dict['calculated_e0'] = dialog.selected_e0

        self.update_status(f"Processing: {os.path.basename(spectrum_file)}...")
        try:
            result = process_xas.process_spectrum(raw_data, config_dict, out_dir, self.broadening_method_var.get())
            if result:
                self._show_spectrum_preview(result, spectrum_file, out_dir)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _get_output_dir(self, spectrum_file):
        if self.spectrum_output_var.get() == "custom_dir":
            d = self.spectrum_custom_path.get()
            return d if d else os.path.dirname(spectrum_file)
        return os.path.dirname(spectrum_file)

    def batch_process(self, spectrum_files):
        """
        批量处理多个文件（无交互对话框，E0 使用自动计算 fallback）。
        """
        config_dict = self._gather_config_from_gui()
        if self.auto_split_var.get():
            config_dict['calculated_e0'] = None

        out_dir = self._get_output_dir(spectrum_files[0])

        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, before=self.status_label)
        self.progress_bar['maximum'] = len(spectrum_files)
        self.progress_bar['value'] = 0
        self.master.update_idletasks()

        success_count = 0
        for i, f in enumerate(spectrum_files):
            self.update_status(f"Batch: {i+1}/{len(spectrum_files)} - {os.path.basename(f)}")
            try:
                raw_data = process_xas.parse_spectrum_data(f, config_dict)
                if raw_data:
                    result = process_xas.process_spectrum(raw_data, config_dict, out_dir,
                                                          self.broadening_method_var.get())
                    if result:
                        success_count += 1
            except Exception as e:
                logger.error(f"Batch processing {f} failed: {e}")

            self.progress_bar['value'] = i + 1
            self.master.update_idletasks()

        self.progress_bar.pack_forget()
        self.update_status(f"Batch done: {success_count}/{len(spectrum_files)} succeeded")
        messagebox.showinfo("Batch Complete",
                            f"Processed {len(spectrum_files)} files.\n"
                            f"Success: {success_count}\n"
                            f"Failed: {len(spectrum_files) - success_count}")
    
    def _show_spectrum_preview(self, result, spectrum_file, output_dir):
        preview_window = tk.Toplevel(self.master)
        preview_window.title(f"Preview: {os.path.basename(spectrum_file)}")
        preview_window.geometry("1200x800")

        calculated_e0 = result.get('e0', None)

        show_deriv1_var = tk.BooleanVar(value=False)
        show_deriv2_var = tk.BooleanVar(value=False) 
        show_sticks_var = tk.BooleanVar(value=True) 

        chart_frame = ttk.Frame(preview_window)
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        fig = result['fig']
        ax = result['ax']
        # 强制更新标题为英文
        ax.set_title("Processed Spectrum", fontsize=14)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Intensity")

        dydx = result.get('dydx', np.gradient(result['y'], result['x']))
        d2ydx2 = result.get('d2ydx2', np.gradient(dydx, result['x']))
        
        ax_deriv1 = ax.twinx()
        ax_deriv1.spines['right'].set_position(('outward', 0))
        ax_deriv1.get_yaxis().set_visible(False)
        line_d1, = ax_deriv1.plot(result['x'], dydx, color='tab:orange', linestyle='--', linewidth=1.5, visible=False, label="1st Deriv")

        ax_deriv2 = ax.twinx()
        ax_deriv2.spines['right'].set_position(('outward', 60)) 
        ax_deriv2.get_yaxis().set_visible(False)
        line_d2, = ax_deriv2.plot(result['x'], d2ydx2, color='tab:purple', linestyle=':', linewidth=1.5, visible=False, label="2nd Deriv")

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, chart_frame)

        def toggle_layers():
            vis1 = show_deriv1_var.get()
            line_d1.set_visible(vis1)
            ax_deriv1.get_yaxis().set_visible(vis1)
            vis2 = show_deriv2_var.get()
            line_d2.set_visible(vis2)
            ax_deriv2.get_yaxis().set_visible(vis2)
            vis_sticks = show_sticks_var.get()
            for coll in ax.collections: coll.set_visible(vis_sticks)
            canvas.draw_idle()

        def save_spectrum():
            base_name_raw = f"{os.path.splitext(os.path.basename(spectrum_file))[0]}_{self.broadening_method_var.get()}.csv"
            csv_path = self._get_unique_filepath(output_dir, base_name_raw)
            png_path = csv_path.replace('.csv', '.png')
            try:
                with open(csv_path, 'w', newline='') as f:
                    if calculated_e0 is not None:
                        f.write(f"# E0_VAL:{calculated_e0:.4f}\n")
                    
                    writer = csv.writer(f)
                    writer.writerow(['Energy', 'Intensity', '1st_Deriv', '2nd_Deriv'])
                    for i in range(len(result['x'])):
                        writer.writerow([result['x'][i], result['y'][i], dydx[i], d2ydx2[i]])
                
                fig.savefig(png_path, format="png", dpi=300, bbox_inches='tight')
                
                msg = f"Saved to:\n{csv_path}"
                if calculated_e0:
                    msg += f"\n(Aligned at E0 = {calculated_e0:.2f} eV)"
                messagebox.showinfo("Success", msg)
                preview_window.destroy()
            except Exception as e: messagebox.showerror("Error", str(e))

        ctrl_frame = ttk.LabelFrame(preview_window, text="Layers")
        ctrl_frame.pack(side=tk.BOTTOM, fill="x", padx=10, pady=5)
        ttk.Checkbutton(ctrl_frame, text="1st Derivative", variable=show_deriv1_var, command=toggle_layers).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(ctrl_frame, text="2nd Derivative", variable=show_deriv2_var, command=toggle_layers).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(ctrl_frame, text="Raw Sticks", variable=show_sticks_var, command=toggle_layers).pack(side=tk.LEFT, padx=10)
        
        btn_frame = ttk.Frame(preview_window)
        btn_frame.pack(side=tk.BOTTOM, pady=5)
        ttk.Button(btn_frame, text="Save Data & Image", command=save_spectrum).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Close", command=preview_window.destroy).pack(side=tk.LEFT, padx=10)

    def plot_csvs_dnd(self, event):
        files = self.master.splitlist(event.data)
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        if csv_files: self._show_plot_preview_window(csv_files)

    def select_csv_files(self):
        csv_files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if csv_files: self._show_plot_preview_window(list(csv_files))

    def _show_plot_preview_window(self, csv_files):
        if self.preview_window and self.preview_window.winfo_exists(): 
            self.preview_window.destroy()
        
        self.preview_window = tk.Toplevel(self.master)
        self.preview_window.title(f"XAS Interactive Plotter ({len(csv_files)} files)")
        self.preview_window.geometry("1100x750") 
        self.plotter = InteractiveXASPlotter(self.preview_window, csv_files)

    def on_closing(self):
        self._save_settings()
        self.master.destroy()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    root = TkDnD.Tk()
    app = XASProcessorGUI(root)
    root.mainloop()