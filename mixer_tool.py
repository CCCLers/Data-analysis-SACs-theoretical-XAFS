import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinterdnd2 import DND_FILES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d
import os
import re

logger = logging.getLogger(__name__)

# --- 辅助函数 ---
def robust_load_csv(filepath):
    """
    读取 CSV，返回 (Energy, Intensity, Derivative, e0_header)。
    - Derivative：若文件无第3列则返回 None。
    - e0_header：若文件首行为 `# E0_VAL:xxx` 则返回该浮点值，否则 None。
    """
    try:
        delimiter = ',' if filepath.lower().endswith('.csv') else None

        # 解析首行的 # E0_VAL: 注释（若有）
        e0_header = None
        n_header = 0
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                n_header += 1
                if line.lstrip().startswith('#'):
                    if e0_header is None and line.lstrip().startswith('# E0_VAL:'):
                        try:
                            e0_header = float(line.split(':')[1].strip())
                        except Exception:
                            pass
                else:
                    break
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=n_header)

        if data.ndim < 2 or data.shape[1] < 2:
            return None, None, None, None

        x = data[:, 0]
        y = data[:, 1]

        deriv = None
        if data.shape[1] >= 3:
            deriv = data[:, 2]

        return x, y, deriv, e0_header
    except Exception:
        return None, None, None, None

class MixerWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Tool 1: Mixer (Auto-Align & Shape Compare)")
        self.master.geometry("1100x750")

        self.components = []    # 存储插值后(可选归一化)的 Y 数据
        self.weights_vars = []
        self.common_energy = None
        self.file_entries = []   # 每个文件的原始数据 {x,y,deriv,e0_header,name}，供模式切换时重建
        
        # --- Cache for Plot Lines ---
        self.line_objects = [] 
        self.line_sum = None   
        self.is_plot_initialized = False 
        
        # --- View Control Var ---
        self.norm_view_var = tk.BooleanVar(value=False) # 默认关闭，显示真实叠加
        # 混合模式: "Aligned"=按 E0 平移对齐后混合; "Direct"=直接按原始能量加权相加
        self.mix_mode_var = tk.StringVar(value="Aligned")
        # 强度归一化开关: 勾选时每个分量除以自身最大值再参与加权和，并影响保存
        self.norm_intensity_var = tk.BooleanVar(value=True)

        # --- Layout ---
        self.left_panel = ttk.Frame(master, padding=10, width=350)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        self.right_panel = ttk.Frame(master, padding=10)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Top Status Area ---
        self.status_frame = ttk.LabelFrame(self.left_panel, text="Composition Control", padding=10)
        self.status_frame.pack(fill=tk.X, pady=(0, 10))

        # Sum Label
        self.lbl_sum = tk.Label(self.status_frame, text="Sum: 0.00", font=("Arial", 12, "bold"), fg="blue")
        self.lbl_sum.pack(side=tk.LEFT, padx=5)

        # Normalize Weights Button
        self.btn_norm = ttk.Button(self.status_frame, text="Normalize Weights", command=self.normalize_weights, width=15)
        self.btn_norm.pack(side=tk.RIGHT, padx=5)

        # --- Drag & Drop Area ---
        lbl_instr = ttk.Label(self.left_panel, text="Drag & Drop CSV files below:", font=("Arial", 10))
        lbl_instr.pack(pady=(0, 5), anchor="w")

        self.drop_target = tk.Listbox(self.left_panel, height=8, bg="#f0f0f0", selectmode=tk.EXTENDED)
        self.drop_target.pack(fill=tk.X, pady=(0, 5))
        self.drop_target.insert(0, "   [ Drop Files Here ]")
        # 按 Delete 键也可移除选中项
        self.drop_target.bind('<Delete>', lambda e: self.remove_selected())

        self.drop_target.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.drop_target.dnd_bind('<<Drop>>', self.on_drop)  # type: ignore[attr-defined]

        # 移除/清空按钮
        remove_frame = ttk.Frame(self.left_panel)
        remove_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(remove_frame, text="✕ Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(remove_frame, text="✕✕ Clear All", command=self.clear_all).pack(side=tk.LEFT)

        # --- Scrollable Controls ---
        self.canvas_ctrl = tk.Canvas(self.left_panel)
        self.scrollbar = ttk.Scrollbar(self.left_panel, orient="vertical", command=self.canvas_ctrl.yview)
        self.scrollable_frame = ttk.Frame(self.canvas_ctrl)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas_ctrl.configure(scrollregion=self.canvas_ctrl.bbox("all")))
        self.canvas_ctrl.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_ctrl.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Plot Area with Toolbar ---
        self.toolbar_frame = ttk.Frame(self.right_panel)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        tk.Label(self.toolbar_frame, text="Preview:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)

        # Save Button —— 先靠右打包，确保始终可见
        self.btn_save = ttk.Menubutton(self.toolbar_frame, text="💾 Save Results...")
        self.save_menu = tk.Menu(self.btn_save, tearoff=0)
        self.save_menu.add_command(label="Save Plot Image (PNG)", command=self.save_plot_image)
        self.save_menu.add_command(label="Save Mixed Data (CSV)", command=self.save_mixed_data)
        self.btn_save["menu"] = self.save_menu
        self.btn_save.pack(side=tk.RIGHT, padx=5)

        # 混合模式切换
        mode_frame = ttk.Frame(self.toolbar_frame)
        mode_frame.pack(side=tk.LEFT, padx=15)
        ttk.Label(mode_frame, text="Mode:", font=("Arial", 9)).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Aligned by E0", value="Aligned",
                        variable=self.mix_mode_var, command=self._on_mode_or_norm_changed).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_frame, text="Direct (raw energy)", value="Direct",
                        variable=self.mix_mode_var, command=self._on_mode_or_norm_changed).pack(side=tk.LEFT, padx=4)

        # 强度归一化开关 (影响混合与保存)
        self.chk_norm_intensity = ttk.Checkbutton(self.toolbar_frame, text="Normalize Intensity (max=1)",
                                                   variable=self.norm_intensity_var,
                                                   command=self._on_mode_or_norm_changed)
        self.chk_norm_intensity.pack(side=tk.LEFT, padx=15)

        # 归一化视图开关
        self.chk_norm_view = ttk.Checkbutton(self.toolbar_frame, text="Normalize View (Shape Compare)",
                                             variable=self.norm_view_var, command=self.update_plot_fast)
        self.chk_norm_view.pack(side=tk.LEFT, padx=15)

        # Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, alpha=0.3)
        self.ax.text(0.5, 0.5, "Drag & Drop Files to Start", ha='center', transform=self.ax.transAxes)

    # --- Save Functionality ---
    def save_plot_image(self):
        if not self.components: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if file_path:
            self.fig.savefig(file_path, dpi=300)
            messagebox.showinfo("Saved", f"Plot saved to:\n{file_path}")

    def save_mixed_data(self):
        if not self.components: return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
        if file_path:
            # 保存的是当前 components 的真实加权和
            # (已根据 Normalize Intensity 开关决定是否归一化)
            total_mu = np.zeros_like(self.common_energy)
            for i, comp_mu in enumerate(self.components):
                w = self.weights_vars[i].get()
                total_mu += w * comp_mu

            data_to_save = np.column_stack((self.common_energy, total_mu))
            mode_txt = "Aligned by E0" if self.mix_mode_var.get() == "Aligned" else "Direct (raw energy)"
            norm_txt = "True" if self.norm_intensity_var.get() else "False"
            header = (f"# Mix Mode: {mode_txt}\n"
                      f"# Intensity Normalized: {norm_txt}\n"
                      f"Energy(eV), Mixed_Intensity_WeightedSum")
            np.savetxt(file_path, data_to_save, delimiter=",", header=header, comments='')
            note = "intensity-normalized" if self.norm_intensity_var.get() else "raw intensity"
            messagebox.showinfo("Saved", f"Data saved to:\n{file_path}\n(Mode: {mode_txt}; {note})")

    # --- Normalization ---
    def normalize_weights(self):
        current_vals = [var.get() for var in self.weights_vars]
        total = sum(current_vals)
        if total <= 0: return
        for i, val in enumerate(current_vals):
            self.weights_vars[i].set(round(val / total, 4))
        # 这里的 set 会触发 trace 自动绘图

    # --- File Loading & Alignment Logic ---
    def parse_dropped_files(self, event_data):
        pattern = r'\{.*?\}|\S+'
        matches = re.findall(pattern, event_data)
        return [m.strip('{}') for m in matches]

    def on_drop(self, event):
        files = self.parse_dropped_files(event.data)
        if files: self.load_files(files)

    # --- Remove / Clear ---
    def remove_selected(self):
        """移除 Listbox 中选中的文件。索引与 file_entries 对应。"""
        if not self.file_entries:
            return
        sel = list(self.drop_target.curselection())
        if not sel:
            messagebox.showinfo("Remove", "请先在列表中选中要移除的文件。")
            return
        try:
            # 从大到小删除，避免索引错位
            for idx in sorted(sel, reverse=True):
                if 0 <= idx < len(self.file_entries):
                    del self.file_entries[idx]

            if not self.file_entries:
                self.clear_all()
                return

            saved_weights = [var.get() for var in self.weights_vars]
            # 删除选中项后保留剩余文件对应的权重
            remaining_weights = [w for i, w in enumerate(saved_weights) if i not in sel]
            self.rebuild_components()
            # 优先恢复剩余权重；数量不匹配时回退到均分
            if remaining_weights and len(remaining_weights) == len(self.weights_vars):
                for var, w in zip(self.weights_vars, remaining_weights):
                    var.set(w)
            else:
                init_val = round(1.0 / len(self.weights_vars), 4)
                for var in self.weights_vars:
                    var.set(init_val)
            self.init_plot_elements()
            self.update_plot_fast()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear_all(self):
        """清空所有载入的文件，恢复初始状态。"""
        self.file_entries = []
        self.components = []
        self.weights_vars = []
        self.line_objects = []
        self.line_sum = None
        self.common_energy = None
        self.is_plot_initialized = False

        self.drop_target.delete(0, tk.END)
        self.drop_target.insert(0, "   [ Drop Files Here ]")
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.ax.clear()
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, alpha=0.3)
        self.ax.text(0.5, 0.5, "Drag & Drop Files to Start", ha='center', transform=self.ax.transAxes)
        self.canvas_plot.draw()

    def load_files(self, file_paths):
        self.components = []
        self.weights_vars = []
        self.line_objects = []
        self.line_sum = None
        self.is_plot_initialized = False
        self.file_entries = []

        self.drop_target.delete(0, tk.END)
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.ax.clear()

        try:
            for fp in file_paths:
                x, y, deriv, e0_header = robust_load_csv(fp)
                if x is None:
                    continue
                fname = os.path.basename(fp)
                self.file_entries.append({'x': x, 'y': y, 'deriv': deriv,
                                           'e0_header': e0_header, 'name': fname})

            if not self.file_entries:
                return

            # 载入时自动均分权重（控件尚未创建，先占位；rebuild 后补建）
            self.rebuild_components()
            n = len(self.file_entries)
            init_val = round(1.0 / n, 4)
            for var in self.weights_vars:
                var.set(init_val)

            self.init_plot_elements()
            self.update_plot_fast()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_mode_or_norm_changed(self):
        """混合模式或强度归一化开关变化时，从原始数据重建分量。权重保持不变。"""
        if not self.file_entries:
            return
        try:
            saved_weights = [var.get() for var in self.weights_vars]
            self.rebuild_components()
            # 恢复权重（分量数应一致）
            if saved_weights and len(saved_weights) == len(self.weights_vars):
                for var, w in zip(self.weights_vars, saved_weights):
                    var.set(w)
            self.init_plot_elements()
            self.update_plot_fast()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def rebuild_components(self):
        """根据当前模式(Aligned/Direct)和归一化开关，从 file_entries 重建 components/common_energy。

        权重控件保持不变；listbox 与分量控件标签随之刷新。
        """
        self.components = []
        self.common_energy = None

        mode = self.mix_mode_var.get()
        norm_intensity = self.norm_intensity_var.get()

        # --- 计算每文件对齐后的能量轴与所用的 E0 ---
        processed = []  # [{'x','y','tag','e0','name'}]
        for item in self.file_entries:
            x = np.array(item['x'])
            y = np.array(item['y'])
            deriv = item['deriv']
            e0_header = item['e0_header']
            name = item['name']

            e0 = 0.0
            tag = "[Raw]"
            used_e0 = None

            if mode == "Aligned":
                # 优先用文件头 E0_VAL；否则回退到导数最大值法
                if e0_header is not None:
                    e0 = float(e0_header)
                    x = x - e0
                    tag = "[Aligned]"
                    used_e0 = e0
                elif deriv is not None:
                    idx = int(np.argmax(deriv))
                    e0 = float(x[idx])
                    x = x - e0
                    tag = "[Aligned*]"  # 标 * 表示导数法而非文件头
                    used_e0 = e0
                else:
                    tag = "[Raw]"
                    logger.warning("File %s lacks E0_VAL header and 3rd column — E0 alignment skipped.", name)
            else:
                # Direct 模式：不平移
                tag = "[Direct]"

            processed.append({'x': x, 'y': y, 'tag': tag, 'e0': used_e0, 'name': name})

        if not processed:
            return

        # --- 公共能量网格 ---
        # Direct 模式：若所有文件能量轴完全一致，直接使用原始网格(不插值)
        all_grids_equal = all(np.allclose(p['x'], processed[0]['x']) for p in processed)
        if all_grids_equal:
            self.common_energy = np.array(processed[0]['x'], dtype=float)
            for p in processed:
                mu = np.array(p['y'], dtype=float)
                if norm_intensity and np.max(mu) > 1e-9:
                    mu = mu / np.max(mu)
                self.components.append(mu)
        else:
            min_e, max_e = -np.inf, np.inf
            for p in processed:
                min_e = max(min_e, p['x'][0])
                max_e = min(max_e, p['x'][-1])
            if max_e <= min_e:
                raise ValueError("公共能量范围为空：各文件能量区间无重叠。请检查 E0 对齐或使用 Direct 模式。")

            self.common_energy = np.linspace(min_e, max_e, 2000)
            for p in processed:
                f = interp1d(p['x'], p['y'], kind='cubic',
                             bounds_error=False,
                             fill_value=(p['y'][0], p['y'][-1]))
                mu_interp = f(self.common_energy)
                if norm_intensity and np.max(mu_interp) > 1e-9:
                    mu_interp = mu_interp / np.max(mu_interp)
                self.components.append(mu_interp)

        # --- 刷新 listbox 与分量控件 ---
        self.drop_target.delete(0, tk.END)
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.weights_vars = []

        for i, p in enumerate(processed):
            self.drop_target.insert(tk.END, f"{p['name']} {p['tag']}")
            self.create_component_control(p['name'], i,
                                          is_aligned=(p['tag'].startswith('[Aligned')),
                                          e0_val=p['e0'])

        # 权重变量绑定 trace，自动刷新绘图
        for var in self.weights_vars:
            var.trace("w", lambda *args: self.update_plot_fast())

    def create_component_control(self, name, index, is_aligned, e0_val):
        frame = ttk.LabelFrame(self.scrollable_frame, text=f"{index+1}. {name}", padding=5)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        info_text = f"E0 Shift: -{e0_val:.1f} eV" if is_aligned else "Raw Energy"
        tk.Label(frame, text=info_text, font=("Arial", 8), fg="gray").pack(side=tk.TOP, anchor='w')

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        
        var = tk.DoubleVar(value=0.0)
        self.weights_vars.append(var)

        ttk.Button(row, text="-", width=3, command=lambda: self.adjust_weight(index, -0.05)).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=var, width=8, justify='center')
        entry.pack(side=tk.LEFT, padx=5)
        entry.bind('<Return>', lambda e: self.update_plot_fast())
        ttk.Button(row, text="+", width=3, command=lambda: self.adjust_weight(index, 0.05)).pack(side=tk.LEFT)

    # --- Plotting ---
    def init_plot_elements(self):
        self.ax.clear()
        self.line_objects = []
        
        # 组分线 (虚线)
        for i in range(len(self.components)):
            l, = self.ax.plot([], [], '--', alpha=0.4, linewidth=1)
            self.line_objects.append(l)
            
        # 总和线 (实线)
        self.line_sum, = self.ax.plot([], [], 'r-', linewidth=2.5, label='Mixed')
        
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity")
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(self.common_energy[0], self.common_energy[-1])
        
        self.is_plot_initialized = True
    
    # 增加权重调整辅助函数
    def adjust_weight(self, index, delta):
        try:
            val = self.weights_vars[index].get()
            new_val = max(0.0, val + delta)
            self.weights_vars[index].set(round(new_val, 4))
            # trace 会自动调用 update_plot_fast
        except: pass

    def update_plot_fast(self, *args):
        if not self.is_plot_initialized or not self.components: return

        total_w = sum([var.get() for var in self.weights_vars])
        self.lbl_sum.config(text=f"Sum: {total_w:.3f}", fg="green" if abs(total_w-1.0)<0.001 else "red")

        total_mu = np.zeros_like(self.common_energy)
        norm_view = self.norm_view_var.get() # 获取复选框状态

        for i, comp_mu in enumerate(self.components):
            try: w = self.weights_vars[i].get()
            except: w = 0.0
            
            # 计算总和 (始终使用真实值)
            total_mu += w * comp_mu
            
            # [核心逻辑修复] 更新分量显示
            # 如果是 Normalized View，现场归一化以便比较形状
            if norm_view:
                disp_y = comp_mu
                if np.max(disp_y) > 1e-9: # 避免除零
                    disp_y = disp_y / np.max(disp_y)
                self.line_objects[i].set_data(self.common_energy, disp_y)
            else:
                # 否则显示真实的贡献 (Intensity = Weight * RawIntensity)
                self.line_objects[i].set_data(self.common_energy, w * comp_mu)

        # [核心逻辑修复] 更新总和显示
        plot_y = total_mu
        label_txt = f'Mixed (Sum={total_w:.2f})'
        
        if norm_view:
            if np.max(total_mu) > 1e-9:
                plot_y = total_mu / np.max(total_mu)
            label_txt = "Mixed (Normalized)"

        self.line_sum.set_data(self.common_energy, plot_y)
        self.line_sum.set_label(label_txt)
        self.ax.legend(loc='upper right')

        # 自动缩放 Y 轴
        # 移除之前的 0-1 限制，改为根据实际数据范围缩放
        if len(plot_y) > 0:
            max_y = np.max(plot_y)
            if max_y <= 0: max_y = 1.0
            
            # 为了更好的视觉效果，留出 10% 的余量
            self.ax.set_ylim(0, max_y * 1.1)
        
        self.canvas_plot.draw()