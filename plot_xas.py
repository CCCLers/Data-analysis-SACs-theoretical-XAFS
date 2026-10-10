import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox

logger = logging.getLogger(__name__)

class InteractiveXASPlotter:
    def __init__(self, master_window, csv_files):
        self.master = master_window
        self.csv_files = csv_files
        self.is_3d = False 
        self.azim_val = tk.DoubleVar(value=-60)
        self.elev_val = tk.DoubleVar(value=25)
        self.default_colors = plt.cm.tab20.colors 
        self.data_objects = [] 
        self._load_and_align_data()
        self._setup_ui()
        self.plot()

    def _load_and_align_data(self):
        """读取CSV，计算E0，并存储初始样式"""
        for i, f in enumerate(self.csv_files):
            try:
                # === 智能读取 E0 Header 并确定跳过的行数 ===
                e0_from_header = None
                n_skip = 1  # 至少跳过列名行
                
                with open(f, 'r') as fp:
                    first_line = fp.readline()
                    if first_line.startswith("# E0_VAL:"):
                        try:
                            val_str = first_line.split(":")[1].strip()
                            e0_from_header = float(val_str)
                            n_skip = 2  # 注释行 + 列名行
                        except:
                            pass
                
                raw_data = np.loadtxt(f, delimiter=',', skiprows=n_skip)
                
                if raw_data.ndim == 1 or raw_data.size == 0: 
                    continue
                if raw_data.shape[1] < 2: 
                    continue
                
                x = raw_data[:, 0]
                y = raw_data[:, 1]
                
                # === E0 对齐逻辑 ===
                if e0_from_header is not None:
                    # 优先使用 Header 中的值
                    e0 = e0_from_header
                    logger.info(f"[{os.path.basename(f)}] Aligned by Header E0: {e0:.2f} eV")
                else:
                    # Fallback 到自动计算
                    if raw_data.shape[1] >= 3:
                        deriv = raw_data[:, 2]
                        e0 = x[np.argmax(deriv)]
                    else:
                        deriv = np.gradient(y, x)
                        e0 = x[np.argmax(deriv)]
                    logger.info(f"[{os.path.basename(f)}] Aligned by Calculated E0: {e0:.2f} eV")
                
                x_aligned = x - e0
                label = os.path.splitext(os.path.basename(f))[0]

                # 样式分配
                color = self.default_colors[i % len(self.default_colors)]
                
                self.data_objects.append({
                    'x': x_aligned,
                    'y': y,
                    'label': label,
                    'e0_val': e0,
                    'color': color,          
                    'linewidth': 2.5,
                    'linestyle': '-',
                    'alpha': 1.0, 
                    'z_order': i             
                })
            except Exception as e:
                logger.error(f"Error loading {f}: {e}")

    def _setup_ui(self):
        toolbar_frame = ttk.Frame(self.master)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar_frame, text="切换 2D / 3D 视图", command=self.toggle_3d).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="打开样式编辑器", command=self.open_style_editor).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="保存图片", command=self.save_figure).pack(side=tk.RIGHT, padx=5)

        self.angle_frame = ttk.LabelFrame(self.master, text="3D 视角调节")
        
        ttk.Label(self.angle_frame, text="水平旋转 (Azimuth):").pack(side=tk.LEFT, padx=5)
        self.sl_azim = ttk.Scale(self.angle_frame, from_=-180, to=180, variable=self.azim_val, 
                                 orient='horizontal', length=200, command=lambda v: self.update_view())
        self.sl_azim.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.angle_frame, text="俯仰角度 (Elevation):").pack(side=tk.LEFT, padx=5)
        self.sl_elev = ttk.Scale(self.angle_frame, from_=-90, to=90, variable=self.elev_val, 
                                 orient='horizontal', length=200, command=lambda v: self.update_view())
        self.sl_elev.pack(side=tk.LEFT, padx=5)

        self.plot_frame = ttk.Frame(self.master)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig = plt.Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.mpl_toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.mpl_toolbar.update()

    def toggle_3d(self):
        self.is_3d = not self.is_3d
        if self.is_3d:
            self.angle_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5, after=self.plot_frame) 
        else:
            self.angle_frame.pack_forget() 
        self.plot()

    def update_view(self):
        if self.is_3d and hasattr(self, 'ax_3d'):
            self.ax_3d.view_init(elev=self.elev_val.get(), azim=self.azim_val.get())
            self.canvas.draw_idle()

    def plot(self):
        self.fig.clear()
        self.fig.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.15)

        if self.is_3d:
            self.ax_3d = self.fig.add_subplot(111, projection='3d')
            self._plot_3d(self.ax_3d)
        else:
            ax = self.fig.add_subplot(111)
            self._plot_2d(ax)
            
        self.canvas.draw()

    def _plot_2d(self, ax):
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        try:
            ax.set_box_aspect(0.6)
        except:
            pass 

        lines = []
        labels = []
        for data in self.data_objects:
            l, = ax.plot(
                data['x'], data['y'],
                color=data['color'],
                linewidth=data['linewidth'],
                linestyle=data['linestyle'],
                alpha=data['alpha']
            )
            lines.append(l)
            labels.append(data['label'])
        
        ax.set_xlabel("Relative Energy (E - E0) [eV]", fontsize=12)
        ax.set_ylabel("Absorption", fontsize=12)
        ax.set_title("2D Comparison", fontsize=14)
        ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.03, 1.0), borderaxespad=0.)

    def _plot_3d(self, ax):
        ax.view_init(elev=self.elev_val.get(), azim=self.azim_val.get())
        try:
            ax.set_box_aspect((2.5, 2, 1))
        except:
            pass 

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        lines = []
        labels = []
        
        for i, data in enumerate(self.data_objects):
            y_indices = np.full_like(data['x'], i)
            l, = ax.plot(
                data['x'],      
                y_indices,      
                data['y'],      
                color=data['color'],
                linewidth=data['linewidth'],
                linestyle=data['linestyle'],
                alpha=data['alpha']
            )
            lines.append(l)
            labels.append(data['label'])

        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Structure ID")
        ax.set_zlabel("Intensity")
        ax.set_title("3D Waterfall Plot")
        ax.set_yticks([]) 
        ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 0.9), borderaxespad=0.)

    def open_style_editor(self):
        editor = tk.Toplevel(self.master)
        editor.title("Style Editor")
        editor.geometry("900x500") 

        canvas = tk.Canvas(editor)
        scrollbar = ttk.Scrollbar(editor, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        headers = ["Label", "Color", "Width", "Style", "Alpha"]
        for i, h in enumerate(headers):
            ttk.Label(scrollable_frame, text=h, font=('bold')).grid(row=0, column=i, padx=5, pady=5)

        for i, data in enumerate(self.data_objects):
            row = i + 1
            ttk.Label(scrollable_frame, text=data['label'][:20]).grid(row=row, column=0, padx=5)
            
            btn_color = tk.Button(scrollable_frame, text="■", fg=self._to_hex(data['color']), 
                                  command=lambda idx=i, b=None: self._pick_color(idx, b))
            btn_color.config(command=lambda idx=i, btn=btn_color: self._pick_color(idx, btn))
            btn_color.grid(row=row, column=1)

            var_width = tk.DoubleVar(value=data['linewidth'])
            sp_width = ttk.Spinbox(scrollable_frame, from_=0.5, to=5.0, increment=0.5, 
                                   textvariable=var_width, width=5)
            sp_width.bind('<Return>', lambda event, idx=i, v=var_width: self._update_attr(idx, 'linewidth', v.get()))
            sp_width.bind('<FocusOut>', lambda event, idx=i, v=var_width: self._update_attr(idx, 'linewidth', v.get()))
            sp_width.grid(row=row, column=2)

            var_style = tk.StringVar(value=data['linestyle'])
            cb_style = ttk.Combobox(scrollable_frame, textvariable=var_style, 
                                    values=['-', '--', '-.', ':'], width=5, state="readonly")
            cb_style.bind("<<ComboboxSelected>>", lambda event, idx=i, v=var_style: self._update_attr(idx, 'linestyle', v.get()))
            cb_style.grid(row=row, column=3)

            var_alpha = tk.DoubleVar(value=data['alpha'])
            sc_alpha = ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=var_alpha, orient='horizontal',
                                 command=lambda val, idx=i: self._update_attr(idx, 'alpha', float(val)))
            sc_alpha.grid(row=row, column=4)

        ttk.Button(editor, text="Refresh Plot", command=self.plot).pack(pady=5, side=tk.BOTTOM)

    def _to_hex(self, color):
        try: return matplotlib.colors.to_hex(color)
        except: return "#000000"

    def _pick_color(self, index, btn_widget):
        curr_color = self.data_objects[index]['color']
        color = colorchooser.askcolor(initialcolor=self._to_hex(curr_color), title="Choose Line Color")
        if color[1]: 
            hex_color = color[1]
            self.data_objects[index]['color'] = hex_color
            btn_widget.config(fg=hex_color) 
            self.plot() 

    def _update_attr(self, index, key, value):
        self.data_objects[index][key] = value
        self.plot()

    def save_figure(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Plot saved to {path}")