# Data-analysis-SACs-therotical-XAFS

一套面向 CP2K 基于 LR-TDDFT 理论的 &XAS_TDP 模块计算理论 XAFS 数据处理的 Python 工具集，实现从原始 `.pdos`/`.spectrum` 输出到图谱的数据处理，含谱线展宽与绘制、混合光谱的构建、PDOS 区域积分、自旋/轨道分辨分析和基态/激发态对比等模块。

## 功能特性

- **CP2K 输出解析**：自动识别 `.pdos` 文件格式，自适应不同原子种类的轨道列数（s/p/d/f）
- **PDOS 区域积分**：在指定能量窗口内对某原子某轨道的投影权重求和，支持自旋/轨道分辨
- **自旋分辨处理**：基态（独立自旋流形）与激发态（同一空间轨道的 α/β 划分）差异化处理策略
- **GS/ES 对比**：基态与 core-hole 激发态并排对比，分析芯空穴对轨道分布的影响
- **谱线展宽**：Lorentzian与 Gaussian两种展宽方式可选
- **混合光谱构建**：两种混合方式可选，强度归一化
- **GUI 集成**：基于 tkinter 的图形界面，支持拖拽导入

## 目录结构

```
Data-analysis-SACs-therotical-XAFS/
├── main.py                      # GUI 入口
├── pdos_region_analysis.py      # PDOS 区域积分核心
├── pdos_tool.py                 # 激发态 PDOS 数据处理（含 spin sum 插值）
├── pdos_gs_tool.py              # 基态 PDOS 数据处理
├── pdos_gs_es_compare.py        # GS/ES 对比分析
├── mixer_tool.py                # 轨道混合与模式切换
├── plot_xas.py                  # XAS 谱绘制
├── process_xas.py              # .spectrum 原始数据解析
├── presets_manager.py           # 预设参数管理
├── export_utils.py              # 数据导出
├── _analysis_run.py             # 端到端验证演示
├── _verify_fe_p_align.py       # Fe 投影对齐验证
├── _verify_region_analysis.py   # 区域积分逻辑验证
├── settings/                    # 配置文件
├── requirements.txt
└── README.md
```

## 环境依赖

- Python 3.12+
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7
- tkinterdnd2 >= 0.4

安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/CCCLers/Data-analysis-SACs-therotical-XAFS.git
cd Data-analysis-SACs-therotical-XAFS

# 安装依赖
pip install -r requirements.txt

# 启动 GUI
python main.py
```

## 支持的数据格式

### CP2K `.pdos` 文件

```
# Projected DOS for atomic kind <KIND> at iteration step i = 0, E(Fermi) = <x> a.u.
# MO Eigenvalue [a.u.]  Occupation  s py pz px d-2 d-1 d0 d+1 d+2 [f-3 ... f+3]
  1  -0.5   2.0000  0.01 0.02 ...
```

- 列 0：MO 序号
- 列 1：能量 [a.u.]
- 列 2：占据数
- 列 3+：轨道投影权重（Mulliken/Löwdin 投影，逐 MO 归一化）

工具自动适配不同轨道基组（C/N: s+p+d = 9 列；Fe: s+p+d+f = 16 列）。

### CP2K `.spectrum` 文件

含 `====` 分隔行，`np.loadtxt` 无法直接读取，工具内置 `parse_spectrum_data()` 处理。

## 开发说明

本工具的**算法设计、科学方法论构建与正确性验证**由本人完成。代码实现过程中借助了 AI 辅助工具，所有代码均经本人审查、调试与验证。工具开发遵循"人工主导设计、AI 辅助实现"的原则。

## 数据说明

本仓库**不包含**课题组计算数据文件（`.pdos`/`.spectrum`/`.csv`），这些数据归属课题组。如需复现，请使用符合格式的自有 CP2K 计算输出。
