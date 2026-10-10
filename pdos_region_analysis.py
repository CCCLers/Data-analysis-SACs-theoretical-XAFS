"""
PDOS 能量区域积分分析 (Region Integration: Centroid & Area)
================================================================
新增功能:
  1. 计算"特定能量区域"内的积分中心 (centroid, 一阶矩) 与积分面积 (area, 投影权重之和)
  2. 计算各轨道分量的积分值及占比 (s / p / px / py / pz / d / d_xy / d_yz / d_z2 / d_xz / d_x2y2 / f)

积分定义 (离散 PDOS 棍谱的严格积分):
    area     = Σ_i w_i                      (窗口内所有 MO 的投影权重之和)
    centroid = Σ_i E_i·w_i / Σ_i w_i        (窗口内的加权平均能量 = 一阶矩)
其中 w_i 为第 i 个分子轨道在所选轨道分量上的投影权重。

占比定义 (分母固定, 不受勾选影响, 便于跨结构比较):
    % of total = area / (s + p + d + f)          × 100
    % of p+d   = area / (p + d)                  × 100     <- 对 p 分量即 p/(p+d)
    % of d     = area / d                        × 100     <- 仅 d 子分量

能量参考 (reference):
  - GS 文件 (*charge*.pdos)  : 解析得到的能量为"相对费米能级"(eV)
  - ES 文件 (*-x-*.pdos)     : 解析得到的能量为"激发能"(绝对 eV, 与 XAS 能量轴一致)
  - 'aligned' : Fermi→E0 对齐 (GS 相对费米; ES 相对 E0)
  - 'fe_p'    : Fe-p→Fe-p 对齐, 与 GS vs ES Compare 工具的第二种方案一致。
                δ = ES_p(rel E0) − GS_p(rel Fermi), 加到 GS 轴上,
                使 GS 未占据 Fe-p 峰与 ES Fe-p 峰重合 (ES 仍相对 E0)。
  - 'raw'     : 保留解析原值 (注意: GS 侧仍为相对费米能级)

能量窗口:
  GS 与 ES 各有一套独立的 Emin/Emax (界面第 3 节两行), State=both 时各用自己那一行,
  避免把 GS 的"占据价带窗口"误套到 ES 的"边前终态窗口"上。

命令行批处理 (无需 GUI):
    python pdos_region_analysis.py --dir . --atom Fe --state both ^
        --window-gs "occ:-13.6:0;unocc:0:8.2" ^
        --window-es "deep:-15:-8;pre:-8:-2;wl:-2:20" ^
        --components "s,p,d,d_xy,d_yz,d_z2,d_xz,d_x2y2" --spin combined --out result.csv

  --window 同时对两态生效; --window-gs / --window-es 按态覆盖。
  --reference fe_p 时自动计算并对 GS 施加 Fe-p→Fe-p 位移。
"""

import os
import re
import glob
import csv
import logging
import argparse
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    _HAS_TK = True
except Exception:  # pragma: no cover - 无 GUI 环境时仍可用命令行模式
    _HAS_TK = False

try:
    from tkinterdnd2 import DND_FILES
    _HAS_DND = True
except Exception:  # pragma: no cover
    _HAS_DND = False


# =========================================================================
# 常量: 轨道分量定义
# =========================================================================
# key -> (显示标签, 数据来源, 说明)
#   'group'   : 取自 parse 结果中的 'groups' (已按 s/p/d/f 与 d 子分量聚合)
#   'orbital' : 取自 'orbitals' (CP2K 原始列名)
COMPONENT_DEFS = OrderedDict([
    ('s',      ('s',            'group')),
    ('p',      ('p',            'group')),
    ('px',     ('px',           'orbital')),
    ('py',     ('py',           'orbital')),
    ('pz',     ('pz',           'orbital')),
    ('d',      ('d',            'group')),
    ('d_xy',   ('d_xy  (d-2)',  'group')),
    ('d_yz',   ('d_yz  (d-1)',  'group')),
    ('d_z2',   ('d_z2  (d0)',   'group')),
    ('d_xz',   ('d_xz  (d+1)',  'group')),
    ('d_x2y2', ('d_x2y2 (d+2)', 'group')),
    ('f',      ('f',            'group')),
])

D_KEYS = ('d_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2')
GROUP_KEYS = ('s', 'p', 'd', 'f')

DEFAULT_COMPONENTS = ['s', 'p', 'd']
FULL_COMPONENTS = ['s', 'p', 'd', 'd_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2']

# 预设能量窗口 (相对所选参考的 eV) —— GS 与 ES 分开, 避免同一组数字误套到两态
GS_WINDOW_PRESETS = OrderedDict([
    ('Occupied',       (-13.6,  0.0)),
    ('Unoccupied',     (  0.0,  8.2)),
    ('Full valence',   (-13.6,  8.2)),
])
ES_WINDOW_PRESETS = OrderedDict([
    ('Deep pre-edge',  (-15.0, -8.0)),
    ('Pre-edge',       ( -8.0, -2.0)),
    ('Main / WL',      ( -2.0, 20.0)),
    ('Full',           (-15.0, 20.0)),
])

# 默认窗口 (GS: 占据价带; ES: 预边)
DEFAULT_WINDOW_GS = (-13.6, 0.0)
DEFAULT_WINDOW_ES = (-8.0, -2.0)

# 兼容旧引用: 全部预设的合集
WINDOW_PRESETS = OrderedDict(list(GS_WINDOW_PRESETS.items()) + list(ES_WINDOW_PRESETS.items()))

SPIN_MODES = ('combined', 'alpha', 'beta')

# 能量参考轴模式
REFERENCE_MODES = ('aligned', 'fe_p', 'raw')
REFERENCE_HINTS = {
    'aligned': 'aligned = Fermi→E0: GS 相对费米(0), ES 相对 E0(0).',
    'fe_p':    'fe_p = Fe-p→Fe-p: GS 的未占据 Fe-p 峰对齐到 ES 的 Fe-p 峰\n'
               '(δ 加到 GS 轴; ES 仍相对 E0). 用于与 GS vs ES Compare 工具一致.',
    'raw':     'raw = GS 相对费米; ES 保留绝对激发能 (~7000 eV).',
}


# =========================================================================
# 加载与解析
# =========================================================================
def load_e0_from_csv(directory):
    """从目录中的 XAS csv 头部读取 E0 (优先 *Lorentzian.csv)。返回 float 或 None。"""
    if not directory or not os.path.isdir(directory):
        return None
    candidates = sorted(glob.glob(os.path.join(directory, '*Lorentzian.csv')))
    candidates += sorted(glob.glob(os.path.join(directory, '*.csv')))
    for fp in candidates:
        try:
            with open(fp, 'r') as f:
                first = f.readline().strip()
            if first.startswith('# E0_VAL:'):
                return float(first.split(':')[1].strip())
        except Exception:
            continue
    return None


def load_pdos(directory, kind='gs', fermi_au=None):
    """加载目录下的 PDOS 文件。

    kind = 'gs' : 匹配 *charge*.pdos
    kind = 'es' : 匹配 *-x-*.pdos
    fermi_au    : 显式指定的费米能级 (a.u.); None 表示使用文件头中的值
    返回 parse_pdos_file_list 的结果, 或 None
    """
    import pdos_tool as es
    if not directory or not os.path.isdir(directory):
        return None
    if kind == 'gs':
        files = [f for f in sorted(glob.glob(os.path.join(directory, '*.pdos')))
                 if 'charge' in os.path.basename(f)]
    else:
        files = [f for f in sorted(glob.glob(os.path.join(directory, '*.pdos')))
                 if re.search(r'-x[-_]', os.path.basename(f))]
    if not files:
        return None
    data = es.parse_pdos_file_list(files)
    if data is not None and fermi_au is not None:
        data['fermi_au'] = float(fermi_au)
        data['fermi_ev'] = float(fermi_au) * 27.2114
    return data


def resolve_atom(pdos, atom):
    """在数据集中查找原子 (自动处理 Fe <-> Fex)。找不到返回 None。"""
    if pdos is None:
        return None
    atoms = pdos.get('atoms', {})
    if atom in atoms:
        return atom
    alt = {'Fe': 'Fex', 'Fex': 'Fe'}.get(atom)
    if alt and alt in atoms:
        return alt
    return None


def list_canonical_atoms(pdos_gs, pdos_es):
    """合并 GS/ES 两个数据集的原子列表, Fex 归一为 Fe。"""
    names = []
    for pdos in (pdos_gs, pdos_es):
        if not pdos:
            continue
        for a in pdos.get('atoms', {}):
            canon = 'Fe' if a == 'Fex' else a
            if canon not in names:
                names.append(canon)
    order = {'Fe': 0, 'N': 1, 'C': 2}
    return sorted(names, key=lambda x: (order.get(x, 99), x))


def energy_axis(energy, state, reference='aligned', e0=None, shift=0.0, fe_p_delta=None):
    """把解析得到的能量换算到所选参考轴。返回 ndarray 或 None(无法对齐)。

    reference 取值:
      'aligned'  : GS 不变(相对费米, Fermi=0); ES 减 E0(相对E0, E0=0)  —— Fermi↔E0 对齐
      'fe_p'     : GS 额外加 fe_p_delta(使 GS Fe-p 峰对齐到 ES Fe-p 峰); ES 同 aligned
      'raw'      : GS 不变(相对费米); ES 不变(绝对激发能)
    """
    if energy is None:
        return None
    e = np.asarray(energy, dtype=float)
    if state == 'es' and reference in ('aligned', 'fe_p'):
        if e0 is None:
            return None
        e = e - float(e0)
    if state == 'gs' and reference == 'fe_p':
        if fe_p_delta is None:
            return None
        e = e + float(fe_p_delta)
    return e + float(shift or 0.0)


def find_pdos_peak(data, atom, group, unocc_only=False, e_range=None,
                   fwhm=0.8, is_gauss=False, spin='alpha'):
    """找展宽后 PDOS 曲线的峰位。返回能量值(与 data 中相同标度), 找不到返回 None。"""
    if data is None or atom not in data.get('atoms', {}):
        return None
    ad = data['atoms'][atom]
    if spin not in ad:
        return None
    e = ad[spin]['energy'].copy()
    w = ad[spin]['groups'].get(group, np.zeros_like(e)).copy()
    if unocc_only:
        mask = ad[spin]['occupation'] < 0.5
        e = e[mask]
        w = w[mask]
    if e_range is not None:
        mask = (e >= e_range[0]) & (e <= e_range[1])
        if mask.sum() < 3:
            return None
        e = e[mask]
        w = w[mask]
    if len(e) < 3:
        return None
    e_min, e_max = e.min(), e.max()
    pad = (e_max - e_min) * 0.1 or 2.0
    xg = np.linspace(e_min - pad, e_max + pad, 2000)
    dos = broaden_curve(e, w, xg, fwhm, is_gauss)
    return float(xg[np.argmax(dos)])


def compute_fe_p_delta(gs_data, es_data, e0, fwhm=0.8, is_gauss=False):
    """计算 Fe p→Fe p 对齐所需的 GS 能量平移量 δ。

    δ = ES_p_rel − GS_p
    其中 GS_p = GS 未占据 Fe-p 峰(相对费米 eV, alpha)
         ES_p_rel = ES Fe-p 峰(相对 E0 eV, alpha) = ES_p_abs − E0

    GS 能量在 fe_p 轴上 = e_r(相对费米) + δ → GS Fe-p 峰落在 ES_p_rel 处(与 ES 重合)
    ES 能量不变(相对 E0)。

    找不到峰返回 None。
    """
    if e0 is None:
        return None
    gs_atom = resolve_atom(gs_data, 'Fe')
    es_atom = resolve_atom(es_data, 'Fe')   # ES 中解析为 Fex
    if gs_atom is None or es_atom is None:
        return None

    gs_p = find_pdos_peak(gs_data, gs_atom, 'p', unocc_only=True,
                          fwhm=fwhm, is_gauss=is_gauss, spin='alpha')

    # ES p 峰在绝对能量空间, 限定在 XANES 窗口 [E0-25, E0+25] 内搜索
    es_range = (e0 - 25.0, e0 + 25.0)
    es_p_abs = find_pdos_peak(es_data, es_atom, 'p', unocc_only=False,
                              e_range=es_range, fwhm=fwhm, is_gauss=is_gauss,
                              spin='alpha')

    if gs_p is None or es_p_abs is None:
        return None

    es_p_rel = es_p_abs - e0
    delta = es_p_rel - gs_p
    return delta


# =========================================================================
# 核心计算
# =========================================================================
def _component_array(spin_data, key):
    if spin_data is None:
        return None
    _, source = COMPONENT_DEFS.get(key, (None, None))
    if source == 'group':
        return spin_data.get('groups', {}).get(key)
    if source == 'orbital':
        return spin_data.get('orbitals', {}).get(key)
    return None


def region_stats(energy, weights, emin, emax):
    """窗口内的积分面积与积分中心。"""
    nan = float('nan')
    if energy is None or weights is None:
        return {'centroid': nan, 'area': 0.0, 'n_mo': 0, 'e_min': nan, 'e_max': nan}
    e = np.asarray(energy, dtype=float)
    w = np.asarray(weights, dtype=float)
    if e.size == 0 or w.size != e.size:
        return {'centroid': nan, 'area': 0.0, 'n_mo': 0, 'e_min': nan, 'e_max': nan}
    lo, hi = (emin, emax) if emin <= emax else (emax, emin)
    mask = (e >= lo) & (e < hi)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return {'centroid': nan, 'area': 0.0, 'n_mo': 0, 'e_min': nan, 'e_max': nan}
    ww = w[mask]
    area = float(ww.sum())
    ee = e[mask]
    if area <= 0.0:
        return {'centroid': nan, 'area': 0.0, 'n_mo': n,
                'e_min': float(ee.min()), 'e_max': float(ee.max())}
    centroid = float((ee * ww).sum() / area)
    return {'centroid': centroid, 'area': area, 'n_mo': n,
            'e_min': float(ee.min()), 'e_max': float(ee.max())}


def compose_spins(stat_a, stat_b, spin_mode='combined'):
    """按自旋模式合成 area / centroid。"""
    nan = float('nan')
    if spin_mode == 'alpha':
        return dict(stat_a)
    if spin_mode == 'beta':
        return dict(stat_b)
    aa, ab = stat_a['area'], stat_b['area']
    tot = aa + ab
    if tot <= 0:
        return {'centroid': nan, 'area': 0.0,
                'n_mo': stat_a['n_mo'] + stat_b['n_mo'],
                'e_min': nan, 'e_max': nan}
    ca = stat_a['centroid'] if np.isfinite(stat_a['centroid']) else 0.0
    cb = stat_b['centroid'] if np.isfinite(stat_b['centroid']) else 0.0
    return {'centroid': (ca * aa + cb * ab) / tot, 'area': tot,
            'n_mo': stat_a['n_mo'] + stat_b['n_mo'],
            'e_min': nan, 'e_max': nan}


def _spin_stats(ad, key, emin, emax, state, reference, e0, shift, fe_p_delta=None):
    """对 alpha / beta 分别求窗口统计, 再按 spin_mode 合成。"""
    out = {}
    for spin in ('alpha', 'beta'):
        sd = ad.get(spin)
        if sd is None:
            out[spin] = {'centroid': float('nan'), 'area': 0.0, 'n_mo': 0,
                         'e_min': float('nan'), 'e_max': float('nan')}
            continue
        e = energy_axis(sd.get('energy'), state, reference, e0, shift, fe_p_delta)
        w = _component_array(sd, key)
        out[spin] = region_stats(e, w, emin, emax)
    return out


def component_stats(ad, key, emin, emax, state='gs', reference='aligned',
                    e0=None, shift=0.0, spin_mode='combined', fe_p_delta=None):
    """单分量在窗口内的统计 (含分自旋结果)。"""
    per = _spin_stats(ad, key, emin, emax, state, reference, e0, shift, fe_p_delta)
    comb = compose_spins(per['alpha'], per['beta'], spin_mode)
    res = dict(comb)
    res['area_alpha'] = per['alpha']['area']
    res['area_beta'] = per['beta']['area']
    res['centroid_alpha'] = per['alpha']['centroid']
    res['centroid_beta'] = per['beta']['centroid']
    return res


def region_totals(ad, emin, emax, state='gs', reference='aligned',
                  e0=None, shift=0.0, spin_mode='combined', fe_p_delta=None):
    """组级总量 (s/p/d/f), 用作占比分母。"""
    totals = {}
    for g in GROUP_KEYS:
        totals[g] = component_stats(ad, g, emin, emax, state, reference,
                                    e0, shift, spin_mode, fe_p_delta)['area']
    totals['total'] = sum(totals[g] for g in GROUP_KEYS)
    totals['p+d'] = totals['p'] + totals['d']
    return totals


def analyze_region(pdos, atom, emin, emax, components=None,
                   state='gs', reference='aligned', e0=None, shift=0.0,
                   spin_mode='combined', fe_p_delta=None):
    """对指定原子、指定能量窗口, 计算各轨道分量的积分值/中心/占比。

    返回 (rows, summary):
      rows    : list of dict, 每个分量一行
      summary : dict, 含 p/(p+d)、主导 d 分量、主导 p 分量、总面积等
    """
    if pdos is None:
        return [], {'error': 'no pdos data'}
    real_atom = resolve_atom(pdos, atom)
    if real_atom is None:
        return [], {'error': f'atom {atom} not found'}
    ad = pdos['atoms'][real_atom]
    comps = list(components) if components else list(DEFAULT_COMPONENTS)

    totals = region_totals(ad, emin, emax, state, reference, e0, shift, spin_mode, fe_p_delta)
    tot_all = totals['total']
    tot_pd = totals['p+d']
    tot_d = totals['d']

    rows = []
    for key in comps:
        if key not in COMPONENT_DEFS:
            continue
        st = component_stats(ad, key, emin, emax, state, reference,
                             e0, shift, spin_mode, fe_p_delta)
        label = COMPONENT_DEFS[key][0]
        row = {
            'component': key,
            'label': label,
            'centroid': st['centroid'],
            'area': st['area'],
            'area_alpha': st['area_alpha'],
            'area_beta': st['area_beta'],
            'centroid_alpha': st['centroid_alpha'],
            'centroid_beta': st['centroid_beta'],
            'share_total': (100.0 * st['area'] / tot_all) if tot_all > 0 else float('nan'),
            'share_pd': (100.0 * st['area'] / tot_pd) if tot_pd > 0 else float('nan'),
            'share_d': (100.0 * st['area'] / tot_d) if (tot_d > 0 and key in D_KEYS) else float('nan'),
        }
        rows.append(row)

    # ---- 摘要 ----
    # 主导分量始终从"完整" d 子分量 / p 子分量中判定, 不受勾选影响
    d_rank = [(k, component_stats(ad, k, emin, emax, state, reference,
                                  e0, shift, spin_mode, fe_p_delta)['area']) for k in D_KEYS]
    p_rank = [(k, component_stats(ad, k, emin, emax, state, reference,
                                  e0, shift, spin_mode, fe_p_delta)['area']) for k in ('px', 'py', 'pz')]
    d_rank = [kv for kv in d_rank if kv[1] > 0]
    p_rank = [kv for kv in p_rank if kv[1] > 0]
    summary = {
        'atom': real_atom,
        'state': state,
        'window': (min(emin, emax), max(emin, emax)),
        'spin_mode': spin_mode,
        'area_total_spdf': tot_all,
        'area_p': totals.get('p', 0.0),
        'area_d': totals.get('d', 0.0),
        'area_s': totals.get('s', 0.0),
        'area_f': totals.get('f', 0.0),
        'p_over_pd': (100.0 * totals.get('p', 0.0) / tot_pd) if tot_pd > 0 else float('nan'),
        'dominant_d': (max(d_rank, key=lambda kv: kv[1])[0] if d_rank else None),
        'dominant_p': (max(p_rank, key=lambda kv: kv[1])[0] if p_rank else None),
        'd_areas': dict(d_rank),
        'p_areas': dict(p_rank),
        'n_components': len(rows),
    }
    return rows, summary


# =========================================================================
# 标准窗口汇总 (批量)
# =========================================================================
def standard_windows(state):
    """按数据集类型返回推荐窗口 (相对所选参考)。"""
    if state == 'gs':
        return OrderedDict([
            ('occ',  (-13.6, 0.0)),
            ('unocc', (0.0, 8.2)),
        ])
    return OrderedDict([
        ('deep', (-15.0, -8.0)),
        ('pre',  (-8.0, -2.0)),
        ('wl',   (-2.0, 20.0)),
    ])


def batch_summary(directory, atom='Fe', components=None, spin_mode='combined',
                  e0=None, reference='aligned', shift=0.0, states=('gs', 'es'),
                  windows=None, verbose=True, fe_p_delta=None, fwhm=0.8, is_gauss=False):
    """对目录下的 GS/ES 数据批量计算标准窗口的积分统计。返回 list of dict。

    reference='fe_p' 时自动计算 Fe p→Fe p 对齐位移 δ (要求 GS/ES 均含 Fe)。
    """
    if components is None:
        components = FULL_COMPONENTS
    out = []
    if e0 is None:
        e0 = load_e0_from_csv(directory)

    # Fe p→Fe p 对齐: 需要同时有 GS 与 ES 数据才能定 δ
    if reference == 'fe_p' and fe_p_delta is None:
        gs_tmp = load_pdos(directory, 'gs')
        es_tmp = load_pdos(directory, 'es')
        fe_p_delta = compute_fe_p_delta(gs_tmp, es_tmp, e0, fwhm, is_gauss)
        if verbose:
            if fe_p_delta is None:
                print('[warn] Fe p→Fe p alignment failed (peaks not found); GS rows will be empty')
            else:
                print(f'[info] Fe p→Fe p delta = {fe_p_delta:+.4f} eV (applied to GS axis)')

    for state in states:
        pdos = load_pdos(directory, kind=state)
        if pdos is None:
            if verbose:
                print(f'[skip] {state}: no pdos files in {directory}')
            continue
        real = resolve_atom(pdos, atom)
        if real is None:
            if verbose:
                print(f'[skip] {state}: atom {atom} not found')
            continue
        wins = windows.get(state) if windows else standard_windows(state)
        for wname, (lo, hi) in wins.items():
            rows, summ = analyze_region(pdos, atom, lo, hi, components,
                                        state=state, reference=reference,
                                        e0=e0, shift=shift, spin_mode=spin_mode,
                                        fe_p_delta=fe_p_delta)
            if not rows:
                continue
            rec = {
                'state': state.upper(),
                'window': wname,
                'e_min': min(lo, hi),
                'e_max': max(lo, hi),
                'atom': summ['atom'],
                'spin': spin_mode,
            }
            for r in rows:
                rec[f"area_{r['component']}"] = r['area']
                rec[f"centroid_{r['component']}"] = r['centroid']
                rec[f"share_{r['component']}"] = r['share_pd']
            rec['p_over_pd'] = summ['p_over_pd']
            rec['area_p'] = summ['area_p']
            rec['area_d'] = summ['area_d']
            rec['area_total_spdf'] = summ['area_total_spdf']
            rec['dominant_d'] = summ['dominant_d']
            out.append(rec)
            if verbose:
                print(f"[{state.upper():2s}] {wname:6s} [{rec['e_min']:+.1f},{rec['e_max']:+.1f}] "
                      f"p/(p+d)={summ['p_over_pd']:6.1f}%  area_p={summ['area_p']:.4f} "
                      f"area_d={summ['area_d']:.4f}  dom_d={summ['dominant_d']}")
    return out


def write_rows_csv(filepath, rows, meta=None):
    """把 analyze_region 的 rows 写成表格 CSV。"""
    if not rows:
        return False
    fields = ['component', 'label', 'centroid', 'area', 'area_alpha', 'area_beta',
              'centroid_alpha', 'centroid_beta', 'share_total', 'share_pd', 'share_d']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        if meta:
            for k, v in meta.items():
                f.write(f'# {k}: {v}\n')
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{r[k]:.6g}' if isinstance(r.get(k), float) and np.isfinite(r.get(k))
                            else ('' if r.get(k) is None else r.get(k))) for k in fields})
    return True


def write_batch_csv(filepath, records, meta=None):
    """把 batch_summary 的记录写成 CSV (并集表头)。"""
    if not records:
        return False
    keys = []
    for rec in records:
        for k in rec.keys():
            if k not in keys:
                keys.append(k)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        if meta:
            for k, v in meta.items():
                f.write(f'# {k}: {v}\n')
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rec in records:
            w.writerow({k: (f'{rec[k]:.6g}' if isinstance(rec.get(k), float) and np.isfinite(rec[k])
                            else rec.get(k, '')) for k in keys})
    return True


# =========================================================================
# 绘图辅助
# =========================================================================
def broaden_curve(energy, weights, x_grid, fwhm=0.8, is_gaussian=False):
    """把棍谱展宽为连续曲线 (仅用于可视化)。"""
    y = np.zeros_like(x_grid, dtype=float)
    if energy is None or weights is None:
        return y
    e = np.asarray(energy, float)
    w = np.asarray(weights, float)
    if e.size == 0:
        return y
    sigma = max(float(fwhm), 1e-6) / 2.3548 if is_gaussian else max(float(fwhm), 1e-6) / 2.0
    for ei, wi in zip(e, w):
        if wi <= 0:
            continue
        d = (x_grid - ei) / sigma
        if is_gaussian:
            y += wi * np.exp(-0.5 * d * d)
        else:
            y += wi / (1.0 + d * d)
    return y


# =========================================================================
# GUI
# =========================================================================
if _HAS_TK:
    class PDOSRegionWindow:
        """能量区域积分分析窗口 (积分中心 / 面积 / 各轨道分量占比)。"""

        def __init__(self, master, directory=None):
            self.master = master
            master.title("Region Integration — Centroid / Area / Component Shares")
            # 按屏幕尺寸自适应, 避免窗口大于屏幕导致左侧/底部控件不可达
            try:
                sw = master.winfo_screenwidth()
                sh = master.winfo_screenheight()
            except Exception:
                sw, sh = 1420, 900
            w = max(980, min(1420, sw - 60))
            h = max(620, min(900, sh - 80))
            master.geometry(f'{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}')
            master.minsize(980, 620)

            self.pdos_gs = None
            self.pdos_es = None
            self.e0 = None
            self.fe_p_delta = None      # Fe p→Fe p 对齐位移 (缓存)
            self.dir = ''
            self.comp_vars = {}
            self.last_rows = {}
            self.last_summary = {}

            self.dir_var = tk.StringVar(value=directory or '')
            self.e0_var = tk.StringVar(value='')
            self.atom_var = tk.StringVar(value='Fe')
            self.spin_var = tk.StringVar(value='combined')
            self.reference_var = tk.StringVar(value='aligned')
            self.shift_var = tk.StringVar(value='0.0')
            self.state_var = tk.StringVar(value='both')
            self.emin_gs_var = tk.StringVar(value=f'{DEFAULT_WINDOW_GS[0]:.1f}')
            self.emax_gs_var = tk.StringVar(value=f'{DEFAULT_WINDOW_GS[1]:.1f}')
            self.emin_es_var = tk.StringVar(value=f'{DEFAULT_WINDOW_ES[0]:.1f}')
            self.emax_es_var = tk.StringVar(value=f'{DEFAULT_WINDOW_ES[1]:.1f}')
            self.fwhm_var = tk.StringVar(value='0.8')
            self.gauss_var = tk.BooleanVar(value=False)
            # 方案2: 手工输入预边峰绝对能量 → 自动换算 rel E0 并设窗口
            self.prepeak_var = tk.StringVar(value='')
            self.prepeak_hw_var = tk.StringVar(value='1.5')
            self.status_var = tk.StringVar(value='Select a PDOS folder (containing *charge*.pdos and/or *-x-*.pdos).')

            self._build_ui()
            if directory:
                self._load_directory(directory)

        # ---------------- UI ----------------
        def _build_ui(self):
            main = ttk.Frame(self.master)
            main.pack(fill=tk.BOTH, expand=True)

            left = ttk.Frame(main, width=420)
            left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
            left.pack_propagate(False)

            cv = tk.Canvas(left, highlightthickness=0, width=404)
            sb = ttk.Scrollbar(left, orient='vertical', command=cv.yview)
            sf = ttk.Frame(cv)
            win_id = cv.create_window((0, 0), window=sf, anchor='nw')
            cv.configure(yscrollcommand=sb.set)

            def _sync_scrollregion(_e=None):
                cv.configure(scrollregion=cv.bbox('all'))

            def _sync_width(e):
                # 关键: 让内层框架宽度始终跟随画布, 否则内容会把框架撑宽并被右侧裁掉
                cv.itemconfigure(win_id, width=e.width)
                _sync_scrollregion()

            sf.bind('<Configure>', _sync_scrollregion)
            cv.bind('<Configure>', _sync_width)
            cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)

            def _mw(ev):
                cv.yview_scroll(int(-ev.delta / 120), 'units')

            def _on(e):
                cv.bind_all('<MouseWheel>', _mw)

            def _off(e):
                cv.unbind_all('<MouseWheel>')

            cv.bind('<Enter>', _on)
            cv.bind('<Leave>', _off)

            right = ttk.Frame(main)
            right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

            # --- 数据源 ---
            f_data = ttk.LabelFrame(sf, text='1. Data Source', padding=6)
            f_data.pack(fill=tk.X, pady=3)
            row = ttk.Frame(f_data)
            row.pack(fill=tk.X)
            ttk.Label(row, text='Folder:').pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=self.dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            btns = ttk.Frame(f_data)
            btns.pack(fill=tk.X, pady=(3, 0))
            ttk.Button(btns, text='Browse folder...', command=self._pick_dir).pack(side=tk.LEFT)
            ttk.Button(btns, text='Load', command=self._load_directory_from_var).pack(side=tk.LEFT, padx=4)
            row2 = ttk.Frame(f_data)
            row2.pack(fill=tk.X, pady=(4, 0))
            ttk.Label(row2, text='E0 (eV):').pack(side=tk.LEFT)
            ttk.Entry(row2, textvariable=self.e0_var, width=12).pack(side=tk.LEFT, padx=3)
            ttk.Button(row2, text='Auto', width=6, command=self._auto_e0).pack(side=tk.LEFT)
            self.lbl_data = ttk.Label(f_data, text='(not loaded)', foreground='#555', wraplength=360)
            self.lbl_data.pack(anchor=tk.W, pady=(4, 0))

            # --- 选择 ---
            f_sel = ttk.LabelFrame(sf, text='2. Selection', padding=6)
            f_sel.pack(fill=tk.X, pady=3)
            r = ttk.Frame(f_sel)
            r.pack(fill=tk.X)
            ttk.Label(r, text='Atom:').pack(side=tk.LEFT)
            self.cb_atom = ttk.Combobox(r, textvariable=self.atom_var, values=['Fe', 'N', 'C'],
                                        state='readonly', width=8)
            self.cb_atom.pack(side=tk.LEFT, padx=3)
            ttk.Label(r, text='Spin:').pack(side=tk.LEFT, padx=(8, 0))
            ttk.Combobox(r, textvariable=self.spin_var, values=list(SPIN_MODES),
                         state='readonly', width=10).pack(side=tk.LEFT, padx=3)
            r2 = ttk.Frame(f_sel)
            r2.pack(fill=tk.X, pady=(4, 0))
            ttk.Label(r2, text='State:').pack(side=tk.LEFT)
            ttk.Combobox(r2, textvariable=self.state_var, values=['both', 'gs', 'es'],
                         state='readonly', width=7).pack(side=tk.LEFT, padx=3)
            ttk.Label(r2, text='Ref:').pack(side=tk.LEFT, padx=(8, 0))
            self.cb_ref = ttk.Combobox(r2, textvariable=self.reference_var,
                                       values=REFERENCE_MODES, state='readonly', width=11)
            self.cb_ref.pack(side=tk.LEFT, padx=3)
            self.cb_ref.bind('<<ComboboxSelected>>', self._on_reference_change)
            r3 = ttk.Frame(f_sel)
            r3.pack(fill=tk.X, pady=(4, 0))
            ttk.Label(r3, text='Extra shift (eV):').pack(side=tk.LEFT)
            ttk.Entry(r3, textvariable=self.shift_var, width=8).pack(side=tk.LEFT, padx=3)
            self.lbl_ref = ttk.Label(f_sel, text=REFERENCE_HINTS.get('aligned', ''),
                                     foreground='#666', wraplength=370, justify=tk.LEFT)
            self.lbl_ref.pack(anchor=tk.W, pady=(3, 0))

            # --- 能量窗口 (GS / ES 各一套) ---
            f_win = ttk.LabelFrame(sf, text='3. Energy Regions (eV, on chosen reference)', padding=6)
            f_win.pack(fill=tk.X, pady=3)

            def _window_row(parent, tag, color, emin_v, emax_v, presets, state_key):
                box = ttk.Frame(parent)
                box.pack(fill=tk.X, pady=(3, 2))
                # 标签单独一行, 数值框单独一行 —— 避免窄面板下被横向裁掉
                ttk.Label(box, text=tag, foreground=color).pack(anchor=tk.W)
                val = ttk.Frame(box)
                val.pack(fill=tk.X, pady=(2, 0))
                ttk.Label(val, text='E min').pack(side=tk.LEFT)
                ttk.Entry(val, textvariable=emin_v, width=8).pack(side=tk.LEFT, padx=(3, 6))
                ttk.Label(val, text='E max').pack(side=tk.LEFT)
                ttk.Entry(val, textvariable=emax_v, width=8).pack(side=tk.LEFT, padx=3)
                pb = ttk.Frame(box)
                pb.pack(fill=tk.X, pady=(2, 2))
                for i, (name, (lo, hi)) in enumerate(presets.items()):
                    # 2 列网格, 无论几个预设都不会横向超出面板
                    ttk.Button(pb, text=name,
                               command=lambda s=state_key, a=lo, b=hi: self._set_window(s, a, b)
                               ).grid(row=i // 2, column=i % 2, sticky='ew', padx=1, pady=1)
                pb.columnconfigure(0, weight=1)
                pb.columnconfigure(1, weight=1)
                return box

            _window_row(f_win, 'GS  (rel. Fermi):', '#0072B2',
                        self.emin_gs_var, self.emax_gs_var, GS_WINDOW_PRESETS, 'gs')
            _window_row(f_win, 'ES  (rel. E0):', '#D55E00',
                        self.emin_es_var, self.emax_es_var, ES_WINDOW_PRESETS, 'es')

            ttk.Label(f_win,
                      text='When State=both, each state uses its own row\n(GS: occupied/unoccupied; ES: pre-edge/main).',
                      foreground='#666', justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

            # --- 方案2: 手工指定预边峰 → 自动设窗口 ---
            f_pre = ttk.LabelFrame(sf, text='3b. Pre-edge Peak (manual → auto window)', padding=6)
            f_pre.pack(fill=tk.X, pady=3)
            ttk.Label(f_pre, text='Peak position (eV, absolute):',
                      foreground='#555').pack(anchor=tk.W)
            rp = ttk.Frame(f_pre)
            rp.pack(fill=tk.X, pady=(2, 0))
            ttk.Entry(rp, textvariable=self.prepeak_var, width=12).pack(side=tk.LEFT)
            ttk.Label(rp, text='±').pack(side=tk.LEFT, padx=(6, 2))
            ttk.Entry(rp, textvariable=self.prepeak_hw_var, width=6).pack(side=tk.LEFT)
            ttk.Label(rp, text='eV').pack(side=tk.LEFT, padx=(2, 0))
            rp2 = ttk.Frame(f_pre)
            rp2.pack(fill=tk.X, pady=(4, 0))
            ttk.Button(rp2, text='Apply to ES', command=self._apply_prepeak).pack(side=tk.LEFT, padx=2)
            ttk.Button(rp2, text='Apply to both', command=self._apply_prepeak_both).pack(side=tk.LEFT, padx=2)
            ttk.Button(rp2, text='Clear', width=7, command=self._clear_prepeak).pack(side=tk.LEFT, padx=2)
            self.lbl_pre = ttk.Label(f_pre, text='Read the pre-edge peak from the XAS spectrum, then enter it here.',
                                     foreground='#666', wraplength=370, justify=tk.LEFT)
            self.lbl_pre.pack(anchor=tk.W, pady=(3, 0))

            # --- 分量选择 ---
            f_comp = ttk.LabelFrame(sf, text='4. Orbital Components', padding=6)
            f_comp.pack(fill=tk.X, pady=3)
            defaults = set(DEFAULT_COMPONENTS)
            for i, (key, (label, _src)) in enumerate(COMPONENT_DEFS.items()):
                v = tk.BooleanVar(value=(key in defaults))
                self.comp_vars[key] = v
                ttk.Checkbutton(f_comp, text=label, variable=v).grid(
                    row=i // 3, column=i % 3, sticky='w', padx=1)
            for c in range(3):
                f_comp.columnconfigure(c, weight=1, minsize=1)
            bf = ttk.Frame(f_comp)
            bf.grid(row=(len(COMPONENT_DEFS) + 2) // 3, column=0, columnspan=3,
                    sticky='ew', pady=(4, 0))
            ttk.Button(bf, text='Default', command=self._select_default).pack(side=tk.LEFT, padx=2)
            ttk.Button(bf, text='Full d', command=self._select_full).pack(side=tk.LEFT, padx=2)
            ttk.Button(bf, text='All', command=self._select_all).pack(side=tk.LEFT, padx=2)

            # --- 执行 ---
            f_run = ttk.LabelFrame(sf, text='5. Compute / Export', padding=6)
            f_run.pack(fill=tk.X, pady=3)
            ttk.Button(f_run, text='Compute Centroid & Area', command=self._compute).pack(fill=tk.X, pady=2)
            ttk.Button(f_run, text='Export table (CSV)', command=self._export_table).pack(fill=tk.X, pady=2)
            ttk.Button(f_run, text='Export standard-window summary (GS+ES)',
                       command=self._export_batch).pack(fill=tk.X, pady=2)

            # --- 右侧: 图 ---
            plot_box = ttk.LabelFrame(right, text='PDOS preview (shaded = region, dashed = centroid)', padding=4)
            plot_box.pack(fill=tk.BOTH, expand=True)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.backends._backend_tk import NavigationToolbar2Tk
            self._plt = plt
            self.fig, self.ax = plt.subplots(figsize=(8, 4.2), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            NavigationToolbar2Tk(self.canvas, plot_box)
            pf = ttk.Frame(plot_box)
            pf.pack(fill=tk.X)
            ttk.Label(pf, text='Broadening FWHM (eV):').pack(side=tk.LEFT)
            ttk.Entry(pf, textvariable=self.fwhm_var, width=6).pack(side=tk.LEFT, padx=3)
            ttk.Checkbutton(pf, text='Gaussian', variable=self.gauss_var).pack(side=tk.LEFT, padx=6)

            # --- 右侧: 表 ---
            table_box = ttk.LabelFrame(right, text='Results: centroid / area / shares', padding=4)
            table_box.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
            cols = ('state', 'spin', 'component', 'centroid', 'area',
                    'share_total', 'share_pd', 'share_d', 'c_alpha', 'c_beta')
            heads = ('State', 'Spin', 'Component', 'Centroid (eV)', 'Area',
                     '% of s+p+d+f', '% of p+d', '% of d', 'Centroid a', 'Centroid b')
            widths = (52, 68, 130, 100, 84, 92, 74, 66, 88, 88)
            self.tree = ttk.Treeview(table_box, columns=cols, show='headings', height=11)
            for c, h, w in zip(cols, heads, widths):
                self.tree.heading(c, text=h)
                self.tree.column(c, width=w, anchor='center')
            vs = ttk.Scrollbar(table_box, orient='vertical', command=self.tree.yview)
            self.tree.configure(yscrollcommand=vs.set)
            self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vs.pack(side=tk.RIGHT, fill=tk.Y)

            self.txt = tk.Text(right, height=6, relief='solid', bd=1, wrap='word')
            self.txt.pack(fill=tk.X, pady=(4, 0))
            self.txt.insert(tk.END, 'Summary will appear here after Compute.')
            self.txt.config(state='disabled')

            ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN,
                      anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

            if _HAS_DND:
                try:
                    self.master.drop_target_register(DND_FILES)
                    self.master.dnd_bind('<<Drop>>', self._on_drop)
                except Exception:
                    pass

        # ---------------- 事件 ----------------
        def _on_drop(self, event):
            try:
                items = self.master.tk.splitlist(event.data)
            except Exception:
                return
            if not items:
                return
            path = items[0]
            if os.path.isfile(path):
                path = os.path.dirname(path)
            self.dir_var.set(path)
            self._load_directory(path)

        def _pick_dir(self):
            d = filedialog.askdirectory(title='Select PDOS folder')
            if d:
                self.dir_var.set(d)
                self._load_directory(d)

        def _load_directory_from_var(self):
            self._load_directory(self.dir_var.get().strip())

        def _auto_e0(self):
            e0 = load_e0_from_csv(self.dir_var.get().strip())
            if e0 is None:
                messagebox.showwarning('E0', 'No "# E0_VAL:" header found in this folder.')
                return
            self.e0_var.set(f'{e0:.4f}')
            if self.pdos_es is not None:
                self._draw()

        def _load_directory(self, directory):
            directory = (directory or '').strip()
            if not os.path.isdir(directory):
                messagebox.showerror('Error', f'Not a folder:\n{directory}')
                return
            self.dir = directory
            self.pdos_gs = load_pdos(directory, 'gs')
            self.pdos_es = load_pdos(directory, 'es')
            if self.e0 is None or not self.e0_var.get().strip():
                e0 = load_e0_from_csv(directory)
                if e0 is not None:
                    self.e0_var.set(f'{e0:.4f}')
            n_gs = len(self.pdos_gs['atoms']) if self.pdos_gs else 0
            n_es = len(self.pdos_es['atoms']) if self.pdos_es else 0
            atoms = list_canonical_atoms(self.pdos_gs, self.pdos_es)
            if atoms:
                self.cb_atom.configure(values=atoms)
                if self.atom_var.get() not in atoms:
                    self.atom_var.set('Fe' if 'Fe' in atoms else atoms[0])
            self.lbl_data.config(text=f'GS kinds: {n_gs} | ES kinds: {n_es} | E0={self.e0_var.get() or "n/a"}')
            self.fe_p_delta = None        # 数据变了, 重新计算 δ
            self.status_var.set(f'Loaded: {directory}')
            self._refresh_ref_info()
            self._draw()

        def _current_e0(self):
            """从 E0 输入框读取当前 E0 (无有效值时返回 None)。"""
            txt = self.e0_var.get().strip()
            if txt:
                try:
                    val = float(txt)
                    self.e0 = val
                    return val
                except ValueError:
                    pass
            return self.e0

        def _on_reference_change(self, _event=None):
            self.fe_p_delta = None        # 切换参考轴 → 失效缓存
            self._refresh_ref_info()
            if self.reference_var.get() == 'fe_p' and self.dir:
                self.fe_p_delta = compute_fe_p_delta(
                    self.pdos_gs, self.pdos_es, self._current_e0(),
                    float(self.fwhm_var.get() or 0.8), bool(self.gauss_var.get()))
            self._refresh_ref_info()
            self._draw()

        def _refresh_ref_info(self):
            mode = self.reference_var.get()
            txt = REFERENCE_HINTS.get(mode, '')
            if mode == 'fe_p':
                if self.fe_p_delta is not None:
                    txt += f'\nδ = {self.fe_p_delta:+.3f} eV (GS 轴平移量)'
                else:
                    txt += '\nδ = 未计算 (需 GS 未占据 Fe-p 峰 + ES Fe-p 峰)'
            try:
                self.lbl_ref.config(text=txt)
            except Exception:
                pass

        def _set_window(self, state, lo, hi):
            """把预设窗口写入对应态的那一行 (state = 'gs' / 'es')。"""
            if state == 'gs':
                self.emin_gs_var.set(f'{lo:.2f}')
                self.emax_gs_var.set(f'{hi:.2f}')
            else:
                self.emin_es_var.set(f'{lo:.2f}')
                self.emax_es_var.set(f'{hi:.2f}')
            self._draw()

        def _window_for(self, state):
            """取某一态的能量窗口 (emin, emax)。"""
            if state == 'gs':
                return self.emin_gs_var.get(), self.emax_gs_var.get()
            return self.emin_es_var.get(), self.emax_es_var.get()

        # ---------------- 方案2: 手工指定预边峰 → 自动设窗口 ----------------
        def _parse_prepeak(self):
            """读取预边峰绝对位置与半宽。返回 (peak_abs, halfwidth) 或 None。"""
            ptxt = self.prepeak_var.get().strip()
            if not ptxt:
                messagebox.showwarning('Pre-edge peak', 'Enter the pre-edge peak position (eV, absolute).')
                return None
            try:
                peak_abs = float(ptxt)
            except ValueError:
                messagebox.showerror('Pre-edge peak', 'Peak position must be a number (eV).')
                return None
            try:
                hw = float(self.prepeak_hw_var.get().strip() or 1.5)
            except ValueError:
                hw = 1.5
            if hw <= 0:
                messagebox.showerror('Pre-edge peak', 'Half-width must be positive.')
                return None
            return peak_abs, hw

        def _apply_prepeak(self):
            """把窗口设为 [峰位−hw, 峰位+hw] (ES 行, rel E0)。"""
            parsed = self._parse_prepeak()
            if parsed is None:
                return
            peak_abs, hw = parsed
            e0 = self._current_e0()
            if e0 is None:
                messagebox.showwarning('Pre-edge peak',
                                       'E0 is not set. Fill the E0 field first\n'
                                       '(the ES axis needs E0 to convert absolute → rel E0).')
                return
            rel = peak_abs - e0
            self._set_window('es', rel - hw, rel + hw)
            self.lbl_pre.config(
                text=(f'peak {peak_abs:.2f} eV (abs)  →  {rel:+.2f} eV (rel E0)\n'
                      f'ES window = [{rel - hw:+.2f}, {rel + hw:+.2f}] eV'),
                foreground='#006400')
            self.status_var.set(f'ES window set from pre-edge peak at {peak_abs:.2f} eV.')

        def _apply_prepeak_both(self):
            """把同一个窗口数字同时写入 GS 与 ES 两行。

            仅在 Fe-p→Fe-p 对齐下物理正确: 该模式下 δ 已把 GS 轴平移到与 ES 同一轴,
            同一个数字在 GS/ES 两行命中的是同一物理区域。
            """
            parsed = self._parse_prepeak()
            if parsed is None:
                return
            peak_abs, hw = parsed
            e0 = self._current_e0()
            if e0 is None:
                messagebox.showwarning('Pre-edge peak',
                                       'E0 is not set. Fill the E0 field first.')
                return
            rel = peak_abs - e0
            if self.reference_var.get() != 'fe_p':
                ok = messagebox.askyesno(
                    'Reference check',
                    'The GS row is only on the same energy axis as the ES row\n'
                    'when Ref = fe_p (Fe-p → Fe-p alignment).\n\n'
                    f'Current Ref = "{self.reference_var.get()}".\n'
                    'Applying the same window to both rows may be physically meaningless.\n\n'
                    'Continue anyway?')
                if not ok:
                    return
            self._set_window('es', rel - hw, rel + hw)
            if self.reference_var.get() == 'fe_p':
                self._set_window('gs', rel - hw, rel + hw)
            self.lbl_pre.config(
                text=(f'peak {peak_abs:.2f} eV (abs)  →  {rel:+.2f} eV (rel E0)\n'
                      f'ES window = [{rel - hw:+.2f}, {rel + hw:+.2f}] eV'
                      + ('  |  GS set to the same values (fe_p)' if self.reference_var.get() == 'fe_p'
                         else '  |  GS left unchanged (Ref is not fe_p)')),
                foreground='#006400')
            self.status_var.set(f'Window set from pre-edge peak at {peak_abs:.2f} eV.')

        def _clear_prepeak(self):
            self.prepeak_var.set('')
            self.lbl_pre.config(
                text='Read the pre-edge peak from the XAS spectrum, then enter it here.',
                foreground='#666')

        def _select_default(self):
            for k, v in self.comp_vars.items():
                v.set(k in DEFAULT_COMPONENTS)

        def _select_full(self):
            for k, v in self.comp_vars.items():
                v.set(k in FULL_COMPONENTS)

        def _select_all(self):
            for v in self.comp_vars.values():
                v.set(True)

        # ---------------- 计算 ----------------
        def _read_inputs(self):
            windows = {}
            for st, tag in (('gs', 'GS'), ('es', 'ES')):
                try:
                    a = float(self._window_for(st)[0])
                    b = float(self._window_for(st)[1])
                except ValueError:
                    messagebox.showerror('Input error', f'{tag} E min / E max must be numbers.')
                    return None
                windows[st] = (min(a, b), max(a, b))
            try:
                shift = float(self.shift_var.get() or 0.0)
            except ValueError:
                messagebox.showerror('Input error', 'Extra shift must be a number.')
                return None
            comps = [k for k, v in self.comp_vars.items() if v.get()]
            if not comps:
                messagebox.showwarning('Components', 'Select at least one orbital component.')
                return None
            e0 = self._current_e0()
            reference = self.reference_var.get()
            # Fe p→Fe p 模式: 计算 (并缓存) 需要施加到 GS 的位移 δ
            fe_p_delta = None
            if reference == 'fe_p':
                if self.fe_p_delta is None:
                    self.fe_p_delta = compute_fe_p_delta(
                        self.pdos_gs, self.pdos_es, e0,
                        float(self.fwhm_var.get() or 0.8), bool(self.gauss_var.get()))
                fe_p_delta = self.fe_p_delta
                self._refresh_ref_info()
                if fe_p_delta is None:
                    messagebox.showwarning(
                        'Fe p→Fe p',
                        'Cannot determine the Fe-p alignment shift.\n'
                        'Need GS Fe (unoccupied p peak) and ES Fex (p peak in XANES window),\n'
                        'plus a valid E0 (aligned / fe_p modes require E0 for the ES axis).')
            return dict(windows=windows, shift=shift, comps=comps, e0=e0,
                        atom=self.atom_var.get(), spin=self.spin_var.get(),
                        reference=reference, state=self.state_var.get(),
                        fe_p_delta=fe_p_delta)

        def _states_to_run(self, state):
            if state == 'both':
                out = []
                if self.pdos_gs is not None:
                    out.append(('gs', self.pdos_gs))
                if self.pdos_es is not None:
                    out.append(('es', self.pdos_es))
                return out
            if state == 'gs':
                return [('gs', self.pdos_gs)] if self.pdos_gs is not None else []
            return [('es', self.pdos_es)] if self.pdos_es is not None else []

        def _compute(self):
            cfg = self._read_inputs()
            if cfg is None:
                return
            self.tree.delete(*self.tree.get_children())
            self.last_rows = {}
            self.last_summary = {}
            lines = []
            ok = False
            for state, pdos in self._states_to_run(cfg['state']):
                lo, hi = cfg['windows'][state]        # 每个态用自己那一行窗口
                rows, summ = analyze_region(pdos, cfg['atom'], lo, hi,
                                            cfg['comps'], state=state,
                                            reference=cfg['reference'], e0=cfg['e0'],
                                            shift=cfg['shift'], spin_mode=cfg['spin'],
                                            fe_p_delta=cfg.get('fe_p_delta'))
                if not rows:
                    lines.append(f'{state.upper()}: {summ.get("error", "no data")}')
                    continue
                ok = True
                self.last_rows[state] = rows
                self.last_summary[state] = summ
                for r in rows:
                    self.tree.insert('', tk.END, values=(
                        state.upper(), cfg['spin'], r['label'],
                        self._f(r['centroid'], 3), self._f(r['area'], 4),
                        self._f(r['share_total'], 2), self._f(r['share_pd'], 2),
                        self._f(r['share_d'], 2),
                        self._f(r['centroid_alpha'], 2), self._f(r['centroid_beta'], 2)))
                lines.append(
                    f"[{state.upper()}] {summ['atom']}  window [{summ['window'][0]:+.2f}, {summ['window'][1]:+.2f}] eV"
                    f"  spin={cfg['spin']}\n"
                    f"    area(s+p+d+f)={summ['area_total_spdf']:.4f}   area(p)={summ['area_p']:.4f}"
                    f"   area(d)={summ['area_d']:.4f}\n"
                    f"    p/(p+d) = {summ['p_over_pd']:.2f}%    dominant d = {summ['dominant_d']}"
                    f"    dominant p = {summ['dominant_p']}")
            if not ok:
                messagebox.showwarning('No data', '\n'.join(lines) if lines else 'Nothing to compute.')
            self._set_text('\n'.join(lines) if lines else 'No data.')
            self._draw()
            self.status_var.set('Compute done.' if ok else 'No data computed.')

        @staticmethod
        def _f(v, nd):
            try:
                if v is None or not np.isfinite(v):
                    return '-'
                return f'{v:.{nd}f}'
            except Exception:
                return '-'

        def _set_text(self, s):
            self.txt.config(state='normal')
            self.txt.delete('1.0', tk.END)
            self.txt.insert(tk.END, s)
            self.txt.config(state='disabled')

        # ---------------- 导出 ----------------
        def _export_table(self):
            if not self.last_rows:
                messagebox.showwarning('Export', 'Run Compute first.')
                return
            fp = filedialog.asksaveasfilename(defaultextension='.csv',
                                              initialfile='region_integration.csv',
                                              filetypes=[('CSV', '*.csv')])
            if not fp:
                return
            cfg = self._read_inputs()
            n = 0
            fields = ['state', 'component', 'label', 'centroid', 'area', 'area_alpha',
                      'area_beta', 'centroid_alpha', 'centroid_beta',
                      'share_total', 'share_pd', 'share_d']
            with open(fp, 'w', newline='', encoding='utf-8') as f:
                f.write(f"# folder: {self.dir}\n")
                f.write(f"# e0: {self.e0}\n")
                f.write(f"# reference: {self.reference_var.get()}  shift: {self.shift_var.get()}\n")
                pk = self.prepeak_var.get().strip()
                if pk:
                    e0_cur = self._current_e0()
                    try:
                        pk_abs = float(pk)
                        rel_txt = (f"  ->  {pk_abs - e0_cur:+.2f} eV (rel E0)"
                                   if e0_cur is not None else '')
                    except ValueError:
                        pk_abs, rel_txt = None, ''
                    f.write(f"# pre-edge peak (manual): {pk} eV (abs)"
                            f"  half-width: {self.prepeak_hw_var.get().strip()} eV{rel_txt}\n")
                for st in ('gs', 'es'):
                    if st in self.last_summary:
                        w0, w1 = self.last_summary[st]['window']
                        f.write(f"# {st.upper()} window: [{w0:+.2f}, {w1:+.2f}] eV\n")
                f.write(f"# components: {','.join(cfg['comps']) if cfg else ''}\n")
                w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                w.writeheader()
                for state, rows in self.last_rows.items():
                    for r in rows:
                        rec = {k: r.get(k) for k in fields}
                        rec['state'] = state.upper()
                        w.writerow({k: (f'{v:.6g}' if isinstance(v, float) and np.isfinite(v) else ('' if v is None else v))
                                    for k, v in rec.items()})
                        n += 1
            self.status_var.set(f'Exported {n} rows -> {fp}')
            messagebox.showinfo('Export', f'Saved {n} rows:\n{fp}')

        def _export_batch(self):
            if not self.dir:
                messagebox.showwarning('Export', 'Load a folder first.')
                return
            comps = [k for k, v in self.comp_vars.items() if v.get()] or list(FULL_COMPONENTS)
            e0 = self.e0
            if e0 is None and self.e0_var.get().strip():
                try:
                    e0 = float(self.e0_var.get())
                except ValueError:
                    e0 = None
            recs = batch_summary(self.dir, atom=self.atom_var.get(), components=comps,
                                 spin_mode=self.spin_var.get(), e0=e0,
                                 reference=self.reference_var.get(),
                                 shift=float(self.shift_var.get() or 0.0), verbose=False,
                                 fe_p_delta=self.fe_p_delta,
                                 fwhm=float(self.fwhm_var.get() or 0.8),
                                 is_gauss=bool(self.gauss_var.get()))
            if not recs:
                messagebox.showwarning('Export', 'No records produced (check atom / files).')
                return
            fp = filedialog.asksaveasfilename(defaultextension='.csv',
                                              initialfile='region_standard_windows.csv',
                                              filetypes=[('CSV', '*.csv')])
            if not fp:
                return
            write_batch_csv(fp, recs, meta={'folder': self.dir, 'e0': e0,
                                            'reference': self.reference_var.get(),
                                            'fe_p_delta': self.fe_p_delta,
                                            'spin': self.spin_var.get()})
            self.status_var.set(f'Exported {len(recs)} window records -> {fp}')
            messagebox.showinfo('Export', f'Saved {len(recs)} records:\n{fp}')

        # ---------------- 绘图 ----------------
        def _draw(self):
            if not hasattr(self, 'ax'):
                return
            # 每个态取自己的窗口; 同时运行的态取并集作为横轴范围
            state_wins = {}
            for st in ('gs', 'es'):
                try:
                    a = float(self._window_for(st)[0])
                    b = float(self._window_for(st)[1])
                except ValueError:
                    continue
                state_wins[st] = (min(a, b), max(a, b))
            active = [st for st, _ in self._states_to_run(self.state_var.get())]
            active = [st for st in active if st in state_wins]
            if not active:
                return
            lo = min(state_wins[st][0] for st in active)
            hi = max(state_wins[st][1] for st in active)
            pad = max(4.0, 0.35 * (hi - lo) if hi > lo else 6.0)
            x_grid = np.linspace(lo - pad, hi + pad, 1200)
            fwhm = float(self.fwhm_var.get() or 0.8)
            is_gauss = bool(self.gauss_var.get())
            comps = [k for k, v in self.comp_vars.items() if v.get()] or list(DEFAULT_COMPONENTS)
            atom = self.atom_var.get()
            spin = self.spin_var.get()
            reference = self.reference_var.get()
            try:
                shift = float(self.shift_var.get() or 0.0)
            except ValueError:
                shift = 0.0
            e0 = self.e0
            if e0 is None and self.e0_var.get().strip():
                try:
                    e0 = float(self.e0_var.get())
                except ValueError:
                    e0 = None

            ax = self.ax
            ax.clear()
            colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#332288',
                      '#999933', '#44AA99', '#CC6677', '#117733', '#882255', '#AA4499']
            drawn = 0
            for state, pdos in self._states_to_run(self.state_var.get()):
                if state not in state_wins:
                    continue
                lo_s, hi_s = state_wins[state]
                real = resolve_atom(pdos, atom)
                if real is None:
                    continue
                ad = pdos['atoms'][real]
                ls = '-' if state == 'gs' else '--'
                shade = '#7fb3d5' if state == 'gs' else '#ffd77a'
                ax.axvspan(lo_s, hi_s, color=shade, alpha=0.18, zorder=0)
                for i, key in enumerate(comps):
                    y = np.zeros_like(x_grid)
                    cent = float('nan')
                    num = den = 0.0
                    for sp in (('alpha', 'beta') if spin == 'combined' else (spin,)):
                        sd = ad.get(sp)
                        if sd is None:
                            continue
                        e = energy_axis(sd.get('energy'), state, reference, e0, shift,
                                        self.fe_p_delta if reference == 'fe_p' else None)
                        if e is None:
                            continue
                        w = _component_array(sd, key)
                        if w is None:
                            continue
                        y = y + broaden_curve(e, w, x_grid, fwhm, is_gauss)
                        st = region_stats(e, w, lo_s, hi_s)
                        num += (st['centroid'] * st['area']) if np.isfinite(st['centroid']) else 0.0
                        den += st['area']
                    if den > 0:
                        cent = num / den
                    ax.plot(x_grid, y, color=colors[i % len(colors)], ls=ls, lw=1.4,
                            label=f"{state.upper()} {COMPONENT_DEFS[key][0]}")
                    if np.isfinite(cent) and den > 0:
                        ax.axvline(cent, color=colors[i % len(colors)], ls=':', lw=1.0, alpha=0.85)
                    drawn += 1
            # 窗口边界: 每个态用各自的颜色
            for st in active:
                lo_s, hi_s = state_wins[st]
                col = '#21618c' if st == 'gs' else '#b8860b'
                for x in (lo_s, hi_s):
                    ax.axvline(x, color=col, ls='-', lw=0.9, alpha=0.65)
            win_txt = '  '.join(
                f'{st.upper()}[{state_wins[st][0]:+.1f},{state_wins[st][1]:+.1f}]' for st in active)
            ax.set_xlabel('Energy (eV)  [reference: %s]' % reference)
            ax.set_ylabel('PDOS (arb.)')
            ax.set_title(f'{atom}  {win_txt}  ({self.state_var.get().upper()}, spin={spin})',
                         fontsize=10)
            ax.grid(True, alpha=0.22, ls='--')
            if drawn and drawn <= 14:
                ax.legend(fontsize=7, ncol=2, loc='upper right')
            self.fig.tight_layout()
            self.canvas.draw_idle()


# =========================================================================
# 命令行
# =========================================================================
def _parse_windows(text):
    """'name:lo:hi;name2:lo:hi2' -> OrderedDict"""
    out = OrderedDict()
    if not text:
        return out
    for part in text.split(';'):
        part = part.strip()
        if not part:
            continue
        bits = part.split(':')
        if len(bits) != 3:
            raise ValueError(f'bad window spec: {part} (expected name:emin:emax)')
        out[bits[0].strip()] = (float(bits[1]), float(bits[2]))
    return out


def cli_main(argv=None):
    ap = argparse.ArgumentParser(
        description='PDOS region integration: centroid, area and orbital-component shares.')
    ap.add_argument('--dir', required=True, help='folder containing *.pdos')
    ap.add_argument('--atom', default='Fe')
    ap.add_argument('--state', default='both', choices=['gs', 'es', 'both'])
    ap.add_argument('--spin', default='combined', choices=list(SPIN_MODES))
    ap.add_argument('--reference', default='aligned', choices=list(REFERENCE_MODES),
                    help="aligned=Fermi→E0 | fe_p=Fe-p→Fe-p (GS p peak aligned to ES p peak) | raw")
    ap.add_argument('--shift', type=float, default=0.0)
    ap.add_argument('--e0', type=float, default=None)
    ap.add_argument('--fwhm', type=float, default=0.8, help='broadening FWHM for Fe-p peak finding')
    ap.add_argument('--gaussian', action='store_true', help='use Gaussian broadening for peak finding')
    ap.add_argument('--components', default=','.join(DEFAULT_COMPONENTS))
    ap.add_argument('--window', default=None,
                    help='windows applied to BOTH states: "name:emin:emax;..."')
    ap.add_argument('--window-gs', default=None,
                    help='GS-only windows "name:emin:emax;..." (rel. Fermi)')
    ap.add_argument('--window-es', default=None,
                    help='ES-only windows "name:emin:emax;..." (rel. E0 if reference=aligned)')
    ap.add_argument('--out', default=None, help='output CSV path')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)

    comps = [c.strip() for c in args.components.split(',') if c.strip()]
    states = ('gs', 'es') if args.state == 'both' else (args.state,)
    e0 = args.e0 if args.e0 is not None else load_e0_from_csv(args.dir)

    print('=' * 78)
    print(f'PDOS Region Integration  |  dir={args.dir}')
    print(f'  atom={args.atom}  states={states}  spin={args.spin}  reference={args.reference}'
          f'  E0={e0}  shift={args.shift}')
    print(f'  components={comps}')
    print('=' * 78)

    # 窗口优先级: 按态指定 > 通用 --window > 内置标准窗口
    windows = None
    if args.window or args.window_gs or args.window_es:
        shared = _parse_windows(args.window) if args.window else None
        per_gs = _parse_windows(args.window_gs) if args.window_gs else None
        per_es = _parse_windows(args.window_es) if args.window_es else None
        windows = {}
        for st in states:
            if st == 'gs':
                windows[st] = per_gs or shared
            else:
                windows[st] = per_es or shared
        windows = {k: v for k, v in windows.items() if v}

    recs = batch_summary(args.dir, atom=args.atom, components=comps, spin_mode=args.spin,
                         e0=e0, reference=args.reference, shift=args.shift,
                         states=states, windows=windows, verbose=True,
                         fwhm=args.fwhm, is_gauss=args.gaussian)

    if args.out and recs:
        write_batch_csv(args.out, recs, meta={'folder': args.dir, 'e0': e0,
                                              'reference': args.reference, 'spin': args.spin,
                                              'components': ','.join(comps)})
        print(f'\nSaved: {args.out}  ({len(recs)} records)')
    return recs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    cli_main()
