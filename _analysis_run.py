"""
XAS + PDOS 联合分析脚本
用法: python _analysis_run.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pdos_tool import parse_pdos_directory, broaden_pdos as broadDOS, _ORBITAL_GROUPS

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'figure.dpi': 120})

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, '4-10-x_Lorentzian.csv')
SPECTRUM_PATH = os.path.join(DATA_DIR, '4-10-x.spectrum')
OUT_DIR = DATA_DIR

def _robust_load_csv(filepath):
    n_skip = 1
    e0 = None
    with open(filepath, 'r') as f:
        first = f.readline()
        if first.startswith('# E0_VAL:'):
            e0 = float(first.split(':')[1].strip())
            n_skip = 2
    data = np.loadtxt(filepath, delimiter=',', skiprows=n_skip)
    return data[:, 0], data[:, 1], e0

# ─── 1. load data ────────────────────────────────────────────────────────

print('=' * 60)
print('XAS + PDOS 联合分析')
print('=' * 60)

print('\n[1] 加载 XAS ...')
xas_e, xas_y, xas_e0 = _robust_load_csv(CSV_PATH)
print(f'    CSV: {os.path.basename(CSV_PATH)}')
print(f'    点数: {len(xas_e)}, 能量范围: [{xas_e[0]:.1f}, {xas_e[-1]:.1f}] eV')
print(f'    E0 (header): {xas_e0:.4f} eV')
print(f'    XAS 最大值: {np.max(xas_y):.6f} @ {xas_e[np.argmax(xas_y)]:.3f} eV')

# 从 spectrum 找主要的激发（fosc > threshold）
print('\n[2] 解析原始激发谱 ...')
spec_energies, spec_fosc = [], []
with open(SPECTRUM_PATH, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].isdigit():
            spec_energies.append(float(parts[1]))
            spec_fosc.append(float(parts[2]))
spec_energies = np.array(spec_energies)
spec_fosc = np.array(spec_fosc)
strong = spec_fosc > 1e-5
print(f'    总激发数: {len(spec_energies)}, 能量范围: [{spec_energies[0]:.1f}, {spec_energies[-1]:.1f}] eV')
print(f'    强激发 (fosc>1e-5): {strong.sum()} 条')
for idx in np.where(strong)[0]:
    print(f'      #{spec_energies[idx]:.4f} eV  fosc={spec_fosc[idx]:.6f}')

print('\n[3] 加载 PDOS ...')
pdos = parse_pdos_directory(DATA_DIR)
atoms = sorted(pdos['atoms'].keys())
spins = ['alpha', 'beta']
print(f'    原子: {", ".join(atoms)}')
for a in atoms:
    for s in spins:
        if s in pdos['atoms'][a]:
            g = list(pdos['atoms'][a][s]['groups'].keys())
            e = pdos['atoms'][a][s]['energy']
            print(f'    {a} {s}: 轨道群={g}, 点数={len(e)}, 范围=[{e[0]:.1f}, {e[-1]:.1f}] eV')

# ─── 2. compute key energies ────────────────────────────────────────────

print('\n' + '=' * 60)
print('关键能量点')
print('=' * 60)

# find max derivative for E0
deriv = np.gradient(xas_y, xas_e)
idx_e0 = np.argmax(deriv)
e0_from_deriv = xas_e[idx_e0]

# white line (global max)
idx_wl = np.argmax(xas_y)
e_wl = xas_e[idx_wl]

# pre-edge: local max before E0
mask_pre = (xas_e < e0_from_deriv - 0.5) & (xas_e > xas_e[0] + 2)
if mask_pre.sum() > 3:
    d_pre = np.diff(xas_y[mask_pre])
    peaks = []
    for i in range(1, len(d_pre)):
        if d_pre[i-1] > 0 and d_pre[i] <= 0:
            peaks.append((xas_e[mask_pre][i], xas_y[mask_pre][i]))
    if peaks:
        peaks.sort(key=lambda x: x[1], reverse=True)
        e_pre = peaks[0][0]
    else:
        e_pre = xas_e[mask_pre][-1]
else:
    e_pre = xas_e[mask_pre][-1] if mask_pre.any() else e0_from_deriv - 2

key_energies = {
    'Pre-edge': e_pre,
    'E0 (deriv max)': e0_from_deriv,
    'E0 (header)': xas_e0,
    'White line': e_wl,
}
for k, v in key_energies.items():
    val = np.interp(v, xas_e, xas_y)
    print(f'  {k:20s}  {v:.4f} eV  (XAS={val:.6f})')

# ─── 3. broadDOS per orbital ────────────────────────────────────────────

print('\n' + '=' * 60)
print('PDOS 轨道分析 (展宽 FWHM=0.8 eV)')
print('=' * 60)

FWHM = 0.8
is_gauss = False
e_min, e_max = xas_e[0], xas_e[-1]
x_grid = np.linspace(e_min, e_max, 2000)

orbital_curves = {}
for a in atoms:
    for s in spins:
        if s not in pdos['atoms'][a]:
            continue
        ad = pdos['atoms'][a][s]
        for g, members in _ORBITAL_GROUPS.items():
            if g not in ad['groups']:
                continue
            e_raw = ad['energy']
            dos_raw = ad['groups'][g]
            dos_b = broadDOS(e_raw, dos_raw, x_grid, FWHM, is_gauss)
            key = f'{a}-{g}-{s}'
            orbital_curves[key] = dos_b

# 自旋求和: 输出 atom-group
summed = {}
for a in atoms:
    for g in ['s', 'p', 'd', 'f']:
        key = f'{a}-{g}'
        a_key = f'{a}-{g}-alpha'
        b_key = f'{a}-{g}-beta'
        d_a = orbital_curves.get(a_key, np.zeros_like(x_grid))
        d_b = orbital_curves.get(b_key, np.zeros_like(x_grid))
        summed[key] = d_a + d_b

# 打印各轨道峰值
print(f'\n{"Atom":>5s} {"Orb":>4s} {"Peak E":>10s} {"Peak DOS":>10s} {"Integ DOS":>10s} {"% Total":>8s}')
all_integ = {k: np.trapz(v, x_grid) for k, v in summed.items()}
total_integ = sum(all_integ.values())
for a in atoms:
    for g in ['s', 'p', 'd', 'f']:
        key = f'{a}-{g}'
        if key not in summed:
            continue
        d = summed[key]
        if np.max(d) < 1e-10:
            continue
        peak_e = x_grid[np.argmax(d)]
        peak_dos = np.max(d)
        integ = all_integ[key]
        pct = integ / total_integ * 100 if total_integ > 0 else 0
        print(f'{a:>5s} {g:>4s}  {peak_e:8.3f}   {peak_dos:8.5f}   {integ:8.3f}   {pct:7.2f}%')

# ─── 4. contributions at key energies ────────────────────────────────────

print('\n' + '=' * 60)
print('关键能量点的 PDOS 贡献')
print('=' * 60)

contrib_header = f'{"Energy":>10s} {"XAS":>10s}'
for a in atoms:
    for g in ['s', 'p', 'd', 'f']:
        k = f'{a}-{g}'
        if k in summed:
            contrib_header += f' {k:>10s}'
print(contrib_header)
for label, e_k in key_energies.items():
    xas_val = np.interp(e_k, xas_e, xas_y)
    line = f'{label:>10s} {xas_val:10.6f}'
    for a in atoms:
        for g in ['s', 'p', 'd', 'f']:
            k = f'{a}-{g}'
            if k in summed:
                val = np.interp(e_k, x_grid, summed[k])
                line += f' {val:10.6f}'
    print(line)

# ─── 5. TME analysis ────────────────────────────────────────────────────

print('\n' + '=' * 60)
print('TME (Transition Matrix Element) 分析')
print('=' * 60)

# Sum all p-orbital PDOS (K-edge selection rule)
pdos_p_sum = np.zeros_like(x_grid)
for a in atoms:
    pk = f'{a}-p'
    if pk in summed:
        pdos_p_sum += summed[pk]

pdos_safe = np.clip(pdos_p_sum, 1e-10, None)
tme = xas_y / np.interp(xas_e, x_grid, pdos_p_sum)
tme = np.clip(tme, 0, np.percentile(tme[tme > 0], 98) if tme[tme > 0].size > 0 else 1)

print(f'TME 范围: [{np.min(tme):.4f}, {np.max(tme):.4f}]')
print(f'\n{"Energy":>8s} {"XAS":>10s} {"Σp-PDOS":>10s} {"TME":>10s}')
for e_pt in [e_pre, e0_from_deriv, e_wl, e_wl + 2, e_wl + 5]:
    if e_pt < xas_e[0] or e_pt > xas_e[-1]:
        continue
    xv = np.interp(e_pt, xas_e, xas_y)
    pv = np.interp(e_pt, x_grid, pdos_p_sum)
    tv = xv / pv if pv > 1e-10 else 0
    print(f'{e_pt:8.3f} {xv:10.6f} {pv:10.6f} {tv:10.4f}')

# ─── 6. Plotting ─────────────────────────────────────────────────────────

print('\n' + '=' * 60)
print('生成图表 ...')
print('=' * 60)

# colorblind palette
COLORS = {'C': '#56B4E9', 'N': '#009E73', 'Fe': '#0072B2'}
HYB_COLORS = {'C_p': '#56B4E9', 'N_p': '#009E73', 'Fex_p': '#0072B2',
              'C_d': '#6699CC', 'N_d': '#44AA66', 'Fex_d': '#CC79A7',
              'C_s': '#AADDFF', 'N_s': '#88DDAA', 'Fex_s': '#88BBEE',
              'Fex_f': '#AA3377'}
LINESTYLES = {'s': ':', 'p': '-', 'd': '--', 'f': '-.'}

# normalize XAS to [0, 1]
xas_norm = xas_y / np.max(xas_y)

fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1],
              hspace=0.35, wspace=0.30)

# ── Ax1: XAS + PDOS comparison ──
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(xas_e, xas_norm, 'k-', lw=2.0, label='XAS (norm)')
ax1.axvline(xas_e0, color='gray', ls=':', lw=1.0, alpha=0.7, label=f'E0={xas_e0:.2f}')
ax1_pdos = ax1.twinx()
for a in atoms:
    for g in ['s', 'p', 'd', 'f']:
        k = f'{a}-{g}'
        if k not in summed:
            continue
        d = summed[k]
        if np.max(d) < 1e-10:
            continue
        dn = d / np.max(d) if np.max(d) > 0 else d
        c = COLORS.get(a, '#7f7f7f')
        ls = LINESTYLES.get(g, '-')
        lw = 2.0 if g in ('p', 'd') else 1.0
        alpha_v = 0.8 if g in ('p', 'd') else 0.4
        ax1_pdos.plot(x_grid, dn, color=c, ls=ls, lw=lw, alpha=alpha_v, label=f'{k}')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1_pdos.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=7, ncol=3)
ax1.set_xlim(xas_e[0], xas_e[-1])
ax1.set_title(f'XAS + PDOS Comparison (E0={xas_e0:.2f} eV, FWHM={FWHM:.1f} eV)')
ax1.set_xlabel('Energy (eV)'); ax1.set_ylabel('XAS (norm)')
ax1_pdos.set_ylabel('PDOS (norm)')
ax1.grid(True, alpha=0.2, ls='--')

# ── Ax2: TME ──
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(xas_e, tme, 'r-', lw=1.5)
ax2.fill_between(xas_e, 0, tme, color='red', alpha=0.1)
ax2.axvline(xas_e0, color='gray', ls=':', lw=1, alpha=0.5)
ax2.set_title('Transition Matrix Element (K-edge → np)')
ax2.set_xlabel('Energy (eV)'); ax2.set_ylabel('TME (a.u.)')
ax2.grid(True, alpha=0.2, ls='--')
ax2.set_xlim(xas_e[0], xas_e[-1])

# ── Ax3: p-orbital comparison ──
ax3 = fig.add_subplot(gs[1, 1])
for a in atoms:
    pk = f'{a}-p'
    if pk not in summed:
        continue
    d = summed[pk]
    dn = d / np.max(d) if np.max(d) > 0 else d
    c = COLORS.get(a, '#7f7f7f')
    ax3.plot(x_grid, dn, color=c, lw=1.5, ls='-', label=f'{pk}')
ax3.axvline(xas_e0, color='gray', ls=':', lw=1, alpha=0.5)
ax3.set_title('p-orbital PDOS Comparison')
ax3.set_xlabel('Energy (eV)'); ax3.set_ylabel('PDOS (norm)')
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.2, ls='--')
ax3.set_xlim(xas_e[0], xas_e[-1])

# ── Ax4: d-orbital comparison ──
ax4 = fig.add_subplot(gs[2, 0])
for a in atoms:
    dk = f'{a}-d'
    if dk not in summed:
        continue
    d = summed[dk]
    dn = d / np.max(d) if np.max(d) > 0 else d
    c = COLORS.get(a, '#7f7f7f')
    ax4.plot(x_grid, dn, color=c, lw=1.5, ls='--', label=f'{dk}')
ax4.axvline(xas_e0, color='gray', ls=':', lw=1, alpha=0.5)
ax4.set_title('d-orbital PDOS Comparison')
ax4.set_xlabel('Energy (eV)'); ax4.set_ylabel('PDOS (norm)')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.2, ls='--')
ax4.set_xlim(xas_e[0], xas_e[-1])

# ── Ax5: Orbital hybridization for Fe ──
ax5 = fig.add_subplot(gs[2, 1])
if 'Fex' in atoms:
    for g in ['s', 'p', 'd', 'f']:
        k = f'Fex-{g}'
        if k not in summed:
            continue
        d = summed[k]
        dn = d / np.max(d) if np.max(d) > 0 else d
        c = HYB_COLORS.get(k, '#7f7f7f')
        ls = LINESTYLES.get(g, '-')
        ax5.plot(x_grid, dn, color=c, ls=ls, lw=1.5, label=k)
        ax5.fill_between(x_grid, 0, dn, color=c, alpha=0.07)
ax5.axvline(xas_e0, color='gray', ls=':', lw=1, alpha=0.5)
ax5.set_title('Fex Orbital Hybridization')
ax5.set_xlabel('Energy (eV)'); ax5.set_ylabel('PDOS (norm)')
ax5.legend(loc='upper left', fontsize=8)
ax5.grid(True, alpha=0.2, ls='--')
ax5.set_xlim(xas_e[0], xas_e[-1])

fig.suptitle(f'XAS-PDOS Combined Analysis — E0={xas_e0:.2f} eV', fontsize=14, y=1.01)
png_path = os.path.join(OUT_DIR, '_analysis_xas_pdos.png')
fig.savefig(png_path, dpi=150, bbox_inches='tight')
print(f'  保存: {png_path}')
plt.close()

# ── Individual TME plot ──
fig2, ax2b = plt.subplots(figsize=(8, 5))
ax2b.plot(xas_e, tme, 'r-', lw=2)
ax2b.fill_between(xas_e, 0, tme, color='red', alpha=0.15)
ax2b.axvline(xas_e0, color='gray', ls=':', lw=1.5, alpha=0.6, label=f'E0={xas_e0:.2f}')
ax2b.set_title('TME = XAS / Sum(p-PDOS)')
ax2b.set_xlabel('Energy (eV)')
ax2b.set_ylabel('TME (a.u.)')
ax2b.grid(True, alpha=0.2, ls='--')
ax2b.legend()
fn2 = os.path.join(OUT_DIR, '_analysis_tme.png')
fig2.savefig(fn2, dpi=150, bbox_inches='tight')
print(f'  保存: {fn2}')
plt.close()

# ── Individual d-orbital detail ──
fig3, ax3b = plt.subplots(figsize=(8, 5))
for a in atoms:
    dk = f'{a}-d'
    if dk not in summed:
        continue
    d = summed[dk]
    dn = d / np.max(d) if np.max(d) > 0 else d
    c = COLORS.get(a, '#7f7f7f')
    ax3b.plot(x_grid, dn, color=c, lw=2, ls='--', label=dk)
    ax3b.fill_between(x_grid, 0, dn, color=c, alpha=0.08)
ax3b.axvline(xas_e0, color='gray', ls=':', lw=1.5, alpha=0.6, label=f'E0={xas_e0:.2f}')
ax3b.set_title('d-orbital PDOS (pre-edge region)')
ax3b.set_xlabel('Energy (eV)')
ax3b.set_ylabel('PDOS (norm)')
ax3b.legend()
ax3b.grid(True, alpha=0.2, ls='--')
fn3 = os.path.join(OUT_DIR, '_analysis_d_orbital.png')
fig3.savefig(fn3, dpi=150, bbox_inches='tight')
print(f'  保存: {fn3}')
plt.close()

print(f'\n完成! 共输出 {3} 张图表到: {OUT_DIR}')
