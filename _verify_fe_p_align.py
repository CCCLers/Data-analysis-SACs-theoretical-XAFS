"""交叉验证: 原脚本 pdos_gs_es_compare 的 Fe_p→Fe_p 对齐 vs 新脚本的 fe_p 模式.

原脚本逻辑 (pdos_gs_es_compare.py):
  _auto_align (Fe_p_Fe_p):
      gs_p     = _find_peak('gs', 'Fe', 'p', unocc_only=True)          # 相对费米
      es_p_abs = _find_peak('es', 'Fex', 'p', e_range=XANES窗口)        # 绝对激发能
      gs_base_shift = es_p_abs - gs_p
  _update_plot:
      gs_shift = gs_base_shift + energy_shift_gs(0) = es_p_abs - gs_p
      GS: e_plot = (e_r + gs_shift) - xas_e0 = e_r + (es_p_abs - gs_p) - xas_e0
      ES: e_plot = (e_r + 0) - xas_e0 = e_r - xas_e0

新脚本:
  δ = ES_p_rel - GS_p = (es_p_abs - e0) - gs_p
  GS: energy_axis(e_r, 'gs', 'fe_p', e0, 0, δ) = e_r + δ
  ES: energy_axis(e_r, 'es', 'fe_p', e0, 0, δ) = e_r - e0

→ GS 应逐元素一致; ES 应逐元素一致。
"""
import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pdos_tool as es_mod
import pdos_region_analysis as R

folder = HERE
gs_data = es_mod.parse_pdos_file_list(sorted(glob.glob(os.path.join(folder, '*charge*.pdos'))))
es_data = es_mod.parse_pdos_file_list(sorted(glob.glob(os.path.join(folder, '*-x-*.pdos'))))
e0 = R.load_e0_from_csv(folder)
fwhm = 0.8
is_gauss = False

print(f'folder = {folder}')
print(f'E0     = {e0:.4f} eV')

# ---- 原脚本的 _find_peak 等价实现 (用 pdos_tool.broaden_pdos) ----
def orig_find_peak(data, atom, group, unocc_only=False, e_range=None, spin='alpha'):
    ad = data['atoms'][atom]
    e = ad[spin]['energy'].copy()
    w = ad[spin]['groups'].get(group, np.zeros_like(e)).copy()
    if unocc_only:
        m = ad[spin]['occupation'] < 0.5
        e, w = e[m], w[m]
    if e_range is not None:
        m = (e >= e_range[0]) & (e <= e_range[1])
        if m.sum() < 3:
            return None
        e, w = e[m], w[m]
    if len(e) < 3:
        return None
    pad = (e.max() - e.min()) * 0.1 or 2.0
    xg = np.linspace(e.min() - pad, e.max() + pad, 2000)
    dos = es_mod.broaden_pdos(e, w, xg, fwhm, is_gauss)
    return float(xg[np.argmax(dos)])


# ---- 原脚本 Fe_p_Fe_p ----
xas_lo, xas_hi = e0 - 25.0, e0 + 25.0   # 无 XAS csv 时的默认窗口
gs_p_orig = orig_find_peak(gs_data, 'Fe', 'p', unocc_only=True)
es_p_abs_orig = orig_find_peak(es_data, 'Fex', 'p', e_range=(xas_lo, xas_hi))
gs_base_shift = es_p_abs_orig - gs_p_orig
gs_shift_orig = gs_base_shift + 0.0

# ---- 新脚本 fe_p ----
delta_new = R.compute_fe_p_delta(gs_data, es_data, e0, fwhm, is_gauss)
gs_p_new = R.find_pdos_peak(gs_data, 'Fe', 'p', unocc_only=True, fwhm=fwhm, is_gauss=is_gauss)
es_p_abs_new = R.find_pdos_peak(es_data, 'Fex', 'p', unocc_only=False,
                                e_range=(xas_lo, xas_hi), fwhm=fwhm, is_gauss=is_gauss)

print()
print('--- Fe-p 峰位 ---')
print(f'  GS Fe-p 峰 (相对费米)   : 原={gs_p_orig:+.4f}  新={gs_p_new:+.4f}')
print(f'  ES Fe-p 峰 (绝对 eV)    : 原={es_p_abs_orig:.4f}  新={es_p_abs_new:.4f}')
print(f'  ES Fe-p 峰 (相对 E0)    : {es_p_abs_orig - e0:+.4f}')
print(f'  δ (应加到 GS 的位移)    : 原={gs_shift_orig - e0:+.4f}  新={delta_new:+.4f}')

checks = 0
fails = 0

def cmpv(tag, a, b, tol=1e-6):
    global checks, fails
    checks += 1
    ok = (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= tol)
    if not ok:
        fails += 1
    print(f'  {"OK  " if ok else "FAIL"} {tag:34s} 原={a}  新={b}')

cmpv('GS Fe-p 峰', gs_p_orig, gs_p_new)
cmpv('ES Fe-p 峰', es_p_abs_orig, es_p_abs_new)
cmpv('δ 位移', gs_shift_orig - e0, delta_new)

# ---- 逐元素比对两套能量轴 ----
def cmp_arr(tag, a, b):
    global checks, fails
    checks += 1
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        print(f'  FAIL {tag}: shape {a.shape} vs {b.shape}')
        fails += 1
        return
    ok = np.allclose(a, b, atol=1e-9)
    if not ok:
        print(f'  FAIL {tag}: maxdiff={np.max(np.abs(a - b)):.3e}')
        fails += 1
    else:
        print(f'  OK   {tag}: identical ({len(a)} elements)')


print()
print('--- 能量轴逐元素比对 ---')
for spin in ('alpha', 'beta'):
    # GS: 原 = (e_r + gs_shift) - e0 ; 新 = e_r + δ
    e_r_gs = gs_data['atoms']['Fe'][spin]['energy']
    orig_gs = (e_r_gs + gs_shift_orig) - e0
    mine_gs = R.energy_axis(e_r_gs, 'gs', 'fe_p', e0, 0.0, delta_new)
    cmp_arr(f'GS Fe {spin}', orig_gs, mine_gs)

    # ES: 原 = (e_r + 0) - e0 ; 新 = e_r - e0
    e_r_es = es_data['atoms']['Fex'][spin]['energy']
    orig_es = (e_r_es + 0.0) - e0
    mine_es = R.energy_axis(e_r_es, 'es', 'fe_p', e0, 0.0, delta_new)
    cmp_arr(f'ES Fex {spin}', orig_es, mine_es)

# ---- 关键物理检验: GS Fe-p 峰与 ES Fe-p 峰在 fe_p 轴上应重合 ----
print()
print('--- 对齐效果检验 (GS p 峰与 ES p 峰是否重合) ---')
gs_peak_on_axis = R.energy_axis(np.array([gs_p_new]), 'gs', 'fe_p', e0, 0.0, delta_new)[0]
es_peak_on_axis = R.energy_axis(np.array([es_p_abs_new]), 'es', 'fe_p', e0, 0.0, delta_new)[0]
print(f'  GS p 峰对齐后位置 : {gs_peak_on_axis:+.4f} eV (rel E0)')
print(f'  ES p 峰位置       : {es_peak_on_axis:+.4f} eV (rel E0)')
checks += 1
ok = abs(gs_peak_on_axis - es_peak_on_axis) < 1e-6
if not ok:
    fails += 1
print(f'  {"OK  " if ok else "FAIL"} 两峰重合 (差值={gs_peak_on_axis - es_peak_on_axis:.2e} eV)')

# ---- 与 aligned 模式对比: 说明两者差异 ----
print()
print('--- aligned vs fe_p 的 GS 轴差异 ---')
gs_aligned = R.energy_axis(np.array([gs_p_new]), 'gs', 'aligned', e0, 0.0, None)[0]
print(f'  GS p 峰在 aligned 轴上 : {gs_aligned:+.4f} eV (rel Fermi)')
print(f'  GS p 峰在 fe_p 轴上    : {gs_peak_on_axis:+.4f} eV (rel E0)')
print(f'  两模式 GS 轴相差       : {gs_peak_on_axis - gs_aligned:+.4f} eV')
print('  → 说明: 两种模式下同一组窗口数字会选中不同的 MO 集合')

print()
print('=' * 62)
print(f'checks={checks}  failures={fails}')
print('RESULT: ' + ('ALL PASS — 新脚本 fe_p 与原脚本 Fe_p_Fe_p 完全一致'
                    if fails == 0 else f'{fails} FAILURES'))
print('=' * 62)
