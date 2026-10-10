"""区域积分模块的独立交叉验证脚本 (regression check).

用一套独立实现 (直接 numpy 读取原始 .pdos 文件) 复核
pdos_region_analysis 的面积 / 积分中心 / 占比计算。

运行:
    python _verify_region_analysis.py
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdos_region_analysis as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AU2EV = 27.2114
TOL = 1e-8

MAPCOL = {'d_xy': 'd-2', 'd_yz': 'd-1', 'd_z2': 'd0', 'd_xz': 'd+1', 'd_x2y2': 'd+2'}
KEYS = ['s', 'p', 'd', 'd_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2y2']


# ---------------- 独立实现 (不依赖被测模块) ----------------
def raw_cols(path):
    """返回 (E_au, fermi_au, {分量: 权重数组})"""
    with open(path) as f:
        fermi = float(f.readline().split('=')[-1].split()[0])
    d = np.loadtxt(path, skiprows=2)
    cols = {
        's': d[:, 3],
        'px': d[:, 4], 'pz': d[:, 5], 'py': d[:, 6],
        'p': d[:, 4] + d[:, 5] + d[:, 6],
        'd-2': d[:, 7], 'd-1': d[:, 8], 'd0': d[:, 9],
        'd+1': d[:, 10], 'd+2': d[:, 11],
        'd': d[:, 7:12].sum(axis=1),
        'f': d[:, 12:19].sum(axis=1) if d.shape[1] >= 19 else np.zeros(len(d)),
    }
    return d[:, 1], fermi, cols


def find_file(folder, state, spin, kind='k3'):
    pat = f'*charge*{spin}*{kind}-1.pdos' if state == 'gs' else f'*-x-*{spin}*{kind}-1.pdos'
    hits = sorted(glob.glob(os.path.join(folder, pat)))
    return hits[0] if hits else None


def raw_stats(folder, state, spin, key, lo, hi, e0, kind='k3'):
    """独立计算单自旋的 (area, centroid)。"""
    fp = find_file(folder, state, spin, kind)
    if fp is None:
        return 0.0, float('nan')
    e_au, fermi, cols = raw_cols(fp)
    e = (e_au - fermi) * AU2EV if state == 'gs' else e_au * AU2EV - e0
    w = cols[MAPCOL.get(key, key)]
    m = (e >= lo) & (e < hi)
    area = float(w[m].sum())
    cent = float((e[m] * w[m]).sum() / area) if area > 0 else float('nan')
    return area, cent


def raw_combined(folder, state, key, lo, hi, e0):
    """独立计算 alpha+beta 合并的 (area, centroid)。"""
    a_sum = c_num = 0.0
    for spin in ('ALPHA', 'BETA'):
        a, c = raw_stats(folder, state, spin, key, lo, hi, e0)
        if np.isfinite(c):
            c_num += c * a
        a_sum += a
    return a_sum, (c_num / a_sum if a_sum > 0 else float('nan'))


# ---------------- 驱动 ----------------
def main():
    folder = HERE
    e0 = R.load_e0_from_csv(folder)
    print(f'folder = {folder}')
    print(f'E0     = {e0}')
    checks, fails = 0, 0

    def cmp(tag, got, exp, tol=TOL):
        nonlocal checks, fails
        checks += 1
        if np.isnan(got) and np.isnan(exp):
            ok = True
        else:
            ok = abs(got - exp) <= tol * max(1.0, abs(exp))
        if not ok:
            fails += 1
        print(f'  {"OK  " if ok else "FAIL"} {tag:44s} got={got: .6f}  exp={exp: .6f}')
        return ok

    def assert_true(tag, cond):
        nonlocal checks, fails
        checks += 1
        if not cond:
            fails += 1
        print(f'  {"OK  " if cond else "FAIL"} {tag}')

    for state, e0_use in (('gs', None), ('es', e0)):
        pdos = R.load_pdos(folder, state)
        if pdos is None:
            print(f'[skip] no {state} files under {folder}')
            continue
        atom = R.resolve_atom(pdos, 'Fe')
        ad = pdos['atoms'][atom]

        for wname, (lo, hi) in R.standard_windows(state).items():
            print(f'\n== {state.upper()} window "{wname}" [{lo}, {hi}] ==')
            for key in KEYS:
                a_exp, c_exp = raw_combined(folder, state, key, lo, hi, e0_use)
                got = R.component_stats(ad, key, lo, hi, state=state,
                                        reference='aligned', e0=e0_use,
                                        spin_mode='combined')
                cmp(f'{key:7s} area', got['area'], a_exp)
                cmp(f'{key:7s} centroid', got['centroid'], c_exp)

            rows, summ = R.analyze_region(pdos, 'Fe', lo, hi, KEYS, state=state,
                                          reference='aligned', e0=e0_use,
                                          spin_mode='combined')
            row = {r['component']: r for r in rows}
            tot_all = summ['area_total_spdf']
            pd_ = summ['area_p'] + summ['area_d']
            for key in KEYS:
                if key not in row:
                    continue
                cmp(f'{key:7s} share_total', row[key]['share_total'],
                    100.0 * row[key]['area'] / tot_all if tot_all else float('nan'))
                cmp(f'{key:7s} share_pd', row[key]['share_pd'],
                    100.0 * row[key]['area'] / pd_ if pd_ else float('nan'))
                if key in R.D_KEYS:
                    cmp(f'{key:7s} share_d', row[key]['share_d'],
                        100.0 * row[key]['area'] / summ['area_d'] if summ['area_d'] else float('nan'))

            # d 子分量之和 == d 组总量 (求和规则自检)
            d_sub = sum(row[k]['area'] for k in R.D_KEYS if k in row)
            cmp('sum(d subcomponents) == area(d)', d_sub, summ['area_d'], 1e-10)

    # ---------------- 边界情形 ----------------
    print('\n== edge cases ==')
    pdos_gs = R.load_pdos(folder, 'gs')
    ad_gs = pdos_gs['atoms'][R.resolve_atom(pdos_gs, 'Fe')]

    st = R.component_stats(ad_gs, 'd', 500.0, 600.0, state='gs', spin_mode='combined')
    cmp('empty window area', st['area'], 0.0)
    assert_true('empty window centroid is NaN', np.isnan(st['centroid']))

    st_fwd = R.component_stats(ad_gs, 'd', 0.0, 8.2, state='gs', spin_mode='combined')
    st_rev = R.component_stats(ad_gs, 'd', 8.2, 0.0, state='gs', spin_mode='combined')
    cmp('reversed window == forward window', st_rev['area'], st_fwd['area'])
    cmp('reversed centroid == forward centroid', st_rev['centroid'], st_fwd['centroid'])

    a_only = R.component_stats(ad_gs, 'd', 0.0, 8.2, state='gs', spin_mode='alpha')
    cmp('spin=alpha area == area_alpha', a_only['area'], a_only['area_alpha'])
    b_only = R.component_stats(ad_gs, 'd', 0.0, 8.2, state='gs', spin_mode='beta')
    cmp('spin=beta  area == area_beta', b_only['area'], b_only['area_beta'])
    cmp('alpha+beta == combined area', a_only['area'] + b_only['area'], st_fwd['area'])

    pdos_es = R.load_pdos(folder, 'es')
    assert_true('resolve_atom(ES, "Fe") -> Fex', R.resolve_atom(pdos_es, 'Fe') == 'Fex')
    assert_true('resolve_atom(GS, "Fex") -> Fe', R.resolve_atom(pdos_gs, 'Fex') == 'Fe')

    rows, summ = R.analyze_region(pdos_gs, 'Fe', -13.6, 0.0, ['p', 'd'], state='gs')
    cmp('p/(p+d) consistency', summ['p_over_pd'],
        100.0 * summ['area_p'] / (summ['area_p'] + summ['area_d']))

    # 非 Fe 原子 (N) 也能算
    rows_n, summ_n = R.analyze_region(pdos_gs, 'N', -13.6, 0.0, ['s', 'p', 'px', 'py', 'pz'],
                                      state='gs')
    assert_true('N analysis returns rows', len(rows_n) == 5 and summ_n.get('area_p', 0) > 0)
    cmp('N: px+py+pz == p', sum(r['area'] for r in rows_n if r['component'] in ('px', 'py', 'pz')),
        summ_n['area_p'], 1e-10)

    print('\n' + '=' * 70)
    print(f'checks={checks}  failures={fails}')
    print('RESULT: ' + ('ALL PASS' if fails == 0 else f'{fails} FAILURES'))
    print('=' * 70)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
