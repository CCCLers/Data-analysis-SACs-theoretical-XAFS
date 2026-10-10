import logging
import numpy as np
import matplotlib.pyplot as plt
import math

logger = logging.getLogger(__name__)

def calculate_y(energy, fdip, x, regions, is_gaussian):
    """
    基础计算函数：将离散的激发态(fdip)展宽为连续光谱(y)
    """
    y = np.zeros_like(x)
    
    # 预计算每个能量点对应的 FWHM，避免内层循环过大
    # 对于简单的两段式（Auto-split），这可以优化，但为了通用性保持区域判定
    
    # 简单的区域查找优化
    if not regions:
        return y

    # 为了性能，直接在循环中处理
    for i in range(len(energy)):
        e_i = energy[i]
        val_i = fdip[i]
        
        # 确定当前能量点的 FWHM
        current_fwhm = 0.5 # Default
        for start, end, val in regions:
            if start <= e_i < end:
                current_fwhm = val
                break
        
        if is_gaussian:
            sigma = current_fwhm / (2 * np.sqrt(2 * np.log(2)))
            prefactor = 1 / (sigma * np.sqrt(2 * np.pi))
            term = np.exp(-((x - e_i)**2) / (2 * sigma**2))
            y += val_i * prefactor * term
        else:
            gamma = current_fwhm / 2
            denom = (x - e_i)**2 + gamma**2
            y += val_i * (1 / math.pi) * (gamma / denom)
    return y

def get_e0_candidates(x, y):
    """
    计算导数并返回三种标准的建议 E0 值。
    返回格式: {'std1': (value, desc), 'std2': ..., 'std3': ...}
    """
    dydx = np.gradient(y, x)
    candidates = {}
    
    # 寻找导数峰值
    deriv_peaks_indices = []
    max_deriv_val = np.max(dydx)
    for i in range(1, len(dydx)-1): 
        if dydx[i] > dydx[i-1] and dydx[i] > dydx[i+1]:
            if dydx[i] > max_deriv_val * 0.05: # 忽略噪音
                deriv_peaks_indices.append(i)
    
    # --- Std 2: Max Derivative (最大斜率) ---
    idx_max_deriv = np.argmax(dydx)
    candidates['std2'] = (x[idx_max_deriv], "Std.2: Max Derivative")

    # --- Std 1: 1st Inflection (Pre-edge) ---
    # 通常是第一个显著的导数峰
    if deriv_peaks_indices:
        idx_std1 = deriv_peaks_indices[0]
        candidates['std1'] = (x[idx_std1], "Std.1: 1st Inflection (Pre-edge)")
    else:
        candidates['std1'] = candidates['std2'] # Fallback

    # --- Std 3: White Line Onset (Physics) ---
    # 定义为：强度最大值（White Line）之前的最后一个导数峰
    max_intensity_idx = np.argmax(y)
    # 筛选出位于最大吸收峰左侧的导数峰
    valid_peaks = [i for i in deriv_peaks_indices if i < max_intensity_idx]
    
    if valid_peaks:
        # 取最靠近主峰的那个导数峰
        best_idx = valid_peaks[-1]
        candidates['std3'] = (x[best_idx], "Std.3: White Line Edge")
    else:
        # 如果找不到，回退到 Std 2 或者直接用主峰位置(虽然物理上不对)
        candidates['std3'] = (x[idx_max_deriv], "Std.3 (Fallback to Std.2)")
        
    return candidates

def parse_spectrum_data(spectrum_file, config_dict):
    """
    只负责读取文件并过滤数据，返回原始的 stick spectrum 数据。
    """
    try:
        atom_kind = config_dict.get("atom_kind", "Fe").strip()
        donor_type = config_dict.get("donor_type", "").strip()
        excitation_type = config_dict.get("excitation_type", "").strip()
    except Exception:
        return None

    energy = []
    fdip = []
    
    check_donor = len(donor_type) > 0
    check_excit = len(excitation_type) > 0
    check_atom = len(atom_kind) > 0
    do_type, ex_type, at_kind = not check_donor, not check_excit, not check_atom

    try:
        with open(spectrum_file, "r") as myfile:
            for line in myfile:
                line_stripped = line.strip()
                if not line_stripped: continue
                if line.startswith("XAS TDP"):
                    do_type = donor_type in line if check_donor else True
                    ex_type = excitation_type in line if check_excit else True
                if line.startswith("from EXCITED ATOM"):
                    at_kind = atom_kind in line if check_atom else True
                
                parts = line.split()
                if do_type and ex_type and at_kind and parts[0].isdigit():
                    try:
                        energy.append(float(parts[1]))
                        fdip.append(float(parts[2]))
                    except ValueError: pass
    except Exception as e:
        logger.error(f"Read error: {e}")
        return None

    if not energy:
        return None

    return {
        'energy': np.array(energy),
        'fdip': np.array(fdip),
        'atom_kind': atom_kind,
        'donor_type': donor_type,
        'excitation_type': excitation_type
    }

def compute_spectrum(raw_data, config_dict, broadening_method="Lorentzian"):
    """
    纯计算函数：返回展宽后的光谱数据，不涉及任何绘图。
    返回 {x, y, dydx, d2ydx2, energy, fdip, e0, title} 或 None。
    """
    energy = raw_data['energy']
    fdip = raw_data['fdip']
    atom_kind = raw_data.get('atom_kind', '')

    try:
        emin = float(config_dict.get("emin", 700))
        emax = float(config_dict.get("emax", 750))
        npts = int(config_dict.get("npts", 3000))
        auto_split = config_dict.get("auto_split", False)
        calculated_e0 = config_dict.get("calculated_e0", None)

        target_key = "broadening_regions_gauss" if "gauss" in broadening_method.lower() else "broadening_regions_lorentz"
        regions_str = config_dict.get(target_key, "0.4,1.0")
    except ValueError:
        return None

    x = np.linspace(emin, emax, npts)
    is_gaussian = "gauss" in broadening_method.lower()
    final_regions = []

    if auto_split:
        try:
            parts = regions_str.replace(";", ",").split(",")
            width_pre = float(parts[0])
            width_post = float(parts[1]) if len(parts) > 1 else width_pre
        except:
            width_pre, width_post = 0.5, 1.0

        if calculated_e0 is None:
            split_standard = config_dict.get("split_strategy", "std3")
            search_fwhm = (width_pre + width_post) / 2
            temp_regions = [(emin, emax, search_fwhm)]
            y_temp = calculate_y(energy, fdip, x, temp_regions, is_gaussian)
            candidates = get_e0_candidates(x, y_temp)
            calculated_e0 = candidates.get(split_standard, candidates['std3'])[0]

        final_regions = [
            (emin, calculated_e0, width_pre),
            (calculated_e0, emax, width_post)
        ]
    else:
        for region in regions_str.split(";"):
            parts = region.split(",")
            if len(parts) != 3:
                continue
            try:
                start_val = parts[0].strip().lower()
                end_val = parts[1].strip().lower()
                start = emin if start_val == "emin" else float(start_val)
                end = emax if end_val == "emax" else float(end_val)
                fwhm = float(parts[2])
                final_regions.append((start, end, fwhm))
            except:
                pass

        if calculated_e0 is None:
            temp_regions = [(emin, emax, 1.0)]
            y_temp = calculate_y(energy, fdip, x, temp_regions, is_gaussian)
            cands = get_e0_candidates(x, y_temp)
            calculated_e0 = cands['std3'][0]

    y = calculate_y(energy, fdip, x, final_regions, is_gaussian)
    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)
    title = f"{atom_kind} ({broadening_method})"

    return {
        'x': x, 'y': y, 'dydx': dydx, 'd2ydx2': d2ydx2,
        'energy': energy, 'fdip': fdip,
        'e0': calculated_e0, 'title': title
    }

def process_spectrum(spectrum_file_or_data, config_dict, output_dir=None, broadening_method="Lorentzian"):
    """
    主处理函数（保持向后兼容）。包装 compute_spectrum + 绘图。
    """
    if isinstance(spectrum_file_or_data, str):
        raw_data = parse_spectrum_data(spectrum_file_or_data, config_dict)
    elif isinstance(spectrum_file_or_data, dict):
        raw_data = spectrum_file_or_data
    else:
        return None

    if not raw_data:
        return None

    data = compute_spectrum(raw_data, config_dict, broadening_method)
    if data is None:
        return None

    x, y = data['x'], data['y']
    energy, fdip = data['energy'], data['fdip']
    calculated_e0 = data['e0']
    atom_kind = raw_data.get('atom_kind', '')

    emin, emax = float(config_dict.get("emin", 700)), float(config_dict.get("emax", 750))

    fig, ax = plt.subplots(figsize=(10, 6))
    lines = ax.plot(x, y, label="Spectrum", linewidth=2.5, color='#1f77b4')

    max_y = np.max(y) if np.max(y) > 0 else 1
    max_fdip = np.max(fdip) if np.max(fdip) > 0 else 1
    scale_factor = (max_y / max_fdip) * 0.3

    mask = (energy >= emin) & (energy <= emax)
    ax.vlines(energy[mask], 0, fdip[mask] * scale_factor, color='gray', alpha=0.3, linewidth=1)

    if calculated_e0:
        ax.axvline(x=calculated_e0, color='#d62728', linestyle='--', linewidth=1.5, label=f"E0={calculated_e0:.2f}")

    ax.set_title(data['title'], fontsize=14)
    ax.set_xlabel("Energy (eV)", fontsize=12)
    ax.set_ylabel("Intensity (arb. units)", fontsize=12)
    ax.legend(loc='upper left')
    ax.set_ylim(0, max_y * 1.15)
    fig.tight_layout()

    data.update({'fig': fig, 'ax': ax, 'lines': lines})
    return data