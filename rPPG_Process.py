# -*- coding: utf-8 -*-
"""
文件描述:
本模块是rPPG信号分析的核心，负责从预处理后的rPPG信号中提取各种生理指标。
主要功能由 `process` 函数驱动，它整合了峰值检测、RR间期计算、
异常搏动检测与修正、时域HRV（心率变异性）指标计算、
庞加莱图（Poincaré Plot）分析、频域分析以及呼吸率估算等一系列复杂步骤。
此外，还加入了更高级的分析，如心律失常状态、血管健康状态和压力水平的初步评估。

原作者: Yongzhi
注释翻译与增强: Gemini
"""

import numpy as np
import filtering # 导入本地的滤波模块
from scipy.signal import welch, periodogram, find_peaks # 从SciPy导入Welch方法（用于功率谱密度估计）、周期图和峰值查找函数
from scipy.interpolate import UnivariateSpline # 从SciPy导入样条插值函数
import sys
import time


def process(hrdata, sample_rate, windowsize=0.75,
            freq_method='welch', welch_wsize=240, bpmmin=40, bpmmax=180,
            reject_segmentwise=False, measures=None, working_data=None):
    '''
    功能: 核心处理函数，分析给定的rPPG信号段，计算并返回详细的生理指标。
    '''

    # --- 1. 初始化 ---
    # 如果未提供字典，则创建新的空字典
    if measures is None:
        measures = {}
    if working_data is None:
        working_data = {}

    # 确保输入数据是一维的
    assert np.asarray(hrdata).ndim == 1, '错误：传入的心率数据应为一维数组或列表'

    # --- 2. 信号预处理 ---
    # 检查并确保信号基线为正，某些后续算法（如峰值检测）可能需要正信号
    bl_val = np.percentile(hrdata, 0.1)
    if bl_val < 0:
        hrdata += abs(bl_val)

    # 将原始信号和采样率存入工作字典
    working_data['hr'] = hrdata
    working_data['sample_rate'] = sample_rate

    # --- 3. 峰值检测 ---
    # 计算滚动平均值，作为动态阈值的一部分
    rol_mean = rolling_mean(hrdata, windowsize, sample_rate)
    try:
        # 拟合最佳的峰值检测参数并找出所有峰值
        working_data = fit_peaks(hrdata, rol_mean, sample_rate, bpmmin=bpmmin, bpmmax=bpmmax, working_data=working_data)
    except Exception as e:
        # 如果峰值检测失败（例如，信号质量太差），打印错误并返回空结果
        # If peak fitting fails, return empty data
        print(f"Peak fitting failed: {e}")
        return working_data, measures


    # --- 4. RR间期计算与修正 ---
    # 根据峰值列表计算RR间期（单位：毫秒）
    working_data = calc_rr(working_data['peaklist'], sample_rate, working_data=working_data)
    # 检查并剔除异常的RR间期（过长或过短）
    working_data = check_peaks(working_data['RR_list'], working_data['peaklist'], working_data['ybeat'],
                               reject_segmentwise, working_data=working_data)

    # --- 5. 计算各项生理指标 ---
    # 计算时域HRV指标 (BPM, IBI, SDNN, RMSSD, pNN50等)
    working_data, measures = calc_ts_measures(working_data['RR_list_cor'], working_data['RR_diff'],
                                              working_data['RR_sqdiff'], measures=measures, working_data=working_data)

    # 计算庞加莱图相关指标 (SD1, SD2)
    measures = calc_poincare(working_data['RR_list'], working_data['RR_masklist'], measures=measures,
                             working_data=working_data)

    # 计算高级指标 (心律、血管、压力状态)
    # Add advanced analysis for arrhythmia, pulse shape, and stress
    measures, working_data = analyze_advanced_metrics(hrdata, sample_rate, working_data, measures)

    # 估算呼吸率
    try:
        measures, working_data = calc_breathing(working_data['RR_list_cor'], method=freq_method,
                                                measures=measures, working_data=working_data)
    except Exception as e:
        # 如果呼吸率计算出错，则设为NaN
        measures['breathingrate'] = np.nan
        # print(f"Error in calculating breathing rate: {e}")

    return working_data, measures


def _sliding_window(data, windowsize):
    """
    功能: 创建一个滑动窗口视图，用于高效计算。
    这是一个底层的NumPy技巧，通过调整数组的步幅(strides)来创建所有可能的窗口，而无需实际复制数据。
    """
    shape = data.shape[:-1] + (data.shape[-1] - windowsize + 1, windowsize)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def rolling_mean(data, windowsize, sample_rate):
    """
    功能: 计算信号的滚动（滑动）平均值。
    """
    data_arr = np.array(data)
    # 使用_sliding_window高效地计算每个窗口的平均值
    rol_mean = np.mean(_sliding_window(data_arr, int(windowsize * sample_rate)), axis=1)
    
    # 由于滑动平均会使结果序列变短，需要在两端填充，以使其与原数据等长
    n_missvals = int(abs(len(data_arr) - len(rol_mean)) / 2)
    missvals_a = np.array([rol_mean[0]] * n_missvals)  # 用第一个值填充开头
    missvals_b = np.array([rol_mean[-1]] * n_missvals) # 用最后一个值填充结尾
    rol_mean = np.concatenate((missvals_a, rol_mean, missvals_b))
    
    # 确保最终长度完全一致
    if len(rol_mean) != len(data):
        lendiff = len(rol_mean) - len(data)
        if lendiff < 0:
            rol_mean = np.append(rol_mean, 0)
        else:
            rol_mean = rol_mean[:-1]
    return rol_mean


def fit_peaks(hrdata, rol_mean, sample_rate, bpmmin=40, bpmmax=180, working_data={}):
    """
    功能: 自动寻找最佳的峰值检测阈值。
    通过尝试一系列不同的阈值（ma_perc），找到一个能产生在[bpmmin, bpmmax]范围内最稳定心率的阈值。
    """
    ma_perc_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 300]
    rrsd = []
    valid_ma = []
    # 遍历所有可能的阈值百分比
    for ma_perc in ma_perc_list:
        # 使用当前阈值检测峰值
        working_data = detect_peaks(hrdata, rol_mean, ma_perc, sample_rate,
                                    update_dict=True, working_data=working_data)
        # 计算该阈值下的BPM
        bpm = ((len(working_data['peaklist']) / (len(hrdata) / sample_rate)) * 60)
        # 记录结果：RR间期的标准差（反映稳定性），BPM，以及当前阈值
        rrsd.append([working_data['rrsd'], bpm, ma_perc])
    # 筛选出所有BPM在有效���围内的结果
    for _rrsd, _bpm, _ma_perc in rrsd:
        if (_rrsd > 0.1) and ((bpmmin <= _bpm <= bpmmax)):
            valid_ma.append([_rrsd, _ma_perc])
    # 如果有有效的结果
    if len(valid_ma) > 0:
        # 选择RRSD最小（最稳定）的那个结果作为最佳阈值
        working_data['best'] = min(valid_ma, key=lambda t: t[0])[1]
        # 使用最佳阈值重新进行一次峰值检测，以更新最终的工作数据
        working_data = detect_peaks(hrdata, rol_mean, min(valid_ma, key=lambda t: t[0])[1],
                                    sample_rate, update_dict=True, working_data=working_data)
        return working_data
    else:
        # 如果没有找到任何有效的阈值，则抛出异常
        raise Exception('Could not determine best fit for signal.')


def detect_peaks(hrdata, rol_mean, ma_perc, sample_rate, update_dict=True, working_data={}):
    """
    功能: 在rPPG信号中检测心跳波峰。
    基本原理是：当信号值超过一个动态阈值（滚动平均值 + 一个百分比偏移）时，就认为可能是一个波峰区域。
    """
    # 计算动态阈值
    rmean = np.array(rol_mean)
    mn = np.mean(rmean / 100) * ma_perc
    rol_mean = rmean + mn
    # 找到所有信号值高于动态阈值的点的索引
    peaksx = np.where((hrdata > rol_mean))[0]
    peaksy = hrdata[peaksx]
    # 将连续的高于阈值的点分组，每组代表一个可能的波峰
    peakedges = np.concatenate((np.array([0]), (np.where(np.diff(peaksx) > 1)[0]), np.array([len(peaksx)])))
    peaklist = []
    # 遍历每个分组
    for i in range(0, len(peakedges) - 1):
        try:
            # 找到该组中的最大值点，作为该波峰的顶点
            y_values = peaksy[peakedges[i]:peakedges[i + 1]].tolist()
            peaklist.append(peaksx[peakedges[i] + y_values.index(max(y_values))])
        except:
            # 如果出错则跳过
            pass
    # 如果需要，更新工作字典
    if update_dict:
        working_data['peaklist'] = peaklist  # 峰值点的x轴索引
        working_data['ybeat'] = [hrdata[x] for x in peaklist]  # 峰值点的y轴值
        working_data['rolling_mean'] = rol_mean  # 最终使用的滚动平均阈值
        # 计算RR间期及其标准差
        working_data = calc_rr(working_data['peaklist'], sample_rate,
                               working_data=working_data)
        if len(working_data['RR_list']) > 0:
            working_data['rrsd'] = np.std(working_data['RR_list'])
        else:
            working_data['rrsd'] = np.inf
        return working_data
    else:
        return peaklist, working_data


def calc_rr(peaklist, sample_rate, working_data={}):
    """
    功能: 根据峰值列表计算RR间期（单位：毫秒）。
    RR间期是相邻两个心跳波峰之间的时间间隔。
    """
    peaklist = np.array(peaklist)
    # 原始代码中有一个删除第一个峰值的逻辑，如果它离开头太近的话。予以保留。
    if len(peaklist) > 0:
        if peaklist[0] <= ((sample_rate / 1000.0) * 150):
            peaklist = np.delete(peaklist, 0)
            working_data['peaklist'] = peaklist
            working_data['ybeat'] = np.delete(working_data['ybeat'], 0)
    # 计算RR间期列表：(相邻峰值索引差 / 采样率) * 1000
    rr_list = (np.diff(peaklist) / sample_rate) * 1000.0
    # 记录每个RR间期对应的原始峰值索引对
    rr_indices = [(peaklist[i], peaklist[i + 1]) for i in range(len(peaklist) - 1)]
    # 计算相邻RR间期的差值和差值的平方，用于后续的HRV计算
    rr_diff = np.abs(np.diff(rr_list))
    rr_sqdiff = np.power(rr_diff, 2)
    # 更新工作字典
    working_data['RR_list'] = rr_list
    working_data['RR_indices'] = rr_indices
    working_data['RR_diff'] = rr_diff
    working_data['RR_sqdiff'] = rr_sqdiff
    return working_data


def check_peaks(rr_arr, peaklist, ybeat, reject_segmentwise=False, working_data={}):
    """
    功能: 检查并标记异常的RR间期。
    如果一个RR间期与平均RR间期的差异过大，则认为它是一个异常搏动，并将其标记。
    """
    rr_arr = np.array(rr_arr)
    peaklist = np.array(peaklist)
    ybeat = np.array(ybeat)
    mean_rr = np.mean(rr_arr)
    # 定义异常的阈值，为平均RR的30%或300ms中的较小者（此处遵循原始代码逻辑）
    thirty_perc = 0.3 * mean_rr
    if thirty_perc <= 300:
        upper_threshold = mean_rr + 300
        lower_threshold = mean_rr - 300
    else:
        upper_threshold = mean_rr + thirty_perc
        lower_threshold = mean_rr - thirty_perc
    # 找到超出阈值的RR间期的索引
    rem_idx = np.where((rr_arr <= lower_threshold) | (rr_arr >= upper_threshold))[0] + 1
    # 记录被移除的搏动点
    working_data['removed_beats'] = peaklist[rem_idx]
    working_data['removed_beats_y'] = ybeat[rem_idx]
    # 创建一个二进制列表，标记每个峰值是否可信（1为可信，0为不可信）
    working_data['binary_peaklist'] = np.asarray([0 if x in working_data['removed_beats']
                                                  else 1 for x in peaklist])
    # 如果设置了分段拒绝，则执行分段质量检查
    if reject_segmentwise:
        working_data = check_binary_quality(peaklist, working_data['binary_peaklist'],
                                            working_data=working_data)
    # 更新RR列表，只保留那些由两个连续可信峰值构成的RR间期
    working_data = update_rr(working_data=working_data)
    return working_data


def check_binary_quality(peaklist, binary_peaklist, maxrejects=3, working_data={}):
    """
    功能: 分段检查信号质量。如果一个10个搏动的段内有超过maxrejects个异常搏动，则拒绝整个段。
    """
    idx = 0
    working_data['rejected_segments'] = []
    # 将信号分为10个搏动长度的段
    for i in range(int(len(binary_peaklist) / 10)):
        # 统计该段内被拒绝的搏动数
        if np.bincount(binary_peaklist[idx:idx + 10])[0] > maxrejects:
            # 如果超过阈值，则将整个段的搏动都标记为不可信
            binary_peaklist[idx:idx + 10] = [0 for _ in range(len(binary_peaklist[idx:idx + 10]))]
            # 记录被拒绝的段的起止点
            if idx + 10 < len(peaklist):
                working_data['rejected_segments'].append((peaklist[idx], peaklist[idx + 10]))
            else:
                working_data['rejected_segments'].append((peaklist[idx], peaklist[-1]))
        idx += 10
    return working_data


def update_rr(working_data={}):
    """
    功能: 根据二进制峰值可信度列表，更新RR相关的所有数据。
    """
    rr_source = working_data['RR_list']
    b_peaklist = working_data['binary_peaklist']
    # 只有当一个RR间期的起始和结束峰值都可信时，才保留该RR间期
    rr_list = [rr_source[i] for i in range(len(rr_source)) if b_peaklist[i] + b_peaklist[i + 1] == 2]
    # 创建一个掩码，标记哪些RR间期被剔除了
    rr_mask = [0 if (b_peaklist[i] + b_peaklist[i + 1] == 2) else 1 for i in range(len(rr_source))]
    # 使用掩码数组重新计算RR_diff和RR_sqdiff
    rr_masked = np.ma.array(rr_source, mask=rr_mask)
    rr_diff = np.abs(np.diff(rr_masked))
    rr_diff = rr_diff[~rr_diff.mask]  # 移除被掩码的值
    rr_sqdiff = np.power(rr_diff, 2)
    # 更新工作字典
    working_data['RR_masklist'] = rr_mask
    working_data['RR_list_cor'] = rr_list # 修正后的RR列表
    working_data['RR_diff'] = rr_diff
    working_data['RR_sqdiff'] = rr_sqdiff
    return working_data


def calc_ts_measures(rr_list, rr_diff, rr_sqdiff, measures={}, working_data={}):
    """
    功能: 计算所有时域（Time Series）的HRV指标。
    """
    if not rr_list: return working_data, measures  # 如果没有有效的RR间期，则直接返回
    
    measures['BPM'] = 60000 / np.mean(rr_list)  # 每分钟心跳数
    measures['IBI'] = np.mean(rr_list)  # 平均心跳间隔 (Inter-Beat Interval)
    measures['SDNN'] = np.std(rr_list)  # 所有RR间期的标准差，反映总体变异性
    measures['SDSD'] = np.std(rr_diff) if len(rr_diff) > 0 else 0  # 相邻RR间期差值的标准差
    measures['RMSSD'] = np.sqrt(np.mean(rr_sqdiff)) if len(rr_sqdiff) > 0 else 0  # 相邻RR间期差值平方和的均方根，反映副交感神经活动
    
    # 计算pNN20和pNN50
    nn20 = rr_diff[np.where(rr_diff > 20.0)]
    nn50 = rr_diff[np.where(rr_diff > 50.0)]
    working_data['nn20'] = nn20
    working_data['nn50'] = nn50
    
    measures['pNN20'] = (float(len(nn20)) / float(len(rr_diff))) * 100 if len(rr_diff) > 0 else 0
    measures['pNN50'] = (float(len(nn50)) / float(len(rr_diff))) * 100 if len(rr_diff) > 0 else 0
    
    # 计算MAD（Median Absolute Deviation）
    med = np.median(rr_list)
    measures['MAD'] = np.median(np.abs(rr_list - med))
    return working_data, measures


def calc_poincare(rr_list, rr_mask=[], measures={}, working_data={}):
    """
    功能: 计算庞加莱图（Poincaré Plot）相关指标。
    庞加莱图是一种将每个RR间期(i)与其后一个RR间期(i+1)作为(x, y)坐标绘制的散点图。
    """
    if len(rr_list) < 2:
        measures.update({'SD1': 0, 'SD2': 0, 'S': 0, 'SD1/SD2': 0})
        working_data['poincare'] = {}
        return measures

    # x轴是RR(i)，y轴是RR(i+1)
    x_plus = np.asarray(rr_list[:-1])
    x_minus = np.asarray(rr_list[1:])
    
    # 将坐标系旋转45度
    x_one = (x_plus - x_minus) / np.sqrt(2)
    x_two = (x_plus + x_minus) / np.sqrt(2)
    
    # SD1是垂直于y=x对角线方向的标准差，反映短期变异性（与RMSSD相关）
    sd1 = np.sqrt(np.var(x_one))
    # SD2是沿着y=x对角线方向的标准差，反映长期变异性
    sd2 = np.sqrt(np.var(x_two))
    # S是庞加莱图椭圆的面积
    s = np.pi * sd1 * sd2
    
    measures['SD1'] = sd1
    measures['SD2'] = sd2
    measures['S'] = s
    measures['SD1/SD2'] = sd1 / sd2 if sd2 != 0 else 0
    
    working_data['poincare'] = {'x_plus': x_plus, 'x_minus': x_minus}
    return measures


def calc_breathing(rrlist, method='welch', filter_breathing=True,
                   bw_cutoff=[0.1, 0.4], measures={}, working_data={}):
    """
    功能: 从RR间期序列中估算呼吸率。
    原理是呼吸会引起心率的周期性变化（呼吸性窦性心律不齐, RSA），通过��谱分析可以找到这个周期。
    """
    if len(rrlist) < 5:
        measures['breathing_rate'] = 0
        return measures, working_data
    # 首先，将离散的RR间期序列插值为等时间间隔的连续信号
    x = np.linspace(0, len(rrlist), len(rrlist))
    x_new = np.linspace(0, len(rrlist), int(np.sum(rrlist)))
    interp = UnivariateSpline(x, rrlist, k=3)
    breathing = interp(x_new)
    # 对插值后的信号进行带通滤波，只保留可能的呼吸频率范围
    if filter_breathing:
        breathing = filtering.filter_signal(breathing, cutoff=bw_cutoff,
                                            sample_rate=1000.0, filtertype='bandpass')
    # 对插值后的信号进行频谱分析
    if method.lower() == 'welch':
        frq, psd = welch(breathing, fs=1000, nperseg=len(breathing))
    else: # 备用方法：FFT
        datalen = len(breathing)
        frq = np.fft.fftfreq(datalen, d=(1 / 1000.0))
        frq = frq[range(int(datalen / 2))]
        Y = np.fft.fft(breathing) / datalen
        Y = Y[range(int(datalen / 2))]
        psd = np.power(np.abs(Y), 2)
        
    # 找到功率谱密度(PSD)最大的频率，即为主导呼吸频率
    measures['breathing_rate'] = frq[np.argmax(psd)] * 60 # 转换为每分钟呼吸次数
    return measures, working_data

def calc_freq_domain(rr_list, fs=4.0):
    """
    功能: 计算频域HRV指标 (VLF, LF, HF)。
    """
    if len(rr_list) < 4:
        return {'vlf': 0, 'lf': 0, 'hf': 0, 'lf_hf_ratio': 0, 'total_power': 0}
    # 同样，先将RR间期插值为等间隔信号
    t = np.cumsum(np.asarray(rr_list) / 1000.0)
    interp_t = np.arange(0, t[-1], 1.0 / fs)
    interp_rr = np.interp(interp_t, t, rr_list)
    # 使用Welch方法计算功率谱密度
    freqs, psd = welch(interp_rr, fs=fs, nperseg=len(interp_rr))
    # 计算不同频带的功率（通过积分）
    vlf_power = np.trapz(psd[(freqs >= 0.0033) & (freqs < 0.04)], freqs[(freqs >= 0.0033) & (freqs < 0.04)]) # 极低频
    lf_power = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)])   # 低频，反映交感和副交感活动
    hf_power = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], freqs[(freqs >= 0.15) & (freqs < 0.4)])    # 高频，主要反映副交感神经活动
    total_power = vlf_power + lf_power + hf_power
    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else np.inf # LF/HF比值，反映交感-副交感平衡
    return {
        'vlf': vlf_power, 'lf': lf_power, 'hf': hf_power,
        'lf_hf_ratio': lf_hf_ratio, 'total_power': total_power
    }

def analyze_advanced_metrics(hrdata, sample_rate, working_data, measures):
    """
    功能: 基于已计算的指标，进行更高级的、解释性的状态评估。
    注意：这些是基于通用规则的初步评估，不能替代专业医疗诊断。
    """
    rr_list = np.array(working_data.get('RR_list_cor', []))
    peaklist = working_data.get('peaklist', [])
    
    # --- 心律状态评估 ---
    if len(rr_list) > 5:
        rr_mean = np.mean(rr_list)
        rr_diff = np.abs(np.diff(rr_list))
        # 检查是否存在早搏（PVC）的模式
        pvc_count = 0
        for i in range(len(rr_list) - 1):
            if rr_list[i] < rr_mean * 0.75 and rr_list[i+1] > rr_mean * 1.25:
                pvc_count += 1
        # 检查是否存在房颤（AF）的可能性
        af_irregularity_index = np.sum(rr_diff > rr_mean * 0.2) / len(rr_diff) if len(rr_diff) > 0 else 0
        if af_irregularity_index > 0.4:
            measures['rhythm_status'] = "Likely AF"
        elif pvc_count > 0:
            measures['rhythm_status'] = f"PVC ({pvc_count})"
        elif measures.get('SDNN', 0) > (rr_mean * 0.15):
             measures['rhythm_status'] = "Irregular"
        else:
            measures['rhythm_status'] = "Normal"
    else:
        measures['rhythm_status'] = "N/A"

    # --- 血管状态评估 (基于脉搏波形态) ---
    if len(peaklist) >= 3:
        # 取最后一个完整的心动周期
        p1, p2 = peaklist[-2], peaklist[-1]
        one_cycle = hrdata[p1:p2]
        if len(one_cycle) > 0:
            # 找到收缩期峰值（systolic peak）
            systolic_idx = np.argmax(one_cycle)
            try:
                # 尝试在收缩期峰值后找到重搏波（dicrotic wave）和切迹（notch）
                notch_search_area = one_cycle[systolic_idx:]
                peaks_after, _ = find_peaks(notch_search_area)
                valleys_after, _ = find_peaks(-notch_search_area)
                dicrotic_idx = systolic_idx + peaks_after[0] if len(peaks_after) > 0 else systolic_idx
                notch_idx = systolic_idx + valleys_after[0] if len(valleys_after) > 0 else systolic_idx
            except IndexError:
                # 如果找不到，则使用收缩期峰值作为替代
                dicrotic_idx = systolic_idx
                notch_idx = systolic_idx
            # 计算脉搏波形态相关指标
            measures['rise_time'] = systolic_idx / sample_rate # 上升时间
            measures['decay_time'] = (dicrotic_idx - systolic_idx) / sample_rate # 衰减时间
            measures['pulse_area'] = np.trapz(one_cycle, dx=1/sample_rate) # 脉搏波面积
            systolic_pressure = one_cycle[systolic_idx]
            dicrotic_pressure = one_cycle[dicrotic_idx]
            measures['reflection_index'] = (dicrotic_pressure / systolic_pressure) * 100 if systolic_pressure > 0 else 0 # 反射指数
            measures['augmentation_index'] = (systolic_pressure - dicrotic_pressure) / systolic_pressure * 100 if systolic_pressure > 0 else 0 # 增强指数
            # 基于指数评估血管状态
            if measures['reflection_index'] < 30 and measures['augmentation_index'] < 10:
                measures['vascular_status'] = "Good"
            elif measures['reflection_index'] < 40 and measures['augmentation_index'] < 20:
                measures['vascular_status'] = "Moderate"
            else:
                measures['vascular_status'] = "Poor"
    
    # --- 压力和焦虑水平评估 ---
    # 计算频域指标
    freq_metrics = calc_freq_domain(rr_list)
    measures.update(freq_metrics)
    lf_hf_ratio = measures.get('lf_hf_ratio', 1.5)
    rmssd = measures.get('RMSSD', 50)
    # 基于LF/HF比值评估压力水平
    if lf_hf_ratio > 2.5:
        measures['stress_level'] = "High"
    elif lf_hf_ratio > 1.5:
        measures['stress_level'] = "Medium"
    else:
        measures['stress_level'] = "Low"
    # 基于RMSSD评估焦虑水平（低RMSSD可能与焦虑相关）
    if rmssd < 25:
        measures['anxiety_level'] = "High"
    elif rmssd < 45:
        measures['anxiety_level'] = "Medium"
    else:
        measures['anxiety_level'] = "Low"
        
    return measures, working_data