# -*- coding: utf-8 -*-
"""
文件描述:
本模块提供了一系列用于生理信号处理的滤波函数。
用于实现标准的数字滤波操作，
如低通、高通、带通滤波，以及一些特殊的去噪方法，如 Hampel 滤波。
这些工具是信号预处理的关键步骤，旨在消除噪声、去除基线漂移，从而提取出有用的生理信息。

原作者: Yongzhi

"""

# 导入必要的库
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter  # 导入滤波器设计、应用、陷波和Savitzky-Golay滤波器
import numpy as np 

# 定义此模块公开的函数名，当使用 'from filtering import *' 时，只有这些函数会被导入
__all__ = ['filter_signal',
           'hampel_filter',
           'hampel_correcter',
           'smooth_signal']

def butter_lowpass(cutoff, sample_rate, order=2):
    '''
    功能: 设计一个标准的巴特沃斯低通滤波器。

    此函数定义了一个巴特沃斯低通滤波器，所有高于 `cutoff` 频率的信号成分都将被衰减。

    参数:
    ----------
    cutoff : int 或 float
        滤波器的截止频率（单位：赫兹 Hz）。
        所有高于此频率的信号都将被滤除。

    sample_rate : int 或 float
        输入信号的采样率（单位：赫兹 Hz）。

    order : int, 可选
        滤波器的阶数，决定了在截止频率附近的衰减速度（滚降陡峭程度）。
        阶数越高，衰减越快，但可能引入更多振荡。通常不常用超过6阶。
        默认值: 2

    返回:
    -------
    out : tuple
        返回一个元组 (b, a)，其中 b 和 a 分别是IIR（无限脉冲响应）滤波器的
        分子和分母多项式系数。
    '''
    # 计算奈奎斯特频率，即采样率的一半，这是数字信号能表示的最高频率
    nyq = 0.5 * sample_rate
    # 将截止频率归一化到 [0, 1] 的范围，其中1对应奈奎斯特频率
    normal_cutoff = cutoff / nyq
    # 使用scipy.signal.butter函数设计滤波器
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, sample_rate, order=2):
    '''
    功能: 设计一个标准的巴特沃斯高通滤波器。

    此函数定义了一个巴特沃斯高通滤波器，所有低于 `cutoff` 频率的信号成分都将被衰减。

    参数:
    ----------
    cutoff : int 或 float
        滤波器的截止频率（单位：赫兹 Hz）。
        所有低于此频率的信号都将被滤除。

    sample_rate : int 或 float
        输入信号的采样率（单位：赫兹 Hz）。

    order : int, 可选
        滤波器的阶数。
        默认值: 2

    返回:
    -------
    out : tuple
        返回 (b, a) 元组，即滤波器的分子和分母系数。
    '''
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_bandpass(lowcut, highcut, sample_rate, order=2):
    '''
    功能: 设计一个标准的巴特沃斯带通滤波器。

    此函数定义了一个巴特沃斯带通滤波器，只允许在 [lowcut, highcut] 频率范围内的信号通过。

    参数:
    ----------
    lowcut : int 或 float
        带通滤波器的下限截止频率（单位：赫兹 Hz）。

    highcut : int 或 float
        带通滤波器的上限截止频率（单位：赫兹 Hz）。

    sample_rate : int 或 float
        输入信号的采样率（单位：赫兹 Hz）。

    order : int, 可选
        滤波器的阶数。
        默认值: 2

    返回:
    -------
    out : tuple
        返回 (b, a) 元组，即滤波器的分子和分母系数。
    '''
    nyq = 0.5 * sample_rate
    # 将通带的上下限频率都进行归一化
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass',
                  return_top = False):
    '''
    功能: 对输入数据应用指定的滤波器。

    这是一个通用的滤波函数，可以根据 `filtertype` 参数应用低通、高通、带通或陷波滤波器。

    参数:
    ----------
    data : 1-dimensional numpy array or list 
        需要被滤波的一维信号数据。

    cutoff : int, float or tuple
        滤波器的截止频率。
        - 对于 'lowpass' 和 'highpass'，它是一个数字。
        - 对于 'bandpass'，它是一个包含 [下限, 上限] 的列表或元组。
        - 对于 'notch'，它是要滤除的中心频率。

    sample_rate : int or float
        信号的采样率。

    order : int
        滤波器阶数 
        默认值 : 2

    filtertype : str
        要应用的滤波器类型。可选值:
        - 'lowpass': 低通滤波器
        - 'highpass': 高通滤波器
        - 'bandpass': 带通滤波器
        - 'notch': 陷波滤波器 (用于消除特定频率干扰，如工频干扰或基线漂移)
        默认值: 'lowpass'

    return_top : bool, 可选
        如果为 True，则只返回信号中大于等于0的部分（将所有负值截断为0）。
        这有时用于只关注信号的波峰部分。
        默认值: False

    返回:
    -------
    out : 1d array
        经过滤波后的一维信号数据。
    '''
    # 根据 filtertype 设计相应的滤波器
    if filtertype.lower() == 'lowpass':
        b, a = butter_lowpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'highpass':
        b, a = butter_highpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'bandpass':
        # 断言检查，确保带通滤波器的截止频率是列表或元组格式
        assert type(cutoff) == tuple or list or np.array, 'if bandpass filter is specified, \
cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
        b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
    elif filtertype.lower() == 'notch':
        # 设计一个陷波滤波器，Q因子设为0.005，表示一个非常窄的陷波
        b, a = iirnotch(cutoff, Q = 0.005, fs = sample_rate)
    else:
        # 如果 filtertype 无效，则抛出异常
        raise ValueError('filtertype: %s is unknown, available are: \
lowpass, highpass, bandpass, and notch' %filtertype)

    # 应用滤波器。使用 filtfilt 进行零相位滤波，它会向前和向后两次应用滤波器，
    # 从而避免了普通滤波（lfilter）带来的相位延迟。
    filtered_data = filtfilt(b, a, data)
    
    # 根据 return_top 参数决定是否截断负值
    if return_top:
        return np.clip(filtered_data, a_min = 0, a_max = None)
    else:
        return filtered_data


def remove_baseline_wander(data, sample_rate, cutoff=0.05):
    '''
    功能: 使用陷波滤波器移除信号的基线漂移。

    基线漂移是生理信号中常见的低频噪声，此函数通过一个低截止频率的陷波滤波器来消除它。

    参数:
    ----------
    data : 1-dimensional numpy array or list 
        需要处理的一维信号数据。

    sample_rate : int or float
        信号的采样率。

    cutoff : int, float 
        陷波滤波器的截止频率。推荐值为0.05Hz，适用于典型的基线漂移。
        默认值 : 0.05

    返回:
    -------
    out : 1d array
        移除了基线漂移的信号数据。
    '''
    # 内部调用 filter_signal 函数，并指定类型为 'notch'
    return filter_signal(data = data, cutoff = cutoff, sample_rate = sample_rate,
                         filtertype='notch')


def hampel_filter(data, filtsize=6):
    '''
    功能: 使用Hampel滤波器检测并修正异常值（outliers）。
    
Hampel滤波器是一种基于滑动窗口的中值滤波器。对于窗口中的每一个样本点，
如果它与窗口中值的偏差超过了3倍的绝对中位差（MAD），
则认为该点是异常值，并用窗口的中值替换它。这是一种稳健的去噪方法。
    
    参数:
    ----------
    data : 1d list or array
        需要被滤波的数据。

    filtsize : int
        滑动窗口的大小，表示中心点周围采样的点数。
        例如，filtsize=6 表示在中心点的每一侧取3个点。
        总的窗口大小是 filtsize + 1。
        默认值: 6

    返回:
    -------
    out :  array containing filtered data
    '''

    # 创建一个数据的副本，以避免修改原始输入数组
    output = np.copy(np.asarray(data)) 
    # 计算单侧窗口的大小
    onesided_filt = filtsize // 2
    # 遍历数据，注意要留出窗口的边界
    for i in range(onesided_filt, len(data) - onesided_filt - 1):
        # 获取当前点的滑动窗口数据切片
        dataslice = output[i - onesided_filt : i + onesided_filt]
        
        # 计算窗口的中值 (median)
        med = np.median(dataslice)
        # 计算绝对中位差 (Median Absolute Deviation, MAD)，这是一种稳健的标准差估计
        mad = np.median(np.abs(dataslice - med))
        # 再次计算中值（此处代码重复，但遵循原始逻辑）
        median = np.median(dataslice)
        
        # 判断当前点是否为异常值：如果它与中值的偏差超过3倍的MAD
        if output[i] > median + (3 * mad):
            # 如果是异常值，则用中值替换它
            output[i] = median
    return output


def hampel_correcter(data, sample_rate):
    '''
    功能: 应用一个修改版的Hampel滤波器来抑制噪声。

    此函数计算原始数据与经过1秒窗口Hampel滤波后的数据之间的差值。
    这可以有效地抑制噪声，但计算成本相对较高。
    应谨慎使用，仅在其他方法效果不佳时考虑。

    参数:
    ----------
    data : 1d numpy array
        需要被滤波的数据。

    sample_rate : int or float
        数据的采样率。
       
    返回:
    -------
    out : 1d numpy array
        经过修正后的数据。
    '''
    # 计算窗口大小，使其对应1秒的数据量，并返回原始数据与Hampel滤波结果的差值
    return data - hampel_filter(data, filtsize=int(sample_rate))


def quotient_filter(RR_list, RR_list_mask = [], iterations=2):
    '''
    功能: 应用商滤波器（Quotient Filter）来清理RR间期序列。

    此滤波器根据相邻RR间期的比率来判断是否存在异常。
    如果 RR(i) / RR(i+1) 的值超出了一个合理的范围（如 [0.8, 1.2]），
    则认为其中一个或两个间期是异常的。

    参数:
    ----------
    RR_list - 1d array or list
        需要被滤波的RR间期（心跳之间的间隔）序列。

    RR_list_mask - 1d array or list
        一个标记哪些间期被拒绝的掩码数组。0表示接受，1表示拒绝。
        如果未提供，将自动生成一个全零的掩码。

    iterations - int
        应用滤波器的次数。多次迭代会产生更强的滤波效果。
        默认值 : 2

    返回:
    -------
    RR_list_mask : 1d array
        更新后的掩码数组。
    '''
    # 如果没有提供掩码，则初始化一个全零的掩码
    if len(RR_list_mask) == 0:
        RR_list_mask = np.zeros((len(RR_list)))
    else:
        # 确保RR列表和掩码列表长度一致
        assert len(RR_list) == len(RR_list_mask), \
        'error: RR_list and RR_list_mask should be same length if RR_list_mask is specified'

    # 进行指定次数的迭代
    for iteration in range(iterations):
        # 遍历RR间期序列
        for i in range(len(RR_list) - 1):
            # 如果相邻的两个间期中已有一个被拒绝，则跳过
            if RR_list_mask[i] + RR_list_mask[i + 1] != 0:
                pass #skip if one of both intervals is already rejected
            # 如果两个间期的比率在 [0.8, 1.2] 之间，认为它们是正常的，跳过
            elif 0.8 <= RR_list[i] / RR_list[i + 1] <= 1.2:
                pass #if R-R pair seems ok, do noting
            # 否则，认为这对间期有问题，更新掩码
            else: #update mask
                RR_list_mask[i] = 1
                #RR_list_mask[i + 1] = 1

    return np.asarray(RR_list_mask)


def smooth_signal(data, sample_rate, window_length=None, polyorder=3):
    '''
    功能: 使用Savitzky-Golay滤波器平滑信号。

    Savitzky-Golay滤波器通过在滑动窗口内用一个低阶多项式拟合数据点，
    从而在平滑信号的同时能较好地保留信号的形状和宽度特征（如波峰）。

    参数:
    ----------
    data : 1d array or list
        需要被平滑的数据。

    sample_rate : int or float
        数据的采样率。

    window_length : int or None
        Savitzky-Golay滤波器的窗口长
        如果为None，则自动设置为采样率的10%。
        如果提供了偶数，会自动加1使其变为奇数。
        默认值 : None

    polyorder : int
        拟合多项式的阶数。必须小于窗口长度。
        默认值 : 3

    返回:
    -------
    smoothed : 1d array
        平滑后的数据。
    '''
    # 如果未指定窗口长度，则根据采样率计算一个默认值
    if window_length == None:
        window_length = sample_rate // 10
        
    # 确保窗口长度是奇数
    if window_length % 2 == 0 or window_length == 0: window_length += 1

    # 应用Savitzky-Golay滤波器
    smoothed = savgol_filter(data, window_length = window_length,
                             polyorder = polyorder)

    return smoothed