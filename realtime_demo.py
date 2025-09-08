# -*- coding: utf-8 -*-
"""

- **模型切换指南**: 
  (1) 推荐的命令行参数 `--model_type` 或 (2) 手动修改代码中的4个标记位置，来在
  N3DED8 和 N3DED128 模型之间进行切换。
  
Author: Yongzhi

"""

# --- 模型切换指南 ---
#
# 如何选择使用的模型:
#
# 方法一 (推荐): 通过命令行参数启动
#   - 运行 N3DED128 (默认): python realtime_demo.py
#   - 运行 N3DED128 (显式): python realtime_demo.py --model_type N3DED128
#   - 运行 N3DED8:         python realtime_demo.py --model_type N3DED8
#
# 方法二 (备用): 直接修改代码
#   - 如果您想硬编码使用的模型，请修改下面代码中带有 `#[模型切换]` 标记的4个位置。
#

# --- 1. 导入模块 (Imports) ---
# 代码段解释: 此代码块导入了项目运行所需的所有外部和本地Python库。
# 每个库都有其特定的功能，例如OpenCV用于视频处理，PyTorch用于神经网络，Matplotlib用于绘图。

import cv2  # OpenCV库，用于摄像头访问和基本的图像处理。
import uuid  # 通用唯一标识符库，用于生成全局唯一ID（UUID），常用于文件/会话/对象等标识。
import os  # 操作系统库，用于文件路径操作等。
import numpy as np  # NumPy库，用于高效的数值计算，特别是数组操作。
import mediapipe as mp  # Google的MediaPipe库，用于高效的人脸检测。
import torch  # PyTorch库，用于加载和运行深度学习模型。
from torch.utils.data import DataLoader  # PyTorch的数据加载工具。
import matplotlib.pyplot as plt  # Matplotlib库，用于创建图表和可视化界面。
import matplotlib.gridspec as gridspec  # Matplotlib的子模块，用于更灵活的子图布局。
import time  # 时间库，用于计时和控制动画。
from collections import deque  # 双端队列，一种高效的数据结构，用于实现固定大小的缓冲区。
import argparse # [新增] 导入命令行参数解析库，用于处理用户从命令行传入的参数。

# 导入本地模块
from networks import N3DED128, N3DED8, N3DED128_Enhanced  #[模型切换] 1. 确保两个模型都已导入
import rPPG_Process  # 从本地文件导入rPPG信号处理函数。
import filtering  # 从本地文件导入信号滤波函数。

# === 新增：指标入库（DuckDB/SQLite） ===
from storage import connect as db_connect, init_db as db_init, upsert_hrv_feature, insert_feature_hash
from datetime import datetime, timezone, timedelta

# === 新增：LLM交互 ===
from llm.schemas import validate_input, validate_output, pretty
from llm.prompt_builder import build_llm_input_payload, compute_confidence_bundle
from llm.llm_adapter import generate_structured_response

LLM_RESPONSE_STYLE_DEFAULT = "balanced"     # formal / casual / balanced
LLM_INPUT_LANGUAGE_DEFAULT = "zh-CN"
LLM_OUTPUT_LANGUAGE_DEFAULT = None          # None -> auto by style
SAVE_JSON_DUMPS = True

# --- 2. 全局配置 (Configuration) ---
# 代码段解释: 此代码块定义了整个脚本中使用的所有重要常量和参数。
# 将这些配置放在一起，方便用户根据自己的需求（如摄像头ID、模型参数）进行统一修改。

WEBCAM_ID = 0  # 摄像头ID。0通常代表内置摄像头。
FACE_DETECTION_CONFIDENCE = 0.5  # MediaPipe人脸检测的最小置信度阈值。

# --- 架构配置 (Architectural Configuration) ---
UNIT_WINDOW_FRAMES = 16      # 每次模型处理的帧数。在流畅度和稳定性之间取得平衡。
BPM_WINDOW_FRAMES = 128      # 用于计算BPM的滑动窗口大小（帧数）。
METRICS_WINDOW_FRAMES = 256  # 用于计算所有健康指标的滑动窗口大小（帧数）。
PPG_PLOT_WINDOW_SIZE = 600   # PPG图表上显示的数据点数量。
PPG_FILTER_CUTOFF = [0.7, 2.8]  # PPG信号带通滤波器的截止频率（Hz），用于滤除噪声并保留心率范围内的信号。

# === Storage / Window metadata config（新增） ===
DB_PATH = "hrv_data.duckdb"   # DuckDB 文件；若要用 SQLite：export HRV_DB_BACKEND=sqlite
WINDOW_OVERLAP = 0.5          # 仅用于入库的 sampling_spec 元数据；若工程已有 overlap 变量，就把它赋给这里
ENV_LABEL_DEFAULT = "rest"    # 默认环境标签；上层若有更准确的情景识别会覆盖

# 在没有实时数据时显示的默认健康指标，提供更友好的初始界面
DEFAULT_METRICS = {
    'BPM': 60.0, 'RMSSD': 30.0, 'SDNN': 50.0, 'pNN50': 10.0,
    'SD1/SD2': 1.0, 'lf_hf_ratio': 1.0, 'SD1': 20.0, 'SD2': 20.0,
    'stress_level': 'Low Stress', 'anxiety_level': 'Normal',
    'rhythm_status': 'Normal', 'vascular_status': 'Normal'
}


# --- 3. 健康状态解释器类 (Health Status Interpreter Class) ---
# 代码段解释: 此类是一个独立的模块，专门负责将数值化的生理指标转换为人类可读的文本状态和颜色。
# 这种设计将数据处理与数据解释分离开来，使得代码更清晰，更容易维护和扩展。

class HealthInterpreter:
    """
    HealthInterpreter 类
    功能: 将原始生理指标转换为定性健康状态，并提供解释和对应的颜色。
    (Converts raw metrics into qualitative health status and provides explanations and corresponding colors.)
    """
    
    def get_metric_info(self, metric, value):
        """
        函数解释: 根据指标名称和值，返回其健康解释和定性状态。
        (Returns the health explanation and qualitative status for a given metric and value.)
        
        参数 (Args):
            metric (str): 指标的名称 (例如, 'BPM')。
            value (float or str): 该指标的数值或状态值。
            
        返回 (Returns):
            tuple: (该指标的解释字符串, 定性状态字符串)。
        """
        status = "N/A"  # 默认状态为"不适用"
        explanation = ""  # 默认解释为空
        
        # 为每个指标定义解释文本和状态判断的阈值
        if metric == 'BPM':
            explanation = "Heart Beats Per Minute:"
            if value < 50: status = "Low (Bradycardia)"
            elif value > 100: status = "High (Tachycardia)"
            else: status = "Normal"
        elif metric == 'RMSSD': # 反映短期副交感神经（休息和消化）活动。
            explanation = "Reflects short-term parasympathetic\n (rest & digest) activity"
            if value < 20: status = "Low"
            elif value < 45: status = "Normal"
            else: status = "High"
        elif metric == 'SDNN': # 反映整体自主神经系统（ANS）健康
            explanation = "Reflects overall autonomic nervous system (ANS) health"
            if value < 50: status = "Low"
            else: status = "Normal"
        elif metric == 'pNN50': # 大节拍变化的比例；表示迷走神经张力
            explanation = "Proportion of large beat-to-beat changes; \n indicates vagal tone"
            if value > 15: status = "High"
            else: status = "Normal"
        elif metric == 'SD1/SD2': # 短期和长期HRV的比率；表示自主神经系统平衡 
            explanation = "Ratio of short-term to long-term HRV; \n indicates ANS balance"
            if value < 0.5: status = "Long-term Dominant"
            elif value > 1.5: status = "Short-term Dominant"
            else: status = "Balanced"
        elif metric == 'LF/HF Ratio':
            explanation = "Sympathovagal balance: \n a common indicator of physiological stress"
            if value > 2.5: status = "High Stress"
            elif value > 1.5: status = "Medium Stress"
            else: status = "Low Stress"
        elif metric == 'rhythm_status': # 心跳的规律性
            explanation = "Regularity of the heartbeat"
            status = value
        elif metric == 'vascular_status': # 动脉健康和僵硬度的指标
            explanation = "Indicator of arterial health & stiffness"
            status = value
        elif metric == 'stress_level': # 基于HRV的自主神经系统压力水平
            explanation = "Autonomic stress level based on HRV"
            status = value
        elif metric == 'anxiety_level': # 基于HRV的焦虑水平指标
            explanation = "Anxiety indicator based on HRV"
            status = value
            
        return explanation, status

    def get_status_color(self, status):
        """
        函数解释: 根据状态字符串返回对应的颜色代码。
        (Returns a color corresponding to the status string.)
        
        参数 (Args):
            status (str): 定性状态字符串 (例如, "High", "Normal")。
            
        返回 (Returns):
            str: 对应颜色的十六进制代码。
        """
        # 定义一个从状态到颜色的映射字典
        color_map = {
            "High": "#ff6b6b", "Medium": "#feca57", "Low": "#ff9ff3", "Balanced": "#7bed9f",
            "Normal": "#7bed9f", "Good": "#7bed9f", "Moderate": "#feca57",
            "Poor": "#ff6b6b", "Irregular": "#feca57", "Likely Atrial Fibrillation": "#ff6b6b",
            "High Stress": "#ff6b6b", "Medium Stress": "#feca57", "Low Stress": "#7bed9f",
            "Low (Bradycardia)": "#feca57", "High (Tachycardia)": "#feca57",
            "Long-term Dominant": "#feca57", "Short-term Dominant": "#feca57",
            "N/A": "#d3d3d3"
        }
        # 返回对应颜色，如果找不到则返回默认的灰色
        return color_map.get(status, "#d3d3d3")


# --- 4. 仪表盘UI类 (Dashboard UI Class) ---
# 代码段解释: 此类封装了所有与Matplotlib图表界面相关的操作，包括创建、配置和更新。
# 将所有UI逻辑集中在这个类中，使得主循环代码更简洁，只负责数据处理和调用UI更新。

class Dashboard:
    """
    Dashboard 类
    功能: 管理Matplotlib仪表盘的创建、布局、数据更新和重绘。
    (Manages the creation, layout, data update, and redrawing of the Matplotlib dashboard.)
    """
    def __init__(self):
        """
        构造函数: 初始化Figure、子图和所有绘图元素。
        (Constructor: Initializes the Figure, subplots, and all plotting elements.)
        """
        # --- 初始化画布和子图布局 ---
        self.fig = plt.figure(figsize=(20, 12)) # 创建一个大的图形窗口
        plt.style.use('seaborn-v0_8-darkgrid') # 设置一个美观的绘图风格
        gs = gridspec.GridSpec(3, 4, figure=self.fig) # 使用GridSpec创建灵活的子图网格
        
        # --- 创建并存储所有子图(Axes)对象 ---
        self.ax = {
            'ppg': self.fig.add_subplot(gs[0, :]),       # PPG信号图，占据第一整行
            'hrv': self.fig.add_subplot(gs[1, 0]),       # HRV（RR间期）图
            'poincare': self.fig.add_subplot(gs[1, 1]),  # 庞加莱散点图
            'freq': self.fig.add_subplot(gs[1, 2]),      # HRV频谱图
            'status': self.fig.add_subplot(gs[1, 3]),    # 健康状态评估面板
            'metrics': self.fig.add_subplot(gs[2, :]),   # 生理指标网格，占据第三整行
        }
        self.interpreter = HealthInterpreter() # 实例化健康状态解释器
        self._create_artists() # 调用私有方法创建所有绘图元素
        self._configure_axes() # 调用私有方法配置所有子图的外观
        self.fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05, hspace=0.7, wspace=0.3) # 调整子图间距
        
        # --- 初始化UI显示 ---
        # 用默认值初始化所有 artist 的数据，确保启动时UI有内容。
        self._initialize_artists_data()
        # 显示图形窗口
        self.fig.show()

    def _create_artists(self):
        """
        私有函数: 初始化所有需要动态更新的绘图元素（在Matplotlib中称为'artists'）。
        (Private method: Initializes all plotting elements that need to be dynamically updated, known as 'artists'.)
        """
        self.artists = {} # 创建一个字典来存储所有的artist
        self.artists['line_ppg'], = self.ax['ppg'].plot([], [], lw=2, color='cyan')
        self.artists['heart_red'] = self.ax['ppg'].text(0.95, 0.85, '❤', fontsize=30, color='red', ha='center', va='center', transform=self.ax['ppg'].transAxes, fontname='Segoe UI Emoji')
        self.artists['line_hrv'], = self.ax['hrv'].plot([], [], marker='o', markersize=4, linestyle='-', color='magenta')
        self.artists['scatter_poincare'] = self.ax['poincare'].scatter([], [], alpha=0.7, c='green')
        self.artists['ellipse_poincare'] = plt.matplotlib.patches.Ellipse((0, 0), 0, 0, angle=45, edgecolor='orange', fc='None', lw=2)
        self.ax['poincare'].add_patch(self.artists['ellipse_poincare']) # 将椭圆添加到庞加莱图中
        self.artists['bars_freq'] = self.ax['freq'].bar(['LF', 'HF', 'LF/HF'], [0, 0, 0], color=['#1f77b4', '#2ca02c', '#d62728'])
        
        self.ax['metrics'].axis('off') # 关闭指标网格的坐标轴
        self.artists['text_metrics'] = self._create_metric_grid(self.ax['metrics'], "Physiological Metrics", rows=2, cols=4)
        self.ax['status'].axis('off') # 关闭状态面板的坐标轴
        self.artists['text_status'] = self._create_status_panel(self.ax['status'], "Health Status Assessment", 4)

    def _create_status_panel(self, ax, title, num_metrics):
        """
        私有函数: 为健康状态面板创建文本元素。
        (Private method: Creates text elements for the health status panel.)
        """
        ax.set_title(title, weight='bold')
        panel = {}
        for i in range(num_metrics):
            y_pos = 0.85 - (i * 0.22)
            # 创建标签文本 (例如, "Stress Level:")
            panel[f'label_{i}'] = ax.text(0.05, y_pos, "", fontsize=11, weight='bold')
            # 创建值文本 (例如, "High")，并带有一个背景框
            panel[f'value_{i}'] = ax.text(0.95, y_pos, "", fontsize=12, weight='bold', ha='right',
                                          bbox=dict(boxstyle="round,pad=0.4", fc='gray', ec='none'))
        return panel

    def _create_metric_grid(self, ax, title, rows, cols):
        """
        私有函数: 为所有生理指标创建一个网格布局的文本元素。
        (Private method: Creates a grid layout of text elements for all physiological metrics.)
        """
        ax.set_title(title, weight='bold', fontsize=14)
        grid = {}
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            x_pos = c * (1.0 / cols) + 0.02
            y_pos = 0.8 - r * 0.4
            # 主指标标题 (例如, "RMSSD: 177.33")
            grid[f'metric_{i}'] = ax.text(x_pos, y_pos, "", fontsize=12, weight='bold') 
            # 指标的描述文本 (例如, "Reflects short-term...")
            grid[f'desc_{i}'] = ax.text(x_pos, y_pos - 0.15, "", fontsize=9, color='gray', wrap=True, va='top', ha='left',
                                       transform=ax.transAxes, bbox=dict(boxstyle='square,pad=0', fc='none', ec='none'))
            # 指标的状态标签 (例如, "Status: High")
            grid[f'status_{i}'] = ax.text(x_pos, y_pos - 0.1, "", fontsize=10,
                                          bbox=dict(boxstyle="round,pad=0.3", fc='gray', ec='none'))
        return grid

    def _configure_axes(self):
        """
        私有函数: 配置所有子图的标题、坐标轴标签和范围等。
        (Private method: Configures titles, labels, and ranges for all subplots.)
        """
        self.ax['ppg'].set_title("Live Photoplethysmography (PPG) Signal", weight='bold', fontsize=14)
        self.ax['ppg'].set_xlabel("Time (Frames)")
        self.ax['ppg'].set_ylim(-3.5, 3.5) # 为标准差归一化设置一个合适的初始Y轴范围
        self.ax['hrv'].set_title("HRV (RR Intervals)")
        self.ax['poincare'].set_title("Poincaré Plot")
        self.ax['freq'].set_title("HRV Power Spectrum")
        for key in ['hrv', 'poincare', 'freq']:
            self.ax[key].grid(True, linestyle='--', alpha=0.6)

    def _initialize_artists_data(self):
        """
        私有函数: 为所有 artist 设置初始的、默认的数据。
        此方法在 __init__ 期间被调用一次，以确保UI在启动时有内容显示。
        (Private method: Sets initial, default data for all artists to ensure the UI has content on startup.)
        """
        # 获取包含默认指标和图表数据的初始数据存储
        initial_data = get_initial_data_store()
        # 使用这个初始数据来更新所有UI元素
        self.update(initial_data, 30)

    def update(self, data_store, beat_font_size):
        """
        公共函数: 用最新的数据更新所有仪表盘元素。
        (Public method: Updates all dashboard elements with the latest data.)
        
        参数 (Args):
            data_store (dict): 包含所有最新计算数据的字典。
            beat_font_size (float): 用于心跳动画的当前字体大小。
        """
        # --- 1. 更新所有 artist 的数据 ---
        # 代码段解释: 这个代码块负责从 data_store 中提取最新的数据，并将其应用到对应的Matplotlib artist对象上。
        # 这只是更新了对象内部的数据，真正的屏幕重绘在最后一步完成。
        
        # 从持久化的 data_store 中获取所有需要显示的数据。
        m = data_store.get('measures', {})
        
        # 更新PPG曲线的数据。
        ppg_buffer = data_store.get('ppg_plot_buffer', [])
        self.artists['line_ppg'].set_data(np.arange(len(ppg_buffer)), ppg_buffer)
        self.ax['ppg'].set_xlim(0, PPG_PLOT_WINDOW_SIZE)
        if ppg_buffer and any(ppg_buffer):
            ppg_min, ppg_max = np.min(ppg_buffer), np.max(ppg_buffer)
            self.ax['ppg'].set_ylim(ppg_min - 0.2, ppg_max + 0.2)

        # 更新HRV（RR间期）图的数据。
        rr_list = data_store.get('rr_list', [])
        self.artists['line_hrv'].set_data(np.arange(len(rr_list)), rr_list)
        self.ax['hrv'].relim(); self.ax['hrv'].autoscale_view()

        # 更新庞加莱散点图和椭圆的数据。
        poincare_data = data_store.get('poincare_data')
        if poincare_data:
            x_plus, x_minus = poincare_data
            self.artists['scatter_poincare'].set_offsets(np.c_[x_plus, x_minus])
            self.artists['ellipse_poincare'].set_center((m.get('IBI', 0), m.get('IBI', 0)))
            self.artists['ellipse_poincare'].width = 2 * m.get('SD2', 0)
            self.artists['ellipse_poincare'].height = 2 * m.get('SD1', 0)
            self.ax['poincare'].relim(); self.ax['poincare'].autoscale_view()

        # 更新HRV频谱图的数据。
        freq_values = data_store.get('freq_values', [0, 0, 0])
        for bar, val in zip(self.artists['bars_freq'], freq_values):
            bar.set_height(val)
        self.ax['freq'].relim(); self.ax['freq'].autoscale_view()

        # 更新所有生理指标的文本和状态。
        metrics_to_display = [
            ('BPM', 'BPM'), ('RMSSD', 'RMSSD'), ('SDNN', 'SDNN'), ('pNN50', 'pNN50'),
            ('SD1/SD2', 'SD1/SD2'), ('lf_hf_ratio', 'LF/HF Ratio'), ('SD1', 'SD1'), ('SD2', 'SD2'),
        ]
        for i, (key, label) in enumerate(metrics_to_display):
            value = m.get(key, DEFAULT_METRICS.get(key, 0))
            format_str = "{:.2f}" if isinstance(value, float) and key not in ['BPM', 'pNN50'] else "{:.1f}"
            metric_text = f"{label}: {format_str.format(value)}"
            explanation, status = self.interpreter.get_metric_info(label, value)
            color = self.interpreter.get_status_color(status)
            self.artists['text_metrics'][f'metric_{i}'].set_text(metric_text)
            self.artists['text_metrics'][f'desc_{i}'].set_text(explanation)
            self.artists['text_metrics'][f'status_{i}'].set_text(f"Status: {status}")
            self.artists['text_metrics'][f'status_{i}'].get_bbox_patch().set_facecolor(color)

        # 更新健康状态评估面板的文本和状态。
        status_keys = ['stress_level', 'anxiety_level', 'rhythm_status', 'vascular_status']
        for i, key in enumerate(status_keys):
            value = m.get(key, DEFAULT_METRICS.get(key, 'N/A'))
            explanation, status = self.interpreter.get_metric_info(key, value)
            color = self.interpreter.get_status_color(status)
            self.artists['text_status'][f'label_{i}'].set_text(explanation + ":")
            self.artists['text_status'][f'value_{i}'].set_text(status)
            self.artists['text_status'][f'value_{i}'].get_bbox_patch().set_facecolor(color)

        # 更新心跳动画的字体大小。
        self.artists['heart_red'].set_fontsize(beat_font_size)
        
        # --- 2. 标准绘图更新 ---
        # 重绘整个画布。这是最简单、最可靠的更新方式。
        self.fig.canvas.draw()
        # 处理所有待处理的UI事件，确保界面响应。
        self.fig.canvas.flush_events()


# --- 5. 状态管理函数 ---
def get_initial_data_store(ppg_plot_window_size=PPG_PLOT_WINDOW_SIZE):
    """
    函数解释: 创建或重置一个包含默认值的数据存储字典。
    (Creates or resets a data store dictionary with default values.)
    
    功能:
    - 用于UI的初始化，确保启动时显示一个有意义的默认仪表盘。
    - 用于在长时间未检测到人脸时重置UI，避免显示空白。
    
    参数 (Args):
        ppg_plot_window_size (int): PPG绘图窗口的大小，用于创建相应大小的缓冲区。
    
    返回 (Returns):
        dict: 一个包含完整默认数据的字典，可以直接被UI使用。
    """
    # 创建一个填满0的PPG绘图缓冲区
    ppg_buffer = deque(np.zeros(ppg_plot_window_size), maxlen=ppg_plot_window_size)
    # 创建一个看起来合理的默认RR间期正弦波，使初始的HRV和庞加莱图不为空
    default_rr = 800 + 30 * np.sin(np.linspace(0, 2 * np.pi, 25))
    default_poincare_plus = default_rr[:-1]
    default_poincare_minus = default_rr[1:]
    
    # 返回一个结构完整的、包含所有默认值的data_store字典
    return {
        'measures': DEFAULT_METRICS.copy(), # 包含所有默认生理指标
        'rr_list': list(default_rr), # 默认的RR间期列表
        'poincare_data': (default_poincare_plus, default_poincare_minus), # 默认的庞加莱图数据
        'freq_values': [1.0, 1.0, 1.0], # 默认的频谱图数据
        'ppg_plot_buffer': list(ppg_buffer) # 默认的PPG绘图数据
    }


# --- 6. 主应用逻辑 (Main Application Logic) ---
def main():
    """
    主函数: 初始化所有组件并运行主循环。
    (Main function: initializes all components and runs the main loop.)
    """
    # --- [新增] 解析命令行参数 ---
    # 代码段解释: 这个代码块使用 argparse 库来处理用户从命令行传入的参数。
    # 这使得脚本更加灵活，用户无需修改代码就可以切换模型。
    
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(description='Real-time rPPG demo with selectable models.')
    # 添加 --model_type 参数，允许用户从命令行选择模型
    parser.add_argument('--model_type', type=str, default='N3DED128_Enhanced', 
                    choices=['N3DED8', 'N3DED128', 'N3DED128_Enhanced'],
                    help='Select the model to use (N3DED8, N3DED128, or N3DED128_Enhanced).')
    # 解析传入的参数
    args = parser.parse_args()

    # --- 初始化 MediaPipe 人脸检测 ---
    # 代码段解释: 此代码块初始化谷歌的MediaPipe人脸检测器，用于在视频帧中定位人脸。
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=FACE_DETECTION_CONFIDENCE)

    # --- [新增] 根据参数选择模型、权重和输入尺寸 ---
    # 代码段解释: 这是一个新的逻辑块，用于根据用户的命令行选择来动态配置模型。
    # 它决定了要加载哪个模型结构、哪个权重文件，以及模型需要多大尺寸的输入图像。
    
    # 选择计算设备（优先使用CUDA GPU，否则使用CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 检查用户选择的是哪个模型
    if args.model_type == 'N3DED128_Enhanced':
        # 如果是增强版 N3DED128
        print("Loading N3DED128_Enhanced model...")
        model = N3DED128_Enhanced() #[模型切换] 2. 选择要实例化的模型
        weights_path = 'weight_DLCN_H5_D128_enhance.pth.tar' #[模型切换] 3. 指定对应的权重文件
        model_input_size = 128 #[模型切换] 4. 指定对应的输入尺寸
    elif args.model_type == 'N3DED128':
        # 如果是原始 N3DED128
        print("Loading N3DED128 model...")
        model = N3DED128() #[模型切换] 2. 选择要实例化的模型
        weights_path = 'weight_DLCN_H5_D128.pth.tar' #[模型切换] 3. 指定对应的权重文件
        model_input_size = 128 #[模型切换] 4. 指定对应的输入尺寸
    else: 
        # 如果是 N3DED8 (或者任何其他情况)
        print("Loading N3DED8 model...")
        model = N3DED8() #[模型切换] 2. 选择要实例化的模型
        weights_path = 'weights.pth.tar.bak' #[模型切换] 3. 指定对应的权重文件
        model_input_size = 8 #[模型切换] 4. 指定对应的输入尺寸
    
    # --- 初始化 PyTorch 模型 ---
    # 代码段解释: 这个 try-except 结构用于加载模型权重，并处理可能发生的错误，如文件找不到或模型与权重不匹配。
    try:
        # 打印将要加载的权重文件路径
        print(f"Attempting to load weights from: {weights_path}")
        # 加载权重文件
        checkpoint = torch.load(weights_path, map_location=device)
        # 将加载的权重应用到模型中
        model.load_state_dict(checkpoint['model_state_dict'])
        # 将模型移动到正确的设备（CPU或GPU）
        model.to(device)
        # 将模型设置为评估模式（这对于推理很重要，会关闭Dropout等层）
        model.eval()
        # 打印成功信息
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        # 如果找不到权重文件，打印致命错误并退出
        print(f"FATAL: Weight file not found at '{weights_path}'.")
        print("Please ensure the correct weights file exists in the root directory.")
        return
    except Exception as e:
        # 如果发生其他错误（例如，模型和权重不匹配），打印致命错误并退出
        print(f"FATAL: An error occurred while loading the model weights: {e}")
        print("This might be due to a model/weight mismatch or a corrupted file.")
        return

    # --- 初始化 Matplotlib UI 和摄像头 ---
    # 代码段解释: 此代码块负责初始化所有与输入（摄像头）和输出（UI界面）相关的组件。
    plt.ion() # 开启Matplotlib的交互模式，允许窗口在不阻塞代码执行的情况下更新。
    dashboard = Dashboard() # 实例化我们自定义的仪表盘UI类。
    cap = cv2.VideoCapture(WEBCAM_ID) # 打开摄像头。
    if not cap.isOpened(): # 检查摄像头是否成功打开。
        print(f"Error: Cannot open webcam ID {WEBCAM_ID}.")
        return
    
    # === 新增：初始化数据库连接与建表（进入 while True 之前） ===
    DB_PATH = "hrv_data.duckdb"  # 如需 SQLite：export HRV_DB_BACKEND=sqlite
    db_conn = db_connect(DB_PATH)
    db_init(db_conn)
    # ==============================================================


    # --- 数据缓冲区和状态变量 ---
    # 代码段解释: 此代码块初始化了主循环中需要用到的所有变量。
    # 包括用于存储数据的缓冲区（deque），以及用于控制动画和逻辑的状态变量。
    frames_buffer = deque(maxlen=METRICS_WINDOW_FRAMES) # 存储处理过的面部帧
    raw_ppg_buffer = deque(maxlen=METRICS_WINDOW_FRAMES * 2) # 存储原始的、未滤波的PPG信号
    hrv_rr_buffer = deque(maxlen=100) # 为频域分析创建一个独立的、更长的RR间期缓冲区
    data_store = get_initial_data_store() # 获取包含默认值的初始数据存储
    last_valid_bpm = 60.0 # 上一个有效的心率值，用于动画同步
    last_beat_time = time.time() # 上次心跳动画触发的时间
    beat_font_size = 30 # 心形图标的初始字体大小
    target_font_size = 30 # 心形图标动画的目标字体大小
    frame_count = 0 # 帧计数器，用于控制处理频率
    last_face_detected_time = time.time() # 用于跟踪上次检测到人脸的时间
    
    # --- 主循环 ---
    # 代码段解释: 这是程序的核心。它会不断地从摄像头读取新的一帧，进行人脸检测，
    # 累积数据，调用模型进行推理，计算生理指标，并更新UI界面。
    while plt.fignum_exists(dashboard.fig.number): # 循环条件：只要UI窗口还存在就继续。
        ret, frame = cap.read() # 从摄像头读取一帧。ret是布尔值，表示是否成功；frame是图像数据。
        if not ret: 
            # 如果摄像头断开或视频文件结束，则等待用户按键后退出
            print("Webcam disconnected or video file ended. Press any key in the OpenCV window to exit.")
            cv2.waitKey(0)
            break 

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 将OpenCV默认的BGR图像格式转换为MediaPipe需要的RGB格式。
        results = face_detection.process(rgb_frame) # 使用MediaPipe进行人脸检测。

        # --- 人脸检测与数据累积 ---
        # 代码段解释: 此代码块检查是否在当前帧中检测到了人脸。如果检测到，就裁剪出人脸区域，
        # 进行预处理，并将其添加到数据缓冲区中，以备后续的模型推理。
        if results.detections: # 如果检测到人脸
            last_face_detected_time = time.time() # 更新最后一次检测到人脸的时间戳
            detection = results.detections[0] # 获取第一个检测到的人脸
            bboxC = detection.location_data.relative_bounding_box # 获取相对边界框
            ih, iw, _ = frame.shape # 获取图像尺寸
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih) # 计算绝对坐标

            if w > 0 and h > 0: # 确保边界框有效
                face_region = rgb_frame[y:y+h, x:x+w] # 裁剪出人脸区域
                if face_region.size > 0: # 确保人脸区域不为空
                    # [修改] 使用变量来设置正确的输入尺寸，以适应不同的模型
                    resized_face = cv2.resize(face_region, (model_input_size, model_input_size))
                    # 预处理（转换为YUV色彩空间）并添加到帧缓冲区
                    frames_buffer.append(cv2.cvtColor(resized_face, cv2.COLOR_RGB2YUV)) 
                    frame_count += 1 # 帧计数器加一
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # 在视频上绘制绿色矩形框
        else: # 如果没有检测到人脸
            # [核心逻辑] 如果超过2秒没有检测到人脸，则重置所有指标为默认状态
            if time.time() - last_face_detected_time > 2.0:
                print("No face detected for >2s. Resetting data store to default state...")
                data_store = get_initial_data_store() # 重置data_store
                frames_buffer.clear() # 清空帧缓冲区
                raw_ppg_buffer.clear() # 清空原始PPG信号缓冲区
                hrv_rr_buffer.clear() # 清空长时RR缓冲区
                frame_count = 0 # 必须同时重置帧计数器
                last_face_detected_time = time.time() # 重置计时器

        cv2.imshow('Webcam Feed -  Press "q" to quit', frame) # 显示摄像头画面
        if cv2.waitKey(1) & 0xFF == ord('q'): break # 按'q'键退出

        # --- 数据处理与计算 ---
        # 代码段解释: 当累积到足够的帧时 (UNIT_WINDOW_FRAMES)，此代码块被触发。
        # 它负责将累积的帧送入神经网络模型进行推理，然后对模型输出的原始rPPG信号进行一系列复杂的信号处理，
        # 以计算出BPM、HRV等所有生理指标。
        if frame_count >= UNIT_WINDOW_FRAMES:
            frame_count = 0 # 重置帧计数器
            
            # 防御性检查，确保我们有足够的数据帧来送入模型
            if len(frames_buffer) < UNIT_WINDOW_FRAMES:
                continue 

            with torch.no_grad(): # 在无梯度的上下文中运行模型，以节省计算资源
                # 准备模型的输入张量
                frame_chunk = list(frames_buffer)[-UNIT_WINDOW_FRAMES:]
                input_array = np.stack(frame_chunk, axis=0)
                input_tensor = torch.tensor(input_array, dtype=torch.float32)
                input_tensor = input_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)
                
                out = model(input_tensor) # 模型推理
                raw_ppg_buffer.extend(out.cpu().numpy().flatten()) # 将预测结果添加到原始PPG缓冲区

            # 当缓冲区足够计算BPM时
            if len(raw_ppg_buffer) >= BPM_WINDOW_FRAMES:
                sample_rate = 30.0
                bpm_signal = filtering.filter_signal(np.array(list(raw_ppg_buffer)[-BPM_WINDOW_FRAMES:]), cutoff=PPG_FILTER_CUTOFF, sample_rate=sample_rate, filtertype='bandpass')
                _, bpm_measures = rPPG_Process.process(bpm_signal, sample_rate)
                if bpm_measures.get('BPM', 0) > 30:
                    last_valid_bpm = bpm_measures['BPM']

            # 当缓冲区足够计算所有指标时
            if len(raw_ppg_buffer) >= METRICS_WINDOW_FRAMES:
                sample_rate = 30.0
                metrics_signal = filtering.filter_signal(np.array(raw_ppg_buffer), cutoff=PPG_FILTER_CUTOFF, sample_rate=sample_rate, filtertype='bandpass')
                
                sd = np.std(metrics_signal)
                if sd > 0:
                    normalized_signal = (metrics_signal - np.mean(metrics_signal)) / sd
                    new_ppg_deque = deque(data_store['ppg_plot_buffer'], maxlen=PPG_PLOT_WINDOW_SIZE)
                    new_ppg_deque.extend(normalized_signal[-UNIT_WINDOW_FRAMES:])
                    data_store['ppg_plot_buffer'] = list(new_ppg_deque)

                working_data, measures = rPPG_Process.process(metrics_signal, sample_rate)

                # === 新增：指标入库（与当前计算窗口自动对齐） =============================
                # 1) 以“本次参与计算的序列长度”反推窗口秒数，避免硬编码 60s
                try:
                    _win_len_sec = float(len(metrics_signal)) / float(sample_rate) if sample_rate else None
                except Exception:
                    _win_len_sec = None

                # 2) 时间戳对齐：以“当前时刻”为 window_end，回溯窗口秒数得到 window_start
                now_utc = datetime.now(timezone.utc)
                window_end = now_utc.isoformat()
                if _win_len_sec and _win_len_sec > 0:
                    window_start_dt = now_utc - timedelta(seconds=_win_len_sec)
                    window_length = float(_win_len_sec)
                else:
                    # 回退：若无法计算长度，保持与你现有设置一致（例如已有 WINDOW_SECONDS 变量）
                    # 如果你的代码没有这个变量，保留 60 只作为兜底；不会影响已有逻辑
                    WINDOW_SECONDS = globals().get("WINDOW_SECONDS", 60)
                    window_start_dt = now_utc - timedelta(seconds=WINDOW_SECONDS)
                    window_length = float(WINDOW_SECONDS)

                window_start = window_start_dt.isoformat()

                # 3) 采样规范（写入 sampling_spec 便于复现）
                # - 若你已有滑窗重叠变量（如 overlap_ratio / hop_size），可替换下面的 0.5 为你的变量
                _sampling_overlap = globals().get("WINDOW_OVERLAP", 0.5)
                sampling_spec = f"len={window_length:.0f}s;overlap={int(_sampling_overlap*100)}%;sr={float(sample_rate):.2f}Hz"

                # 4) 派生 SD1/SD2 比值（若未返回）
                sd1 = measures.get("sd1"); sd2 = measures.get("sd2")
                sd1_sd2_ratio = measures.get("sd1_sd2_ratio")
                if sd1_sd2_ratio is None and sd1 is not None and sd2 not in (None, 0):
                    sd1_sd2_ratio = sd1 / sd2

                # 5) 组织入库行（字段名与 storage.init_db 的列一致）
                row = {
                    "patient_id":       str(globals().get("user_id", "demo")),  # 如有用户ID上下文可替换
                    "window_start":     window_start,
                    "window_end":       window_end,
                    "window_length":    window_length,
                    "env_label":        measures.get("env_label", globals().get("ENV_LABEL_DEFAULT", "rest")),

                    "rmssd":            measures.get("rmssd"),
                    "sdnn":             measures.get("sdnn"),
                    "pnn50":            measures.get("pnn50"),
                    "sd1":              sd1,
                    "sd2":              sd2,
                    "sd1_sd2_ratio":    sd1_sd2_ratio,

                    "lf":               measures.get("lf"),
                    "hf":               measures.get("hf"),
                    "lf_hf_ratio":      measures.get("lf_hf_ratio"),

                    "hti":              measures.get("hti"),
                    "rhythm_status":    measures.get("rhythm_status"),

                    "pwv":              measures.get("pwv"),
                    "aix":              measures.get("aix"),
                    "vascular_status":  measures.get("vascular_status"),

                    "stress_index":     measures.get("stress_index"),
                    "stress_level":     measures.get("stress_level"),
                    "anxiety_score":    measures.get("anxiety_score"),
                    "anxiety_level":    measures.get("anxiety_level"),

                    "sampling_spec":    sampling_spec,
                    "notes":            measures.get("notes"),
                }

                # 6) 幂等入库 + 哈希摘要
                rec = upsert_hrv_feature(db_conn, row)                 # 返回 {id, feature_key}
                insert_feature_hash(db_conn, rec["feature_key"], row)  # 记录 SHA-256 与 LSH 桶
                # ======================================================================

                history_rows = []
                try:
                    from query_aggregator import create_connection, query_hrv_rows
                    _conn = create_connection(DB_PATH)
                    hist = query_hrv_rows(
                        _conn,
                        patient_id=row["patient_id"],
                        start_time=(window_start_dt - timedelta(minutes=30)).isoformat(),
                        end_time=window_end
                    )
                    history_rows = sorted(hist, key=lambda x: x.get("window_start",""), reverse=True)
                except Exception:
                    history_rows = []

                # === 5) 组装 LLM 输入并调用 API ===
                llm_input = build_llm_input_payload(
                    interaction_id=str(uuid.uuid4()),
                    patient_id=row["patient_id"],
                    now_iso=window_end,
                    measures={**row, "feature_key": rec["feature_key"]},
                    symptom_description=globals().get("SYMPTOM_DESCRIPTION", "未提供主诉"),
                    response_style=LLM_RESPONSE_STYLE_DEFAULT,
                    input_language=LLM_INPUT_LANGUAGE_DEFAULT,
                    output_language=LLM_OUTPUT_LANGUAGE_DEFAULT,
                    return_explanations=True,
                    history_rows=history_rows,
                    signal_quality=measures.get("sqi")
                )

                llm_output = generate_structured_response(llm_input)

                # === 6) （可选）保存 JSON / 展示到 UI ===
                os.makedirs("out", exist_ok=True)
                with open("out/llm_input_latest.json", "w", encoding="utf-8") as f:
                    f.write(pretty(llm_input))
                with open("out/llm_output_latest.json", "w", encoding="utf-8") as f:
                    f.write(pretty(llm_output))


                # [核心逻辑] 只有当计算出的BPM有效时，才更新data_store
                if measures.get('BPM', 0) > 30:
                    time_domain_measures = {k: v for k, v in measures.items() if k not in ['vlf', 'lf', 'hf', 'lf_hf_ratio', 'total_power']}
                    data_store['measures'].update(time_domain_measures)
                    if 'RR_list_cor' in working_data and working_data['RR_list_cor']:
                        data_store['rr_list'] = working_data['RR_list_cor']
                        hrv_rr_buffer.extend(working_data['RR_list_cor'])
                    if 'poincare' in working_data and working_data['poincare'].get('x_plus') is not None:
                        data_store['poincare_data'] = (working_data['poincare']['x_plus'], working_data['poincare']['x_minus'])
                    
                    # [HRV修复] 只有当长缓冲区中的数据足够时，才运行频域分析
                    if len(hrv_rr_buffer) >= 50:
                        print(f"Running frequency analysis with {len(hrv_rr_buffer)} RR intervals...")
                        freq_metrics = rPPG_Process.calc_freq_domain(list(hrv_rr_buffer))
                        data_store['measures'].update(freq_metrics)
                        data_store['freq_values'] = [freq_metrics.get('lf', 0), freq_metrics.get('hf', 0), freq_metrics.get('lf_hf_ratio', 0)]

        # --- 心跳动画逻辑 ---
        # 代码段解释: 此代码块根据当前有效的心率（BPM）来控制UI中心形图标的放大和缩小，以模拟心跳。
        beat_interval = 60.0 / last_valid_bpm # 根据最新的有效BPM计算心跳间隔
        if time.time() - last_beat_time >= beat_interval: # 如果距离上次心跳的时间超过一个心跳间隔
            last_beat_time = time.time() # 更新上次心跳动画的时间
            target_font_size = 50  # 将目标字体大小设为放大状态
        
        beat_font_size += (target_font_size - beat_font_size) * 0.25 # 平滑地改变当前字体大小以接近目标
        if abs(beat_font_size - target_font_size) < 1: # 如果已经接近目标大小
            target_font_size = 30 # 将目标设回缩小状态
        
        # --- 更新UI ---
        # 调用dashboard的update方法，传入所有最新数据，该方法内部会处理重绘
        dashboard.update(data_store, beat_font_size)

    # --- 清理 ---
    # 代码段解释: 当主循环结束后，此代码块负责安全地释放所有资源，如关闭摄像头和销毁所有窗口。
    cap.release() # 释放摄像头
    cv2.destroyAllWindows() # 关闭所有OpenCV窗口
    plt.ioff() # 关闭Matplotlib的交互模式
    print("Demo finished.")

# --- 程序入口 ---
# 代码段解释: 这是Python脚本的标准入口点。当该文件被直接运行时，`if __name__ == '__main__':`
# 后面的代码块将被执行，从而启动整个程序。
if __name__ == '__main__':
    main()
