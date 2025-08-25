# CardiacDoc 实时生理监控Demo 安装与使用指南

## 1. 项目简介

本项目是一个实时的生理信号监控工具，通过普通的网络摄像头（Webcam）捕捉面部视频流，利用深度学习和计算机视觉技术，非接触式地估算用户的心率（BPM）、心率变异性（HRV）以及其他相关的生理健康指标。

其核心技术包括：
- 高效、实时的人脸检测。
- 预训练的3D卷积神经网络 (`N3DED128_Enhanced`)，从面部视频中提取原始的光体积描记（rPPG）信号。
- 对提取的rPPG信号进行滤波、处理和分析，计算各项生理指标。
- 一个动态仪表盘，实时可视化PPG波形、心率、HRV指标等数据。

## 2. 文件结构说明

部署目录中包含以下核心文件：

- `realtime_demo.py`: **主程序**。运行此文件以启动应用。
- `environment1_fixed.yml`: **Conda环境依赖文件**。用于一键创建和配置运行环境。
- `weight_DLCN_H5_D128.pth.tar`: **模型权重**。预训练的深度学习模型参数。
- `weight_DLCN_H5_D128_enhance.pth.tar`: **增强版模型权重**。预训练的增强版深度学习模型参数。
- `networks.py`: 定义`N3DED128_Enhanced`等神经网络的结构。
- `rPPG_Process.py`: 包含了从rPPG信号计算心率和HRV指标的核心算法。
- `filtering.py`: 提供信号滤波和去噪的辅助函数。
- `pytorch_datasets.py`: 定义了用于模型训练的数据集类（在实时Demo中未直接使用，但为项目完整性保留）。

## 3. 安装步骤

### 步骤一：前期准备
确保您的电脑上已经安装了 **Miniconda** 或 **Anaconda**。

### 步骤二：创建并激活Conda环境
1.  打开一个终端（在Windows上是 Anaconda Prompt 或 PowerShell）。
2.  `cd` 到 `CardiacDoc_realtime_demo` 这个文件夹目录。
3.  运行以下命令，Conda将会自动读取 `environment1_fixed.yml` 文件，并创建好一个包含所有依赖的独立环境，环境名为 `rPPG_demo`。

    ```bash
    conda env create -f environment1_fixed.yml
    ```

4.  创建完成后，激活这个新环境：

    ```bash
    conda activate rPPG_demo
    ```
    成功激活后，您的终端提示符前会显示 `(rPPG_demo)`。

## 4. 如何运行

在**已激活** `rPPG_demo` 环境的终端中，确保您仍处于 `CardiacDoc_realtime_demo` 目录下，然后运行以下命令：

```bash
python realtime_demo.py
```

程序启动后，会打开两个窗口：
- **一个Matplotlib仪表盘**: 显示所有实时更新 生理指标图表。
- **一个OpenCV摄像头窗口**: 显示您的实时视频流，并用绿色框标出检测到的人脸。

## 5. 如何退出

要完全退出程序，请**点击选中OpenCV摄像头窗口**，然后按下键盘上的 **`q`** 键。


