import numpy as np
import matplotlib.pyplot as plt
import wave
import pandas as pd
import os
import torchaudio

from utils import *

# 参数设置
derad = np.pi / 180  # 角度 -> 弧度

# 1. 设置螺旋阵列几何参数 (仅x, y坐标)
def read_spiral_array():
    file_path = '/old/4-MEMS坐标.xlsx'  # 例如：'data.xlsx'
    df = pd.read_excel(file_path)

    # 假设Excel中有两个列名为 'x' 和 'y'，存储坐标点
    x = df['x']
    y = df['y']

    x = np.array(x)
    y = np.array(y)
    return np.vstack((x, y)).T  # 返回每个麦克风的 (x, y) 坐标


# 2. 读取.wav文件数据
def load_wav_files():
    filename = "/data/record 10-27-58 CH01~CH32.wav"
    wf, sr = torchaudio.load(filename)
    signals1 = wf.numpy()

    filename = "/data/record 10-27-58 CH33~CH64.wav"
    wf, sr = torchaudio.load(filename)
    signals2 = wf.numpy()
    signals = np.concatenate((signals1, signals2), axis=0)
    return signals, sr


# 3. 计算协方差矩阵
def compute_covariance_matrix(signals):
    # 计算协方差矩阵
    Rxx = np.dot(signals, signals.conj().T) / 512
    # Rxx = np.dot(signals[:, :512], signals[:, :512].conj().T) / 512
    return Rxx


# 4. MUSIC算法定位声源
def music_algorithm(Rxx, array_geometry):
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(Rxx)
    eigenvalues_sorted_indices = np.argsort(eigenvalues)  # 特征值从小到大排序
    eigenvectors_sorted = eigenvectors[:, eigenvalues_sorted_indices]  # 对应特征向量排序
    eigenvectors_sorted = np.fliplr(eigenvectors_sorted)  # 从大到小排序

    # 噪声子空间
    En = eigenvectors_sorted[:, 64:1]

    # 遍历角度，计算空间谱
    angles = np.linspace(-90, 90, 361)
    Pmusic = []
    for angle in angles:
        phim = angle * derad
        a = np.exp(-1j * 2 * np.pi * np.dot(array_geometry, np.array(
            [np.cos(np.radians(phim)), np.sin(np.radians(phim))])))
        a = a.reshape(-1, 1)
        Pmusic.append(1 / np.abs(a.conj().T @ En @ En.conj().T @ a))

    Pmusic = np.abs(Pmusic).flatten()
    Pmmax = np.max(Pmusic)
    Pmusic = 10 * np.log10(Pmusic / Pmmax)  # 归一化处理
    return Pmusic


# 5. 可视化MUSIC谱
def plot_music_spectrum(Pmusic):
    angles = np.linspace(-90, 90, 361)
    # 绘制空间谱
    plt.figure(figsize=(10, 6))
    plt.plot(angles, Pmusic, linewidth=2)
    plt.xlabel('入射角 (degrees)')
    plt.ylabel('空间谱 (dB)')
    plt.title('MUSIC Algorithm Spatial Spectrum')
    plt.grid(True)
    plt.xticks(np.arange(-90, 91, 30))
    # 调整图形布局，使图形填满画布
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    array_geometry = read_spiral_array()

    signals, sr = load_wav_files()

    Rxx = compute_covariance_matrix(signals)

    Pmusic = music_algorithm(Rxx, array_geometry)

    plot_music_spectrum(Pmusic)
