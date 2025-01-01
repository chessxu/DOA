import numpy as np
import matplotlib.pyplot as plt
import wave
import pandas as pd
import os

from utils import *

# 1. 设置螺旋阵列几何参数 (仅x, y坐标)
def read_spiral_array(file_path):
    # file_path = 'D:\\project\DOA\\4-MEMS坐标.xlsx'  # 例如：'data.xlsx'
    df = pd.read_excel(file_path)

    # 假设Excel中有两个列名为 'x' 和 'y'，存储坐标点
    x = df['x']
    y = df['y']

    x = np.array(x)
    y = np.array(y)
    return np.vstack((x, y)).T  # 返回每个麦克风的 (x, y) 坐标


# 2. 读取.wav文件数据
def load_wav_files(input_dir, num_mics):
    """加载每个通道的.wav数据"""
    signals = np.zeros((num_mics, 0))

    filename = "D:\\project\\DOA\data\\record 09-33-39 CH01~CH32.wav"
    with wave.open(filename, 'r') as wf:
        pass

    for i in range(num_mics):
        filename = os.path.join(input_dir, f"channel_{i + 1}.wav")
        with wave.open(filename, 'r') as wf:
            fs = wf.getframerate()
            num_samples = wf.getnframes()
            signal = np.frombuffer(wf.readframes(num_samples), dtype=np.int16)
            signals = np.vstack((signals, signal)) if signals.shape[1] == 0 else np.hstack(
                (signals, signal.reshape(1, -1)))
    return signals, fs

# 参数设置
derad = np.pi / 180  # 角度 -> 弧度
N = 64                # 阵元个数
M = 1                # 信源数目
theta = np.array([-30, 0, 60])  # 待估计角度
snr = 10             # 信噪比
K = 512              # 快拍数
dd = 0.5             # 阵元间距
d = np.arange(0, N * dd, dd)  # 阵元位置

# 构造方向矢量
A = np.exp(-1j * 2 * np.pi * np.outer(d, np.sin(theta * derad)))

# 构建信号模型
S = np.random.randn(M, K)  # 信源信号
X = np.dot(A, S)           # 构造接收信号
noise = np.random.normal(0, 1, X.shape) + 1j * np.random.normal(0, 1, X.shape)
X1 = X + noise * 10 ** (-snr / 20)  # 加入噪声

# 计算协方差矩阵
Rxx = np.dot(X1, X1.conj().T) / K
print(Rxx[0][1])
plot_covariance_matrix(Rxx)

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(Rxx)
eigenvalues_sorted_indices = np.argsort(eigenvalues)  # 特征值从小到大排序
eigenvectors_sorted = eigenvectors[:, eigenvalues_sorted_indices]  # 对应特征向量排序
eigenvectors_sorted = np.fliplr(eigenvectors_sorted)  # 从大到小排序
# plot_EVD_matrix(eigenvectors_sorted)


# 噪声子空间
En = eigenvectors_sorted[:, M:N]
# plot_noise_subspace_matrix(En)

# 遍历角度，计算空间谱
angles = np.linspace(-90, 90, 361)
Pmusic = []
for angle in angles:
    phim = angle * derad
    a = np.exp(-1j * 2 * np.pi * d * np.sin(phim))
    a = a.reshape(-1, 1)
    Pmusic.append(1 / np.abs(a.conj().T @ En @ En.conj().T @ a))

Pmusic = np.abs(Pmusic).flatten()
Pmmax = np.max(Pmusic)
Pmusic = 10 * np.log10(Pmusic / Pmmax)  # 归一化处理

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
