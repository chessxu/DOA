import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.signal import find_peaks

# 参数设置
M = 8              # 阵列传感器数量
d = 0.5            # 传感器间距（单位：波长）
D = 6              # 信号源数量
theta = np.array([-30, -10, 0, 20, 40, 60])   # 信号源的DOA（单位：度）
SNR = 10           # 信噪比（dB）
N = 1000           # 快拍数

# 生成阵列流形矩阵
theta_rad = np.deg2rad(theta)
A = np.exp(-1j * 2 * np.pi * d * np.arange(M)[:, np.newaxis] * np.sin(theta_rad))

# 生成信号
S = (np.random.randn(D, N) + 1j * np.random.randn(D, N)) / np.sqrt(2)

# 生成噪声
noise_power = 10 ** (-SNR / 10)
noise = np.sqrt(noise_power) * (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)

# 接收信号
X = A @ S + noise

# 计算协方差矩阵
Rxx = (X @ X.conj().T) / N

# 特征值分解
eigenvalues, eigenvectors = eig(Rxx)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sorted_indices]

# 分离信号子空间和噪声子空间
Es = eigenvectors[:, :D]
En = eigenvectors[:, D:]

# 构造MUSIC谱
theta_range = np.arange(-90, 90.1, 0.1)
P_music = np.zeros_like(theta_range, dtype=complex)

for i, theta_i in enumerate(theta_range):
    a = np.exp(-1j * 2 * np.pi * d * np.arange(M) * np.sin(np.deg2rad(theta_i)))
    P_music[i] = 1 / (a.conj().T @ (En @ En.conj().T) @ a)

P_music = np.abs(P_music)

# 寻找峰值
peaks, _ = find_peaks(P_music, height=0.1 * np.max(P_music))
estimated_DOA = theta_range[peaks]

# 显示结果
plt.figure()
plt.plot(theta_range, 10 * np.log10(P_music / np.max(P_music)))
plt.xlabel('DOA (degrees)')
plt.ylabel('MUSIC Spectrum (dB)')
plt.title('MUSIC Spectrum')
plt.grid(True)
plt.scatter(estimated_DOA, np.zeros_like(estimated_DOA), color='red', label='Estimated DOA')
plt.legend()
plt.show()

print('Estimated DOA:', estimated_DOA)