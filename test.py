"""
    希尔伯特黄变换代码
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from PyEMD import EMD


# 生成一个示例时域信号（比如混合正弦波信号）
def generate_signal():
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    return t, signal


# 进行经验模态分解 (EMD)
def emd_decompose(signal):
    emd = EMD()
    imfs = emd(signal)
    return imfs


# 进行希尔伯特变换
def hilbert_transform(imfs):
    analytic_signals = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        analytic_signals.append(analytic_signal)
    return analytic_signals


# 绘制IMFs和对应的包络线
def plot_imfs(t, imfs, analytic_signals):
    plt.figure(figsize=(12, 8))

    for i, (imf, analytic_signal) in enumerate(zip(imfs, analytic_signals)):
        plt.subplot(len(imfs), 1, i + 1)
        plt.plot(t, imf, label=f'IMF {i + 1}')
        plt.plot(t, np.abs(analytic_signal), label=f'Envelope {i + 1}', linestyle='--')
        plt.legend()
        plt.title(f'IMF {i + 1} and its Envelope')

    plt.tight_layout()
    plt.show()


# 主程序
t, signal = generate_signal()

# 1. 对信号进行EMD分解
imfs = emd_decompose(signal)

# 2. 对每个IMF进行希尔伯特变换
analytic_signals = hilbert_transform(imfs)
print(analytic_signals)

# 3. 绘制IMFs及其包络线
plot_imfs(t, imfs, analytic_signals)
