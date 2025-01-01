import numpy as np
import matplotlib.pyplot as plt

# 可视化协方差矩阵
def plot_covariance_matrix(Rxx):
    """
    可视化协方差矩阵: 可视化的是协方差矩阵的幅值, 由于协方差矩阵的元素是复数, 因此此处指的是复数的模长0

    """
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(Rxx), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Magnitude')
    plt.title('Covariance Matrix |Rxx|')
    plt.xlabel('Microphone Index')
    plt.ylabel('Microphone Index')
    # 调整图形布局，使图形填满画布
    plt.tight_layout()
    plt.show()

# 可视化特征值分解结果
def plot_EVD_matrix(Rxx):
    """
    可视化协方差矩阵: 可视化的是协方差矩阵的幅值, 由于协方差矩阵的元素是复数, 因此此处指的是复数的模长0

    """
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(Rxx), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Magnitude')
    plt.title('EVD Matrix |Rxx|')
    plt.xlabel('Index')
    plt.ylabel('Index')
    # 调整图形布局，使图形填满画布
    plt.tight_layout()
    plt.show()

# 可视化噪声子空间结果
def plot_noise_subspace_matrix(Rxx):
    """
    可视化协方差矩阵: 可视化的是协方差矩阵的幅值, 由于协方差矩阵的元素是复数, 因此此处指的是复数的模长0

    """
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(Rxx), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Magnitude')
    plt.title('noise subspace Matrix |Rxx|')
    plt.xlabel('Index')
    plt.ylabel('Index')
    # 调整图形布局，使图形填满画布
    plt.tight_layout()
    plt.show()
