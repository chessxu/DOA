import numpy as np
import matplotlib.pyplot as plt


# 生成模拟数据
def generate_signal(num_elements, num_snapshots, angles, snr):
    k = 2 * np.pi / 1  # 波长为 1
    d = 0.5  # 阵元间距为半波长
    theta = np.deg2rad(angles[:, 0])  # 俯仰角
    phi = np.deg2rad(angles[:, 1])  # 方位角
    num_sources = len(angles)

    # 生成信号矩阵
    A = np.exp(1j * k * d * (np.outer(np.arange(num_elements), np.sin(theta) * np.cos(phi)) +
                             np.outer(np.arange(num_elements), np.sin(theta) * np.sin(phi))))
    s = np.random.randn(num_sources, num_snapshots) + 1j * np.random.randn(num_sources, num_snapshots)
    s = s / np.sqrt(num_sources)  # 归一化
    s = s * (10 ** (snr / 20))  # 调整信噪比
    noise = np.random.randn(num_elements, num_snapshots) + 1j * np.random.randn(num_elements, num_snapshots)
    noise = noise / np.sqrt(num_elements)

    X = np.dot(A, s) + noise  # 生成阵列信号
    return X, A


# 计算协方差矩阵
def calculate_covariance_matrix(X):
    Rxx = np.dot(X, X.conj().T) / X.shape[1]
    return Rxx


# 特征分解
def eigen_decomposition(Rxx, num_sources):
    eigenvalues, eigenvectors = np.linalg.eigh(Rxx)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    U_s = eigenvectors[:, :num_sources]
    U_n = eigenvectors[:, num_sources:]

    return U_s, U_n


# 计算 3D MUSIC 谱
def music_spectrum(U_n, num_elements, num_points=180):
    k = 2 * np.pi / 1  # 波长为 1
    d = 0.5  # 阵元间距为半波长
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)  # 俯仰角范围
    phi = np.linspace(-np.pi, np.pi, num_points)  # 方位角范围
    P_music = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            a = np.exp(1j * k * d * (np.arange(num_elements) * np.sin(theta[i]) * np.cos(phi[j]) +
                                     np.arange(num_elements) * np.sin(theta[i]) * np.sin(phi[j])))
            P_music[i, j] = 1 / (a.conj().dot(U_n).dot(U_n.conj().T).dot(a))

    return theta, phi, P_music


# 绘制结果
def plot_results(theta, phi, P_music, angles):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(10 * np.log10(P_music), extent=[-180, 180, -90, 90], aspect='auto', cmap='jet')
    plt.colorbar(label='Power (dB)')
    plt.title('3D MUSIC DOA Estimation')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Elevation Angle (degrees)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(angles[:, 1], angles[:, 0], 'ro')
    plt.title('True DOA Angles')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Elevation Angle (degrees)')
    plt.grid(True)
    plt.show()


# 参数设置
num_elements = 8  # 阵元数量
num_snapshots = 1000  # 快照数量
angles = np.array([[30, 60], [45, -30]])  # 信号角度 (俯仰角, 方位角)
snr = 10  # 信噪比

# 生成信号
X, A = generate_signal(num_elements, num_snapshots, angles, snr)

# 计算协方差矩阵
Rxx = calculate_covariance_matrix(X)

# 特征分解
num_sources = len(angles)
U_s, U_n = eigen_decomposition(Rxx, num_sources)

# 计算 3D MUSIC 谱
theta, phi, P_music = music_spectrum(U_n, num_elements)

# 绘制结果
plot_results(theta, phi, P_music, angles)
