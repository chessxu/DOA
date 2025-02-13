import numpy as np
import matplotlib.pyplot as plt

derad = np.pi / 180  # 角度->弧度

# 模拟麦克风阵列信号
def simulate_multichannel_signals(M, K, A):
    """
    M: 信源数
    K：快拍数
    A：阵列流行矩阵
    """
    # 构建信号模型
    S = np.random.randn(M, K)  # 信源信号，入射信号
    X = A @ S  # 构造接收信号
    X1 = X + np.random.randn(*X.shape) * 10 ** (-snr / 20)  # 将白色高斯噪声添加到信号中

    return X1


# MUSIC算法谱估计
def music_spectrum_3d(Rxx, azimuth_range, elevation_range):
    EVA, EV = np.linalg.eig(Rxx)  # 特征值分解
    EVA = np.real(EVA)  # 取实部
    idx = np.argsort(EVA)  # 将特征值排序 从小到大
    EV = EV[:, idx[::-1]]  # 对应特征矢量排序

    # 遍历每个角度，计算空间谱
    Pmusic = np.zeros((len(azimuth_range), len(elevation_range)))

    for i, az in enumerate(azimuth_range):
        for j, el in enumerate(elevation_range):
            az_rad = az * derad
            el_rad = el * derad
            kx = np.sin(el_rad) * np.cos(az_rad)
            ky = np.sin(el_rad) * np.sin(az_rad)
            a = np.exp(-1j * 2 * np.pi * (kx * array_positions[0] + ky * array_positions[1]))
            En = EV[:, M:N]  # 取矩阵的第M+1到N列组成噪声子空间
            Pmusic[i, j] = 1 / np.abs(a.conj().T @ En @ En.conj().T @ a)

    Pmusic = np.abs(Pmusic)
    Pmmax = np.max(Pmusic)
    Pmusic = 10 * np.log10(Pmusic / Pmmax)  # 归一化处理
    return Pmusic


# 峰值检测与估计角度
def estimate_source_angles_with_peaks(Pmusic):
    peak_indices = np.unravel_index(np.argsort(Pmusic, axis=None)[-M:], Pmusic.shape)
    estimated_azimuth = azimuth_range[peak_indices[0]]
    estimated_elevation = elevation_range[peak_indices[1]]

    return estimated_azimuth, estimated_elevation


# 主程序
if __name__ == "__main__":
    # 定义阵元信息
    Nx = 4  # X 方向阵元个数
    Ny = 4  # Y 方向阵元个数
    N = Nx + Ny - 1  # 总阵元个数（减去重复的原点阵元）
    dd = 0.5  # 阵元间距
    x = np.arange(0, Nx) * dd  # X 方向阵元位置
    y = np.arange(0, Ny) * dd  # Y 方向阵元位置

    # 定义麦克风阵列位置: L 型阵列的阵元位置
    array_positions_x = np.vstack((x, np.zeros(Nx), np.zeros(Nx)))  # X 轴阵元
    array_positions_y = np.vstack((np.zeros(Ny), y, np.zeros(Ny)))  # Y 轴阵元
    array_positions = np.hstack((array_positions_x, array_positions_y[:, 1:]))  # 合并阵元位置，去掉重复的原点

    # 定义信号源（方位角，俯仰角）
    M = 2  # 信源数目
    azimuth = np.array([45, 0])  # 方位角
    elevation = np.array([45, 10])  # 俯仰角
    snr = 10  # 信噪比
    K = 512  # 快拍数，信号长度

    # 方向矢量计算
    A = np.zeros((N, M), dtype=complex)
    for i in range(M):
        az = azimuth[i] * derad
        el = elevation[i] * derad
        kx = np.sin(el) * np.cos(az)
        ky = np.sin(el) * np.sin(az)
        A[:, i] = np.exp(-1j * 2 * np.pi * (kx * array_positions[0] + ky * array_positions[1]))

    # 构建信号模型
    signals = simulate_multichannel_signals(M, K, A)

    # 计算协方差矩阵
    Rxx = signals @ signals.conj().T / K

    # MUSIC算法谱估计
    azimuth_range = np.linspace(-90, 90, 181)  # 方位角范围
    elevation_range = np.linspace(-90, 90, 181)  # 俯仰角范围
    Pmusic = music_spectrum_3d(Rxx, azimuth_range, elevation_range)

    # 峰值检测与估计角度（估计的角度）
    estimated_azimuth, estimated_elevation = estimate_source_angles_with_peaks(Pmusic)



    print("估计的方位角：", estimated_azimuth)
    print("估计的俯仰角：", estimated_elevation)

    # 绘图：空间谱
    fig = plt.figure(figsize=(12, 6))

    # 空间谱图
    plt.subplot(1, 2, 1)
    plt.imshow(Pmusic.T, extent=[-90, 90, -90, 90], aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='空间谱 (dB)')
    plt.xlabel('方位角 (度)')
    plt.ylabel('俯仰角 (度)')
    plt.title('MUSIC 空间谱')

    # 阵元位置和角度方向图（3D）
    ax = fig.add_subplot(122, projection='3d')

    # 绘制阵元位置
    ax.scatter(array_positions[0], array_positions[1], array_positions[2], color='blue', label='阵元位置', s=10)

    # 绘制估计的角度方向
    for az, el in zip(estimated_azimuth, estimated_elevation):
        x_end = 10 * np.cos(el * derad) * np.cos(az * derad)
        y_end = 10 * np.cos(el * derad) * np.sin(az * derad)
        z_end = 10 * np.sin(el * derad)
        ax.plot([0, x_end], [0, y_end], [0, z_end], 'r--', label=f'估计角度: ({az:.1f}°, {el:.1f}°)')

    # 绘制声源位置（假设声源在远场，方向与估计角度一致）
    source_distance = 10  # 假设声源距离
    for az, el in zip(azimuth, elevation):
        x_source = source_distance * np.cos(el * derad) * np.cos(az * derad)
        y_source = source_distance * np.cos(el * derad) * np.sin(az * derad)
        z_source = source_distance * np.sin(el * derad)
        ax.scatter(x_source, y_source, z_source, color='green', marker='*', s=200, label=f'真实声源: ({az}°, {el}°)')

    ax.set_xlim([-12, 12])
    ax.set_ylim([-12, 12])
    ax.set_zlim([-12, 12])
    ax.set_xlabel('X 轴')
    ax.set_ylabel('Y 轴')
    ax.set_zlabel('Z 轴')
    ax.legend(loc='upper right')
    ax.set_title('L 型阵列位置和角度方向 (3D)')

    plt.tight_layout()
    plt.show()
