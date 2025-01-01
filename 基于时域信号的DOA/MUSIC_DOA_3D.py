import scipy.linalg as la
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import generate_data_3D

# 1. 加载数据
def load_audio_data(file_paths):
    signals = []
    for file_path in file_paths:
        signal, sr = torchaudio.load(file_path)  # 保持原始采样率
        for i in range(32):
            signals.append(signal[i][:192000*2].tolist())
    return np.array(signals)


# 2. 计算协方差矩阵
def compute_covariance_matrix(signals):
    # 假设signals是一个(通道数 x 时间点)的数组
    return np.cov(signals)


# 3. 特征分解，得到噪声子空间
def compute_noise_subspace(R, num_sources):
    eigvals, eigvecs = la.eigh(R)  # 升序排序特征值和特征向量
    noise_subspace = eigvecs[:, :R.shape[0] - num_sources]  # 获取噪声子空间
    return noise_subspace


# 4. MUSIC算法
def music_spectrum(R, noise_subspace, theta_range, array_geometry):
    P_music = np.zeros(len(theta_range))
    for idx, theta in enumerate(theta_range):
        steering_vector = get_steering_vector(theta, array_geometry)
        P_music[idx] = 1 / np.abs(steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector)
    return P_music

def get_steering_vector(theta, array_geometry):
    """
    计算阵列响应向量（Steering Vector）

    参数:
    theta: 声源方向（角度，单位为度）。
    array_geometry: 阵列几何形状 (num_mics, 3) 的数组，表示每个麦克风的位置。

    返回:
    阵列响应向量，形状为 (num_mics, )
    """
    num_mics = array_geometry.shape[0] # 麦克风数目
    # 转换角度为弧度，并假设平面波传播方向为二维 (x, y)
    theta_rad = np.deg2rad(theta)
    direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0]) # 假设波从 z=0 平面传播

    # 声波波数
    frequency = 1000 # 假设频率为 1000 Hz，可调整
    speed_of_sound = 343 # 声速为 343 m/s
    k = 2 * np.pi * frequency / speed_of_sound # 波数

    # 阵列响应计算
    steering_vector = np.exp(1j * k * (array_geometry @ direction))
    return steering_vector


def create_spiral_array_geometry(num_mics, radius=0.1, pitch=0.05):
    """
    创建一个多臂螺旋麦克风阵列的几何模型。

    参数:
        num_mics: 麦克风的总数。
        radius: 螺旋的半径（单位：米）。
        pitch: 螺旋的每圈高度（单位：米）。

    返回:
        阵列几何 (num_mics, 3) 的数组，每行表示一个麦克风的位置。
    """
    angles = np.linspace(0, 2 * np.pi * (num_mics // 8), num_mics)  # 分布在多圈上
    z_positions = pitch * angles / (2 * np.pi) * 0  # 螺旋轴上的位置
    x_positions = radius * np.cos(angles)
    y_positions = radius * np.sin(angles)
    array_geometry = np.vstack((x_positions, y_positions, z_positions)).T  # 合并为 (num_mics, 3)
    return array_geometry


# 6. 声源定位
def localize_sources(P_music, theta_range):
    peaks = []
    for i in range(1, len(P_music) - 1):
        if P_music[i] > P_music[i - 1] and P_music[i] > P_music[i + 1]:  # 简单的峰值检测
            peaks.append((theta_range[i], P_music[i]))
    return peaks


# 7. 可视化MUSIC谱和定位结果
def plot_music_spectrum(P_music, theta_range, peaks):
    plt.plot(theta_range, 10 * np.log10(P_music))
    plt.xlabel("Angle (degrees)")
    plt.ylabel("MUSIC Spectrum (dB)")
    plt.title("MUSIC Spectrum for Source Localization")
    plt.grid(True)

    # 绘制峰值
    for peak in peaks:
        plt.scatter(peak[0], 10 * np.log10(peak[1]), color='red', zorder=5)

    plt.show()


# 主函数
def main():
    # 1. 加载数据
    #! 从文件中加载数据
    # file_paths = ["./data/record 10-27-58 CH01~CH32.wav", "./data/record 10-27-58 CH33~CH64.wav"]
    # num_channels = 64
    # signals = load_audio_data(file_paths)

    #! 使用仿真数据
    signals, array_geometry = generate_data_3D.create_simulate_data()

    # 2. 计算协方差矩阵
    R = compute_covariance_matrix(signals)

    # 3. 计算噪声子空间
    num_sources = 2  # 假设有2个声源
    noise_subspace = compute_noise_subspace(R, num_sources)

    # 4. MUSIC谱计算
    theta_range = np.linspace(-90, 90, 180)  # 从-90°到90°，假设声源方位角在此范围内
    # 构建阵列几何
    P_music = music_spectrum(R, noise_subspace, theta_range, array_geometry)

    # 5. 声源定位
    peaks = localize_sources(P_music, theta_range)

    # 6. 可视化
    plot_music_spectrum(P_music, theta_range, peaks)


if __name__ == "__main__":
    main()