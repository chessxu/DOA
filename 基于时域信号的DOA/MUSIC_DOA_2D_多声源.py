import numpy as np
import os
import wave
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import generate_data_2D

# 1. 设置螺旋阵列几何参数 (仅x, y坐标)
def generate_spiral_array(num_mics, radius, turns):
    """生成平面螺旋麦克风阵列的几何位置"""
    theta = np.linspace(0, 2 * np.pi * turns, num_mics)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.vstack((x, y)).T  # 返回每个麦克风的 (x, y) 坐标


# 2. 读取.wav文件数据
def load_wav_files(input_dir, num_mics):
    """加载每个通道的.wav数据"""
    signals = np.zeros((num_mics, 0))
    for i in range(num_mics):
        filename = os.path.join(input_dir, f"channel_{i + 1}.wav")
        with wave.open(filename, 'r') as wf:
            fs = wf.getframerate()
            num_samples = wf.getnframes()
            signal = np.frombuffer(wf.readframes(num_samples), dtype=np.int16)
            signals = np.vstack((signals, signal)) if signals.shape[1] == 0 else np.hstack(
                (signals, signal.reshape(1, -1)))
    return signals, fs


# 3. 计算协方差矩阵
def compute_covariance_matrix(signals):
    """计算信号的协方差矩阵"""
    num_mics = signals.shape[0]
    # 对信号进行归一化
    signals_centered = signals - np.mean(signals, axis=1, keepdims=True)
    R = np.dot(signals_centered, signals_centered.T.conj()) / signals.shape[1]
    # 对协方差矩阵进行正则化, 减少噪声的影响
    R += 0.01 * np.eye(num_mics)
    return R


# 4. MUSIC算法定位声源 (支持多声源)
def music_algorithm(array_geometry, covariance_matrix, num_sources, fs, angles=np.linspace(-180, 180, 360)):
    """基于MUSIC算法估计多个声源的DOA"""
    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = eigh(covariance_matrix)
    # 按特征值从小到大排序，取噪声子空间
    noise_eigenvectors = eigenvectors[:, :-num_sources]

    # 计算MUSIC谱
    spectrum = []
    for angle in angles:
        steering_vector = np.exp(1j * 2 * np.pi * np.dot(array_geometry, np.array(
            [np.cos(np.radians(angle)), np.sin(np.radians(angle))])) / fs)
        music_spectrum = 1 / np.abs(
            np.conj(steering_vector) @ noise_eigenvectors @ noise_eigenvectors.T.conj() @ steering_vector)
        spectrum.append(music_spectrum)

    spectrum = np.array(spectrum)

    # 找到MUSIC谱中的多个峰值，返回对应的角度
    peaks_indices = np.argsort(spectrum)[-num_sources:][::-1]  # 取谱值最大的几个索引
    estimated_angles = angles[peaks_indices]

    return estimated_angles, spectrum


# 5. 可视化MUSIC谱
def plot_music_spectrum(spectrum, angles, predicted_doa=None):
    """绘制MUSIC谱"""
    plt.plot(angles, 10 * np.log10(spectrum))

    # 如果有预测的DOA角度，进行可视化
    if predicted_doa is not None:
        if isinstance(predicted_doa, np.ndarray):  # 如果是多个DOA预测
            for doa in predicted_doa:
                plt.axvline(doa, color='r', linestyle='--', label=f'Predicted DOA: {doa:.2f}°', linewidth=0.5)
        else:  # 单个DOA预测
            plt.axvline(predicted_doa, color='r', linestyle='--', label=f'Predicted DOA: {predicted_doa:.2f}°')

    plt.xlabel('Angle (Degrees)')
    plt.ylabel('MUSIC Spectrum (dB)')
    plt.title('MUSIC Spectrum for DOA Estimation')
    plt.legend(loc='best')
    plt.show()


# 6. 可视化阵列和声源位置
def plot_array_and_sources(array_geometry, sources):
    """绘制麦克风阵列和声源位置"""
    # 绘制麦克风阵列
    plt.scatter(array_geometry[:, 0], array_geometry[:, 1], c='blue', label='Microphones')

    # 绘制声源位置
    for i, source in enumerate(sources):
        plt.scatter(source['position'][0], source['position'][1], c='red', marker='x', label=f'Source {i + 1}')

    # 绘制阵列中心点
    plt.scatter(0, 0, c='green', marker='o', label='Array Center')

    # 标注图例
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Microphone Array and Source Locations')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# 主程序
if __name__ == "__main__":
    fs = 16000
    # 设定声源位置 (假设为真实位置)
    sources = [
        # {"position": np.array([1.0, 0.5])},  # 声源1的位置
        {"position": np.array([-0.5, 1.0])},  # 声源2的位置
        # {"position": np.array([-0.5, 0.8])}
    ]

    # ! 使用仿真数据
    signals, array_geometry = generate_data_2D.create_simulate_data(sources)


    # 计算协方差矩阵
    covariance_matrix = compute_covariance_matrix(signals)

    # 使用MUSIC算法估计DOA
    estimated_angles, spectrum = music_algorithm(array_geometry, covariance_matrix, len(sources), fs)

    # 打印估计的DOA
    print(f"估计的声源方向角 (DOA): {estimated_angles}")

    # 可视化阵列和声源位置
    plot_array_and_sources(array_geometry, sources)

    # 可视化MUSIC谱
    plot_music_spectrum(spectrum, np.linspace(-180, 180, 360), estimated_angles)