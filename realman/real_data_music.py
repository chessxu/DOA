import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import torch
import torchaudio
# #新增加的两行
import matplotlib
matplotlib.rc("font", family='SimHei, STIXGeneral')

def generate_circular_array(num_channels, radius):
    """
    生成环形阵列的位置坐标
    :param num_channels: 阵元数量
    :param radius: 圆的半径（单位：cm）
    :return: 阵元的位置坐标数组，形状为 (num_channels, 2)
    """
    # 计算每个阵元的角度间隔（弧度制）
    angles = np.linspace(0, 2 * np.pi, num_channels, endpoint=False)

    # 计算每个阵元的x和y坐标
    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)

    # 将x和y坐标组合成一个数组
    positions = np.column_stack((x_coords, y_coords))

    return positions


def generate_steering_matrix(array_positions, frequencies, directions, speed_of_sound=34300):
    """
    生成阵列流行矩阵
    :param array_positions: 阵元位置坐标，形状为 (num_channels, 2)
    :param frequencies: 信号频率列表（单位：Hz）
    :param directions: 信号方向列表（单位：度）
    :param speed_of_sound: 声速（单位：cm/s，默认为34300 cm/s）
    :return: 阵列流行矩阵，形状为 (num_channels, num_frequencies * num_directions)
    """
    num_channels = array_positions.shape[0]
    num_frequencies = len(frequencies)
    num_directions = len(directions)

    # 将方向从度转换为弧度
    directions_rad = np.deg2rad(directions)

    # 初始化阵列流行矩阵
    steering_matrix = np.zeros((num_channels, num_frequencies * num_directions), dtype=complex)

    # 计算每个频率和方向的阵列响应向量
    for freq_idx, freq in enumerate(frequencies):
        wavelength = speed_of_sound / freq  # 波长（单位：cm）
        for dir_idx, direction in enumerate(directions_rad):
            # 计算波数向量
            k = 2 * np.pi / wavelength * np.array([np.cos(direction), np.sin(direction)])

            # 计算每个阵元的相位延迟
            phase_delays = np.exp(-1j * (array_positions @ k))

            # 将响应向量存入流行矩阵
            steering_matrix[:, freq_idx * num_directions + dir_idx] = phase_delays

    return steering_matrix


# MUSIC算法
def music_algorithm(array_positions, received_signal, frequencies, search_grid, speed_of_sound=34300):
    num_channels = array_positions.shape[0]
    num_snapshots = received_signal.shape[1]

    # 计算协方差矩阵
    covariance_matrix = (received_signal @ received_signal.conj().T) / num_snapshots

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 排序特征值和特征向量
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 估计信号子空间和噪声子空间
    num_signals = 1  # 假设只有一个信号
    noise_subspace = eigenvectors[:, num_signals:]

    # 构建空间谱
    spectrum = np.zeros(len(search_grid))
    for idx, direction in enumerate(search_grid):
        steering_vector = generate_steering_matrix(array_positions, frequencies, [direction], speed_of_sound)
        steering_vector = steering_vector.flatten()
        spectrum[idx] = 1 / np.linalg.norm(noise_subspace.conj().T @ steering_vector) ** 2

    return spectrum

# 参数设置
num_channels = 8  # 阵元数量
radius = 3  # 半径，单位：cm
frequencies = [1000]  # 信号频率（单位：Hz）
search_grid = np.linspace(0, 360, 361)  # 搜索方向网格（单位：度）
speed_of_sound = 34300  # 声速，单位：cm/s

sampling_rate = 8000  # 采样率，单位：Hz
num_samples = 1024  # 每个阵元的采样点数

# 生成阵列
array_positions = generate_circular_array(num_channels, radius)

# 加载真实时域数据（假设已经加载为 shape=(num_channels, num_samples)）
# 生成示例信号
true_direction = 181.8957  # 真实信号方向（单位：度）
# t = np.linspace(0, num_samples / sampling_rate, num_samples, endpoint=False)
# signal = np.sin(2 * np.pi * frequencies[0] * t)  # 示例信号
#
# # 计算波数向量
# wavelength = speed_of_sound / frequencies[0]  # 波长
# k = 2 * np.pi / wavelength * np.array([np.cos(np.deg2rad(true_direction)), np.sin(np.deg2rad(true_direction))])
#
# # 计算每个阵元的相位延迟
# phase_delays = np.exp(-1j * (array_positions @ k))
#
# # 生成时域信号矩阵
# received_signal_time = phase_delays.reshape(-1, 1) * signal

received_signal_time = []
root_filename = "/home/ubuntu/project/DOA/手撕代码/data/Auditorium/static/P0011/TRAIN_S_AUDI_P0011_P0011W0003_CH"
for i in range(num_channels):
    filename = root_filename + str(i) + ".flac"
    waveform, sr = torchaudio.load(filename)
    received_signal_time.append(waveform)
received_signal_time = torch.stack(received_signal_time, dim=0)
received_signal_time = received_signal_time.squeeze(1)[:, :1024].numpy()


# 时域数据预处理：FFT转换到频域
received_signal_freq = fft(received_signal_time, axis=1)
freq_bins = np.fft.fftfreq(num_samples, 1 / sampling_rate)
freq_idx = np.argmin(np.abs(freq_bins - frequencies[0]))  # 选择信号频率对应的频点
received_signal = received_signal_freq[:, freq_idx].reshape(-1, 1)  # 提取频域数据

# 使用MUSIC算法进行DOA估计
spectrum = music_algorithm(array_positions, received_signal, frequencies, search_grid, speed_of_sound)

# 可视化空间谱
plt.figure()
plt.plot(search_grid, 10 * np.log10(spectrum / np.max(spectrum)), label='MUSIC Spectrum')
plt.axvline(x=true_direction, color='r', linestyle='--', label='True Direction')
plt.xlabel('Direction (degrees)')
plt.ylabel('Spatial Spectrum (dB)')
plt.title('MUSIC Spectrum for DOA Estimation')
plt.legend()
plt.grid(True)
plt.show()