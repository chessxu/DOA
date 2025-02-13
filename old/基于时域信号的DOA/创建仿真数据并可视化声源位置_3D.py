import numpy as np
import os
import wave
from scipy.signal import chirp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 设置螺旋阵列几何参数
def generate_spiral_array(num_mics, radius, height, turns):
    """生成螺旋麦克风阵列的几何位置"""
    theta = np.linspace(0, 2 * np.pi * turns, num_mics)
    z = np.linspace(0, height, num_mics)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.vstack((x, y, z)).T  # 返回每个麦克风的 (x, y, z) 坐标

# 2. 生成声源信号
def generate_source_signal(duration, fs, f_start, f_end):
    """生成一个线性调频信号 (chirp)"""
    t = np.linspace(0, duration, int(fs * duration))
    signal = chirp(t, f_start, duration, f_end)
    return signal

# 3. 计算阵列接收到的信号
def simulate_array_signals(array_geometry, sources, fs, duration, c=343):
    """模拟阵列接收到的多声源信号"""
    num_mics = array_geometry.shape[0]
    num_samples = int(fs * duration)
    signals = np.zeros((num_mics, num_samples))

    for source in sources:
        position, signal = source['position'], source['signal']
        distances = np.linalg.norm(array_geometry - position, axis=1)
        delays = distances / c

        for i, delay in enumerate(delays):
            delayed_signal = np.roll(signal, int(delay * fs))
            signals[i, :] += delayed_signal

    return signals

# 4. 保存多通道信号为 .wav 文件
def save_multichannel_data(signals, fs, output_dir):
    """保存每个通道的数据为单独的 .wav 文件"""
    os.makedirs(output_dir, exist_ok=True)
    num_mics = signals.shape[0]

    for i in range(num_mics):
        filename = os.path.join(output_dir, f"channel_{i + 1}.wav")
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(fs)
            wf.writeframes((signals[i, :] * 32767).astype(np.int16).tobytes())

# 5. 可视化螺旋阵列和声源位置
def visualize_sources_and_array(array_geometry, sources):
    """可视化螺旋阵列和声源的位置"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制螺旋阵列
    ax.scatter(array_geometry[:, 0], array_geometry[:, 1], array_geometry[:, 2], color='b', label='Microphone Array', s=10)

    # 绘制每个声源位置
    for source in sources:
        position = source['position']
        ax.scatter(position[0], position[1], position[2], color='r', label='Source', s=50)

    # 添加标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Spiral Microphone Array and Source Locations')

    plt.legend()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 阵列参数
    num_mics = 64
    radius = 0.1  # 螺旋阵列的半径 (m)
    height = 0.2  # 螺旋阵列的高度 (m)
    turns = 3     # 螺旋阵列的圈数

    # 声源参数
    duration = 1.0  # 信号时长 (s)
    fs = 16000      # 采样率 (Hz)
    sources = [
        {"position": np.array([1.0, 0.0, 0.5]), "signal": generate_source_signal(duration, fs, 100, 1000)},
        {"position": np.array([-0.5, 1.0, 0.3]), "signal": generate_source_signal(duration, fs, 200, 800)}
    ]

    # 生成螺旋阵列几何
    array_geometry = generate_spiral_array(num_mics, radius, height, turns)

    # 可视化阵列和声源位置
    visualize_sources_and_array(array_geometry, sources)

    # 模拟阵列接收到的信号
    signals = simulate_array_signals(array_geometry, sources, fs, duration)

    # # 保存数据
    # output_dir = "simulated_data"
    # save_multichannel_data(signals, fs, output_dir)
    #
    # print(f"模拟数据已保存到目录: {output_dir}")