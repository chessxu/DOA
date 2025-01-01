import numpy as np
import os
import wave
from scipy.signal import chirp


# 1. 设置平面多臂螺旋麦克风阵列几何参数
def generate_spiral_array(num_mics, radius, turns):
    """生成平面多臂螺旋麦克风阵列的几何位置"""
    theta = np.linspace(0, 2 * np.pi * turns, num_mics)  # 角度
    x = radius * np.cos(theta)  # x坐标
    y = radius * np.sin(theta)  # y坐标
    return np.vstack((x, y)).T  # 返回每个麦克风的 (x, y) 坐标


# 2. 生成线性调频信号 (chirp)
def generate_source_signal(duration, fs, f_start, f_end):
    """生成一个线性调频信号 (chirp)"""
    t = np.linspace(0, duration, int(fs * duration))

    #! 参数说明: t 是时间数组，f_start 是时间为0时的瞬时频率，duration 是参考时间，f_end 是时间duration时的瞬时频率
    signal = chirp(t, f_start, duration, f_end)  #! chirp函数 用于生成扫频余弦波形的函数
    return signal


# 3. 模拟麦克风阵列接收到的信号
def simulate_array_signals(array_geometry, sources, fs, duration, c=343):
    """模拟阵列接收到的多个声源信号"""
    num_mics = array_geometry.shape[0]
    num_samples = int(fs * duration)
    signals = np.zeros((num_mics, num_samples))

    for source in sources:
        position, signal = source['position'], source['signal']
        distances = np.linalg.norm(array_geometry - position, axis=1)  # 每个麦克风与声源的距离
        delays = distances / c  # 计算延迟

        for i, delay in enumerate(delays):
            # 根据延迟时间来模拟信号传播
            delayed_signal = np.roll(signal, int(delay * fs))
            signals[i, :] += delayed_signal

    return signals


# 4. 保存每个通道的数据为单独的 .wav 文件
def save_multichannel_data(signals, fs, output_dir):
    """保存每个麦克风通道的信号为独立的.wav文件"""
    os.makedirs(output_dir, exist_ok=True)
    num_mics = signals.shape[0]

    for i in range(num_mics):
        filename = os.path.join(output_dir, f"channel_{i + 1}.wav")
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit音频
            wf.setframerate(fs)
            wf.writeframes((signals[i, :] * 32767).astype(np.int16).tobytes())


# 主程序
def create_simulate_data(sources):
    # 阵列参数
    num_mics = 64  # 麦克风数目
    radius = 0.1  # 螺旋阵列的半径 (米)
    turns = 3  # 螺旋阵列的圈数

    # 声源参数
    duration = 1.0  # 信号持续时间 (秒)
    fs = 16000  # 采样率 (Hz)
    for item in sources:
        item["signal"] = generate_source_signal(duration, fs, 100, 1000)

    # sources = [
    #     # {"position": np.array([1.0, 0.0]), "signal": generate_source_signal(duration, fs, 100, 1000)},  # 声源1: 0度
    #     # {"position": np.array([-0.5, 1.0]), "signal": generate_source_signal(duration, fs, 200, 800)},  # 声源2: -63.43度
    #     # {"position": np.array([4.51, -9.02]), "signal": generate_source_signal(duration, fs, 200, 800)},  # 声源3: -63.43度
    #     {"position": np.array([9.90, 1.76]), "signal": generate_source_signal(duration, fs, 200, 800)},  # 声源4: 10.12度
    #     {"position": np.array([-9.90, 1.76]), "signal": generate_source_signal(duration, fs, 200, 800)},  # 声源5:
    #
    # ]

    # 生成平面螺旋阵列几何
    array_geometry = generate_spiral_array(num_mics, radius, turns)

    # 模拟阵列接收到的信号
    signals = simulate_array_signals(array_geometry, sources, fs, duration)

    return signals, array_geometry

    # # 保存数据到 .wav 文件
    # output_dir = "simulated_data"
    # save_multichannel_data(signals, fs, output_dir)
    #
    # print(f"模拟数据已保存到目录: {output_dir}")