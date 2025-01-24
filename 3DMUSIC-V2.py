import numpy as np
import matplotlib.pyplot as plt

# 生成方向矢量
def steer_vector(fre, theta, phi, speed, numbers, space):
    k = 2 * np.pi / fre
    alphas = []
    for i in range(numbers):
        alphas.append(np.exp(-1j * k * i * space * (np.sin(theta) * np.cos(phi) + np.sin(theta) * np.sin(phi))))
    return np.array(alphas).reshape(-1, 1)

# MUSIC算法
def cal_music(fre, speed, numbers, space, signals, method='noise'):
    R_x = np.matmul(signals, np.conjugate(signals.T)) / signals.shape[1]
    lamda, u = np.linalg.eig(R_x)
    u_s = u[:, :numbers]
    u_n = u[:, numbers:]
    P = []
    thetas = np.linspace(-np.pi/2, np.pi/2, 180)
    phis = np.linspace(-np.pi, np.pi, 180)
    for _theta in thetas:
        for _phi in phis:
            _alphas = steer_vector(fre, _theta, _phi, speed, numbers, space).reshape(-1, 1)
            if method == 'signal':
                P_x = 1 / np.abs(np.matmul(np.conjugate(_alphas).T, np.eye(len(u_s)) - np.matmul(u_s, np.conjugate(u_s.T))), _alphas)
            elif method == 'noise':
                P_x = 1 / np.abs(np.matmul(np.conjugate(_alphas).T, u_n).dot(np.conjugate(u_n.T)).dot(_alphas))
            P.append(P_x)
    P = np.array(P).reshape(180, 180)
    return thetas, phis, P

# 初始化数据
fs = 20000
fre = 200
t = np.arange(0, 0.01, 1/fs)
theta1 = np.pi / 3
theta2 = 2 * np.pi / 3
speed = 340
numbers = 32
space = 1
signals = []

# 生成模拟快拍数据
for t_0 in t:
    signal1 = np.exp(2j * np.pi * fre * t_0 - 2j * np.pi * fre * np.arange(numbers) * space * np.cos(theta1) / speed)
    signal2 = np.exp(2j * np.pi * fre * t_0 - 2j * np.pi * fre * np.arange(numbers) * space * np.cos(theta2) / speed)
    signal = signal1 + signal2
    signals.append(signal.tolist())
signals = np.array(signals)

# MUSIC算法处理结果
thetas, phis, P = cal_music(fre, speed, numbers, space, signals, method='noise')

# 找到峰值位置
max_indices = np.unravel_index(np.argmax(P, axis=None), P.shape)
predicted_theta = thetas[max_indices[0]] * 180 / np.pi
predicted_phi = phis[max_indices[1]] * 180 / np.pi

# 输出预测的俯仰角和方位角
print(f"Predicted Elevation Angle: {predicted_theta:.2f} degrees")
print(f"Predicted Azimuth Angle: {predicted_phi:.2f} degrees")

# 绘制结果
plt.figure(figsize=(10, 8))
plt.imshow(10 * np.log10(P), extent=[-180, 180, -90, 90], aspect='auto', cmap='jet')
plt.colorbar(label='Power (dB)')
plt.title('3D MUSIC DOA Estimation')
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Elevation Angle (degrees)')
plt.grid(True)
plt.show()