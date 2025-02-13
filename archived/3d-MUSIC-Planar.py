import numpy as np
import matplotlib.pyplot as plt

# MUSIC for Uniform Planar Array (UPA)  平面阵
derad = np.pi / 180  # 角度->弧度
Nx = 4               # X 方向阵元个数
Ny = 4               # Y 方向阵元个数
N = Nx * Ny          # 总阵元个数
M = 3                # 信源数目
azimuth = np.array([30, 0, -45])  # 方位角
elevation = np.array([45, 10, -20])  # 俯仰角
snr = 10             # 信噪比
K = 512              # 快拍数

dd = 0.5             # 阵元间距
x = np.arange(0, Nx) * dd  # X 方向阵元位置
y = np.arange(0, Ny) * dd  # Y 方向阵元位置
X, Y = np.meshgrid(x, y)  # 生成网格
array_positions = np.vstack((X.ravel(), Y.ravel(), np.zeros(N)))  # 阵元位置 (x, y, z)

# 方向矢量计算
A = np.zeros((N, M), dtype=complex)
for i in range(M):
    az = azimuth[i] * derad
    el = elevation[i] * derad
    kx = np.sin(el) * np.cos(az)
    ky = np.sin(el) * np.sin(az)
    A[:, i] = np.exp(-1j * 2 * np.pi * (kx * array_positions[0] + ky * array_positions[1]))

# 构建信号模型
S = np.random.randn(M, K)  # 信源信号，入射信号
X = A @ S  # 构造接收信号
X1 = X + np.random.randn(*X.shape) * 10 ** (-snr / 20)  # 将白色高斯噪声添加到信号中

# 计算协方差矩阵
Rxx = X1 @ X1.conj().T / K

# 特征值分解
EVA, EV = np.linalg.eig(Rxx)  # 特征值分解
EVA = np.real(EVA)  # 取实部
idx = np.argsort(EVA)  # 将特征值排序 从小到大
EV = EV[:, idx[::-1]]  # 对应特征矢量排序

# 遍历每个角度，计算空间谱
azimuth_range = np.linspace(-90, 90, 181)  # 方位角范围
elevation_range = np.linspace(-90, 90, 181)  # 俯仰角范围
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

# 找到空间谱的峰值（估计的角度）
peak_indices = np.unravel_index(np.argsort(Pmusic, axis=None)[-M:], Pmusic.shape)
estimated_azimuth = azimuth_range[peak_indices[0]]
estimated_elevation = elevation_range[peak_indices[1]]
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
# fig = plt.figure(figsize=(8, 8))
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
ax.set_title('阵元位置和角度方向 (3D)')

plt.tight_layout()
plt.show()