import numpy as np
import matplotlib.pyplot as plt
# #新增加的两行
import matplotlib
matplotlib.rc("font", family='SimHei, STIXGeneral')

# MUSIC for Uniform Linear Array
derad = np.pi / 180  # 角度->弧度
N = 8                # 阵元个数
M = 6                # 信源数目
theta = np.array([-30, -10, 0, 20, 40, 60])  # 待估计角度
snr = 10             # 信噪比
K = 512              # 快拍数

dd = 0.5             # 阵元间距, 阵元间距0.5, 默认假设波长为1
d = np.arange(0, N) * dd
# 阵元间距0.5, 默认假设波长为1, 因此此处不需要除以波长
A = np.exp(-1j * 2 * np.pi * d.reshape(-1, 1) @ np.sin(theta * derad).reshape(1, -1))  # 方向矢量, 此处本应该除以波长, 但是省略了

# 构建信号模型
S = np.random.randn(M, K)  # 信源信号，入射信号, 窄带信号, 所以信源信号可以直接是时域表达

# 生成信号
# S = (np.random.randn(M, K) + 1j * np.random.randn(M, K)) / np.sqrt(2)  # 如果不是窄带信号, 那就要这么生成源信号

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
Pmusic = np.zeros(361)
angles = np.linspace(-90, 90, 361)  # 生成角度范围
for iang, angle in enumerate(angles):
    phim = derad * angle
    a = np.exp(-1j * 2 * np.pi * d * np.sin(phim)).reshape(-1, 1)
    En = EV[:, M:N]  # 取矩阵的第M+1到N列组成噪声子空间
    Pmusic[iang] = 1 / np.abs(a.conj().T @ En @ En.conj().T @ a)

Pmusic = np.abs(Pmusic)
Pmmax = np.max(Pmusic)
Pmusic = 10 * np.log10(Pmusic / Pmmax)  # 归一化处理

# 找到空间谱的峰值（估计的角度）
peak_indices = np.argsort(Pmusic)[-M:]  # 找到最大的 M 个峰值
estimated_angles = angles[peak_indices]  # 获取对应的角度
print("估计的角度值：", np.sort(estimated_angles))  # 打印估计的角度值

# 绘图：空间谱
plt.figure(figsize=(12, 6))

# 空间谱图
plt.subplot(1, 2, 1)
plt.plot(angles, Pmusic, linewidth=2)
plt.xlabel('入射角/(degree)')
plt.ylabel('空间谱/(dB)')
plt.xticks(np.arange(-90, 91, 30))
plt.grid(True)
plt.title('MUSIC 空间谱')

# 阵元位置和角度方向图
plt.subplot(1, 2, 2)
# 绘制阵元位置
array_positions = np.vstack((d, np.zeros(N)))  # 阵元位置 (x, y)
plt.scatter(array_positions[0], array_positions[1], color='blue', label='阵元位置', s=10)

# 绘制估计的角度方向
for angle in estimated_angles:
    x_end = 10 * np.cos(angle * derad)  # 直线终点 x
    y_end = 10 * np.sin(angle * derad)  # 直线终点 y
    plt.plot([0, x_end], [0, y_end], 'r--', label=f'估计角度: {angle:.1f}°')

# 绘制声源位置（假设声源在远场，方向与估计角度一致）
source_distance = 10  # 假设声源距离
for angle in theta:
    x_source = source_distance * np.cos(angle * derad)
    y_source = source_distance * np.sin(angle * derad)
    plt.scatter(x_source, y_source, color='green', marker='*', s=200, label=f'真实声源: {angle}°')

plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend(loc='upper right')
plt.title('阵元位置和角度方向')

plt.tight_layout()
plt.show()