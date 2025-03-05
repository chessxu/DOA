import math
import numpy as np
import matplotlib.pyplot as plt
# #新增加的两行
import matplotlib
matplotlib.rc("font", family='SimHei, STIXGeneral')

#! 完整32通道阵列坐标
array_coords = {
    "25": (-12, 0, 0), "21": (-9, 0, 0), "13": (-6, 0, 0), "-3": (-3, 0, 0),
    "0": (0, 0, 0), "1": (3, 0, 0), "9": (6, 0, 0), "17": (9, 0, 0), "26": (12, 0, 0),
    "27": (15, 0, 0), "3": (0, 3, 0), "11": (0, 6, 0), "19": (0, 9, 0), "7": (0, -3, 0),
    "15": (0, -6, 0), "23": (0, -9, 0), "29": (0, 0, 4.5), "28": (0, 0, 9), "30": (0, 0, -4.5),
    "31": (0, 0, -9), "2": (1.5 * math.sqrt(2), 1.5 * math.sqrt(2), 0), "10": (3 * math.sqrt(2), 3 * math.sqrt(2), 0),
    "18": (4.5 * math.sqrt(2), 4.5 * math.sqrt(2), 0), "4": (-1.5 * math.sqrt(2), 1.5 * math.sqrt(2), 0),
    "12": (-3 * math.sqrt(2), 3 * math.sqrt(2), 0), "20": (-4.5 * math.sqrt(2), 4.5 * math.sqrt(2), 0),
    "6": (-1.5 * math.sqrt(2), -1.5 * math.sqrt(2), 0), "14": (-3 * math.sqrt(2), -3 * math.sqrt(2), 0),
    "22": (-4.5 * math.sqrt(2), -4.5 * math.sqrt(2), 0), "8": (1.5 * math.sqrt(2), -1.5 * math.sqrt(2), 0),
    "16": (3 * math.sqrt(2), -3 * math.sqrt(2), 0), "24": (4.5 * math.sqrt(2), -4.5 * math.sqrt(2), 0)
}

def array_3D():
    # 设置传感器间距
    d_xy = 3.0  # X/Y 轴阵元间距 (cm)
    d_z = 4.5  # Z 轴阵元间距 (cm)

    # 初始化阵元坐标列表
    sensor_positions = []

    # 原点 (0,0,0)
    sensor_positions.append((0, 0, 0))

    # X 轴上的 10 个阵元（正半轴 5 个，负半轴 4 个，原点 1 个）
    for i in range(1, 6):  # 向正方向 5 个
        sensor_positions.append((i * d_xy, 0, 0))
    for i in range(1, 5):  # 向负方向 4 个
        sensor_positions.append((-i * d_xy, 0, 0))

    # Y 轴上的 7 个阵元（上下对称，各 3 个）
    for i in range(1, 4):
        sensor_positions.append((0, i * d_xy, 0))
        sensor_positions.append((0, -i * d_xy, 0))

    # 对角线方向的阵元（每个象限 3 个，沿 ±45° 方向）
    for i in range(1, 4):
        sensor_positions.append((i * d_xy / np.sqrt(2), i * d_xy / np.sqrt(2), 0))  # 第一象限
        sensor_positions.append((-i * d_xy / np.sqrt(2), i * d_xy / np.sqrt(2), 0))  # 第二象限
        sensor_positions.append((-i * d_xy / np.sqrt(2), -i * d_xy / np.sqrt(2), 0))  # 第三象限
        sensor_positions.append((i * d_xy / np.sqrt(2), -i * d_xy / np.sqrt(2), 0))  # 第四象限

    # Z 轴方向的 5 个阵元（包括原点）
    for i in range(1, 3):
        sensor_positions.append((0, 0, i * d_z))  # Z 轴正方向
        sensor_positions.append((0, 0, -i * d_z))  # Z 轴负方向

    # 转换为 NumPy 数组
    sensor_positions = np.array(sensor_positions)

    # 绘制 3D 阵列
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制阵元
    ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2],
               c="red", s=60, edgecolors="k", depthshade=True)

    # 添加坐标轴
    ax.quiver(-20, 0, 0, 40, 0, 0, color="r", arrow_length_ratio=0.1, linewidth=2)  # X 轴
    ax.quiver(0, -20, 0, 0, 40, 0, color="b", arrow_length_ratio=0.1, linewidth=2)  # Y 轴
    ax.quiver(0, 0, -15, 0, 0, 30, color="g", arrow_length_ratio=0.1, linewidth=2)  # Z 轴
    ax.text(22, 0, 0, "X", fontsize=12, color="r")
    ax.text(0, 22, 0, "Y", fontsize=12, color="b")
    ax.text(0, 0, 18, "Z", fontsize=12, color="g")

    # 设置坐标轴范围
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-15, 15)

    # 设置坐标轴标签
    ax.set_xlabel("X Axis (cm)")
    ax.set_ylabel("Y Axis (cm)")
    ax.set_zlabel("Z Axis (cm)")
    ax.set_title("3D Sensor Array Visualization")

    plt.show()


def array_xoy():
    # 传感器间距
    d = 3.0  # 3cm

    # X 轴阵元（编号 + 位置）
    x_labels = [25, 21, 13, 5, 0, 1, 9, 17, 26, 27]
    x_positions = np.arange(-4, 6) * d  # -12cm 到 12cm
    y_positions_x_axis = np.zeros_like(x_positions)

    # Y 轴阵元（编号 + 位置）
    y_labels = [7, 15, 23, 0, 3, 11, 19]  # 从负向到正向
    y_positions = np.arange(-3, 4) * d  # -9cm 到 9cm
    x_positions_y_axis = np.zeros_like(y_positions)

    # 对角线（米字形）阵元计算（圆与45°对角线交点）
    radii = [3, 6, 9]  # 半径
    diag_x = []
    diag_y = []

    # 以半径 r=3,6,9 计算 45° 交点坐标
    for r in radii:
        x_val = r / np.sqrt(2)  # 计算交点坐标
        y_val = r / np.sqrt(2)

        # 第一象限 (+x, +y)
        diag_x.append(x_val)
        diag_y.append(y_val)

        # 第二象限 (-x, +y)
        diag_x.append(-x_val)
        diag_y.append(y_val)

        # 第三象限 (-x, -y)
        diag_x.append(-x_val)
        diag_y.append(-y_val)

        # 第四象限 (+x, -y)
        diag_x.append(x_val)
        diag_y.append(-y_val)

    # 更新对角线阵元编号
    diag_labels = [
        2, 10, 18,  # 第一象限
        4, 12, 20,  # 第二象限
        6, 14, 22,  # 第三象限
        8, 16, 24  # 第四象限
    ]

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制 X 轴阵元
    ax.scatter(x_positions, y_positions_x_axis, c='red', s=80, edgecolors='k')
    for x, label in zip(x_positions, x_labels):
        if label != 0:  # 避免 0 号编号重复
            ax.text(x, 1, str(label), fontsize=12, ha='center', va='bottom', color='black')

    # 绘制 Y 轴阵元
    ax.scatter(x_positions_y_axis, y_positions, c='blue', s=80, edgecolors='k')
    for y, label in zip(y_positions, y_labels):
        if label != 0:
            ax.text(1, y, str(label), fontsize=12, ha='left', va='center', color='black')

    # 绘制对角线上的米字形阵元
    ax.scatter(diag_x, diag_y, c='green', s=80, edgecolors='k')
    for x, y, label in zip(diag_x, diag_y, diag_labels):
        ax.text(x + 1, y, str(label), fontsize=12, ha='left', va='center', color='black')

    # 绘制原点
    ax.scatter(0, 0, c='purple', s=80, edgecolors='k', label="中心阵元")
    ax.text(1, 0, "0", fontsize=12, ha='left', va='center', color='black')

    # 设置坐标轴刻度
    ticks = np.arange(-16, 17, 3)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(int(x)) for x in ticks])
    ax.set_yticklabels([str(int(y)) for y in ticks])

    # 绘制坐标轴
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    # 设定显示范围
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)

    # 坐标轴标签和标题
    ax.set_xlabel("X Axis (cm)")
    ax.set_ylabel("Y Axis (cm)")
    ax.set_title("XOY 平面阵元分布（含米字形对角线）")

    # 添加网格
    ax.grid(True, linestyle="--", alpha=0.6)

    # 显示图像
    plt.show()


def array_xoz():
    # 传感器间距
    d_x = 3.0  # X 轴阵元间距 (cm)
    d_z = 4.5  # Z 轴阵元间距 (cm)

    # X 轴阵元编号与坐标
    x_labels = [25, 21, 13, 5, 0, 1, 9, 17, 26, 27]
    x_positions = np.arange(-4, 6) * d_x  # X 轴坐标

    # Z 轴阵元编号与坐标（按要求调整顺序）
    z_labels = [31, 30, 0, 29, 28]  # 负半轴 -> 原点 -> 正半轴
    z_positions = np.array([-2, -1, 0, 1, 2]) * d_z  # Z 轴坐标

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制 X 轴上的阵元
    for x, label in zip(x_positions, x_labels):
        ax.scatter(x, 0, c='red', s=80, edgecolors='k')  # 阵元
        if label != 0:  # 仅保留一个 0 号编号
            ax.text(x, 1, str(label), fontsize=12, ha='center', va='bottom', color='black')

    # 绘制 Z 轴上的阵元
    for z, label in zip(z_positions, z_labels):
        ax.scatter(0, z, c='blue', s=80, edgecolors='k')  # 阵元
        if label != 0:  # 仅保留一个 0 号编号
            ax.text(1, z, str(label), fontsize=12, ha='left', va='center', color='black')

    # 设置坐标轴刻度（X 轴 3cm，Z 轴 4.5cm）
    x_ticks = np.arange(-15, 19, 3)  # X 轴刻度
    z_ticks = np.arange(-15, 16, 4.5)  # Z 轴刻度
    ax.set_xticks(x_ticks)
    ax.set_yticks(z_ticks)
    ax.set_xticklabels([str(int(x)) for x in x_ticks])
    ax.set_yticklabels([str(int(z)) for z in z_ticks])

    # 绘制坐标轴
    ax.axhline(0, color='black', linewidth=1)  # X 轴
    ax.axvline(0, color='black', linewidth=1)  # Z 轴

    # 设定显示范围
    ax.set_xlim(-15, 18)
    ax.set_ylim(-12, 12)

    # 坐标轴标签和标题
    ax.set_xlabel("X Axis (cm)")
    ax.set_ylabel("Z Axis (cm)")
    ax.set_title("XOZ 平面阵元分布")

    # 添加网格，增强可读性
    ax.grid(True, linestyle="--", alpha=0.6)

    # 显示图像
    plt.show()




