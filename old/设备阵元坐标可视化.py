import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
# 请根据实际情况修改文件路径和文件名
file_path = '/old/4-MEMS坐标.xlsx'  # 例如：'data.xlsx'
df = pd.read_excel(file_path)

# 假设Excel中有两个列名为 'x' 和 'y'，存储坐标点
x = df['x']
y = df['y']

# 绘制坐标点
plt.figure(figsize=(6, 6))
plt.scatter(x, y, color='blue', marker='o', label='Coordinates')

# 设置图表标题和标签
plt.title('Scatter Plot of Coordinates')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
