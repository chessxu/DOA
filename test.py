import numpy as np
import matplotlib.pyplot as plt

# 创建一个示例协方差矩阵 Rxx（随机生成）
np.random.seed(0)
A = np.random.randn(10, 10)  # 创建一个 10x10 的矩阵
Rxx = np.dot(A, A.T)  # 协方差矩阵

# 进行特征值分解
eigenvalues, eigenvectors = np.linalg.eig(Rxx)

# 1. 可视化特征值
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)  # 子图 1：特征值的条形图
plt.bar(range(len(eigenvalues)), np.sort(eigenvalues)[::-1], color='blue')
plt.title('Eigenvalues')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')

# 2. 可视化特征向量（按列）
plt.subplot(1, 2, 2)  # 子图 2：特征向量的条形图
for i in range(len(eigenvectors)):
    plt.plot(eigenvectors[:, i], label=f'Eigenvector {i+1}')
plt.title('Eigenvectors')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# 调整布局并显示
plt.tight_layout()
plt.show()
