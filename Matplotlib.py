import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# 加载原始数据和插补后的数据
df_original = pd.read_csv('/root/SDCA-imputation/suboutput/b_out.csv')
df_imputed = pd.read_csv('/root/SDCA-imputation/output/b_out_dca_out.csv')

# 提取数据
X_original = df_original.values
X_imputed = df_imputed.values

# 使用PCA将数据投影到二维空间中
pca = PCA(n_components=2)
X_original_pca = pca.fit_transform(X_original)
X_imputed_pca = pca.transform(X_imputed)

# 可视化原始数据和插补后的数据
fig, ax = plt.subplots()
ax.scatter(X_original_pca[:, 0], X_original_pca[:, 1], label='Original data')
ax.scatter(X_imputed_pca[:, 0], X_imputed_pca[:, 1], label='Imputed data')
ax.legend()

# 导出图片到output1文件夹
if not os.path.exists('output1'):
    os.makedirs('output1')
plt.savefig('output1/data_comparison.png')
print("运行完成")