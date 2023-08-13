import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Create the output folder
os.makedirs("/root/SDCA-imputation/suboutput", exist_ok=True)

# 读取单细胞矩阵文件
data = pd.read_csv('/root/SDCA-imputation/csvdata/zeisel_samp.csv')

# 将矩阵的第一行去除
data = data.iloc[1:]

# 检查数据矩阵的行数是否足够
if data.shape[0] < 2:
    raise ValueError("数据矩阵的行数不足，无法使用PCA进行降维！")

# 使用PCA进行降维
pca = PCA(n_components=min(2, data.shape[0]))
reduced_data = pca.fit_transform(data)

# 找到缺失值的索引
missing_indices = np.isnan(reduced_data).any(axis=1)

# 将已知值和缺失值分开
known_data = reduced_data[~missing_indices]
missing_data = reduced_data[missing_indices]

# 训练线性回归模型，得到权重向量
features = known_data[:, :-1]  # 获取特征
target = known_data[:, -1]  # 获取目标
model = LinearRegression()
model.fit(features, target)
weights = model.coef_

# 使用子空间回归对缺失值进行插补
imputed_missing_data = np.dot(missing_data[:, :-1], weights) + model.intercept_

# 将插补后的数据重新映射回高纬空间
imputed_data = np.empty((data.shape[0], reduced_data.shape[1]))
imputed_data.fill(np.nan)
imputed_data[~missing_indices] = known_data
imputed_data[missing_indices] = np.hstack((imputed_missing_data, np.nan))

# 输出插补后的矩阵
imputed_data = pca.inverse_transform(imputed_data)
imputed_df = pd.DataFrame(imputed_data, columns=data.columns)
imputed_df.to_csv('/root/SDCA-imputation/suboutput/f_out.csv', index=False)
print("插补完成！")

