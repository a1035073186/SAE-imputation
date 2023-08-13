import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# 读取原始数据B.CSV
df_b = pd.read_csv('/root/SDCA-imputation/data/50data/A.csv')

# 读取预测结果A.CSV
df_a = pd.read_csv('/root/SDCA-imputation/suboutput/50suboutput/a_out.csv')

# 合并A和B数据
df_c = pd.concat([df_a, df_b], axis=1)

# 将矩阵C.CSV的第一行去除
df_c = df_c.iloc[1:]

# 将nan替换为0
df_c = df_c.fillna(0)

# 将超出float64范围的数值四舍五入到float64范围内
df_c = df_c.round(6)

# 输出到A文件夹下
df_c.to_csv('/root/SDCA-imputation/output1/A-50.CSV', index=False)

# 提取特征列并进行归一化
scaler = StandardScaler()
features = scaler.fit_transform(df_c.iloc[:, :-1])

# 构建自编码器模型
model = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
model.fit(features, features)

# 使用自编码器模型插补测试集中的缺失值
imputed_data = model.predict(features)
print("训练成功")

# 将nan替换为0
imputed_data = np.nan_to_num(imputed_data)

# 将超出float64范围的数值四舍五入到float64范围内
imputed_data = np.round(imputed_data, 6)

# 将插补后的数据保存到A文件夹下
imputed_df = pd.DataFrame(imputed_data, columns=df_c.columns[:-1])
imputed_df.to_csv('/root/SDCA-imputation/output1/imputed_data_A_50.csv', index=False)

# 读取插补后的数据
imputed_df = pd.read_csv('/root/SDCA-imputation/output1/imputed_data_A_50.csv')

# 读取未知矩阵作为输入数据
df_1 = pd.read_csv("/root/SDCA-imputation/output1/imputed_data_A_50.csv")

# 删除前一半的列数
num_columns = df_1.shape[1]
df_1 = df_1.iloc[:, num_columns//2:]

# 保存后一半列数的矩阵到D文件夹下
df_1.to_csv("/root/SDCA-imputation/output1/imputed_data_A_50_1.csv", index=False)
print("写入成功")

