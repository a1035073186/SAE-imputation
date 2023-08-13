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
df_b = pd.read_csv('/root/SDCA-imputation/csvdata/manno_human_samp.csv')

# 读取预测结果A.CSV
df_a = pd.read_csv('/root/SDCA-imputation/suboutput/d_out.csv')

# 合并A和B数据
df_c = pd.concat([df_a, df_b], axis=1)

# 将矩阵C.CSV的第一行去除
df_c = df_c.iloc[1:]

# 将nan替换为0
df_c = df_c.fillna(0)

# 将超出float64范围的数值四舍五入到float64范围内
df_c = df_c.round(6)

# 输出到A文件夹下
df_c.to_csv('/root/SDCA-imputation/output1/D.CSV', index=False)

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
imputed_df.to_csv('/root/SDCA-imputation/output1/imputed_data_D.csv', index=False)

# 读取插补后的数据
imputed_df = pd.read_csv('/root/SDCA-imputation/output1/imputed_data_D.csv')

# 计算插补准确度（均方根误差）
mse = mean_squared_error(features, imputed_data)
rmse = mse ** 0.5
print(f"插补前后的均方误差：{mse}")
print(f"插补准确度（均方根误差）：{rmse}")

