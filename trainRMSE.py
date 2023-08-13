import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

# 读取原始数据B.CSV
df_b = pd.read_csv('/root/SDCA-imputation/csvdata/baron_human_samp.csv')

# 读取预测结果A.CSV
df_a = pd.read_csv('/root/SDCA-imputation/suboutput/a_out.csv')

# 将缺失值替换为0
#df_a.fillna(0, inplace=True)

# 输出修改后的数据
#df_a.to_csv('/root/SDCA-imputation/suboutput/a_out_modified-1.csv', index=False)
#print("修改成功")

# 合并A和B数据
df_c = pd.concat([df_a, df_b], axis=1)

# 将矩阵C.CSV的第一行去除
df_c = df_c.iloc[1:]

# 将输入数据C分为训练集和测试集
train_data, test_data = train_test_split(df_c, test_size=0.2, random_state=42)

# 提取特征列并进行归一化
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_data.iloc[:, :-1])
test_features = scaler.transform(test_data.iloc[:, :-1])

# 构建自编码器模型
input_dim = train_features.shape[1]
hidden_dim = 64
autoencoder = Sequential([
    Dense(hidden_dim, input_dim=input_dim, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')

# 训练自编码器模型
autoencoder.fit(train_features, train_features, epochs=50, batch_size=32, shuffle=True, validation_data=(test_features, test_features))

# 使用训练好的自编码器模型进行插补
train_data.iloc[:, :-1] = autoencoder.predict(train_features)
test_data.iloc[:, :-1] = autoencoder.predict(test_features)

# 衡量插补准确度
train_rmse = np.sqrt(np.mean(np.square(train_data.iloc[:, :-1] - df_b.iloc[:, :-1])))
test_rmse = np.sqrt(np.mean(np.square(test_data.iloc[:, :-1] - df_b.iloc[:, :-1])))
print(f"训练集插补准确度（均方根误差）：{str(train_rmse)}")
print(f"测试插补准确度（均方根误差）：{str(test_rmse)}")
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")
