import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 原始数据和插补后的数据路径
original_data_path = 'csvdata/original.csv'
imputed_data_path = 'csvdata/imputed.csv'

# 加载原始数据和插补后的数据
original_data = np.genfromtxt(original_data_path, delimiter=',')
imputed_data = np.genfromtxt(imputed_data_path, delimiter=',')

# 划分训练集和测试集
train_data, test_data, train_imputed, test_imputed = train_test_split(original_data, imputed_data, test_size=0.2, random_state=42)

# 数据归一化
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data)
test_data_normalized = scaler.transform(test_data)

# 自编码器模型
input_dim = train_data_normalized.shape[1]
hidden_dim = 32

input_data = Input(shape=(input_dim,))
encoded = Dense(hidden_dim, activation='relu')(input_data)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_data, decoded)

# 编译自编码器模型
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练自编码器模型
epochs = 100
batch_size = 32

autoencoder.fit(train_imputed, train_data_normalized, epochs=epochs, batch_size=batch_size, validation_data=(test_imputed, test_data_normalized))

# 使用自编码器进行插补
imputed_values = autoencoder.predict(imputed_data)

# 插补效果评估
mse = np.mean(np.power(original_data - imputed_values, 2), axis=0)
rmse = np.sqrt(mse)

print('各特征的均方误差（MSE）：')
for i, error in enumerate(mse):
    print(f'特征{i+1}: {error}')

print('各特征的均方根误差（RMSE）：')
for i, error in enumerate(rmse):
    print(f'特征{i+1}: {error}')