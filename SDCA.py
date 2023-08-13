import pandas as pd
import pyarrow as pa
from pyarrow import feather
import numpy as np 
from sklearn.decomposition import PCA 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import cross_val_score 
from sklearn.neural_network import MLPRegressor 
from sklearn.preprocessing import StandardScaler 
from sklearn.manifold import TSNE
import os
# 定义文件路径
data_path = './data/'
csv_path = './csvdata/'
# 定义函数：解压rds文件并保存为csv文件
def rds_to_csv(rds_path, csv_path):
    data = pa.read_rds_file(data_path)
    df = data.to_pandas()
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
#data = pd.read_rds(rds_path)
#data.to_csv(csv_path, index=False)
# 解压并保存rds文件为csv文件
if not os.path.exists(csv_path):
    os.makedirs(csv_path)
for file in os.listdir(data_path):
    if file.endswith('.rds'):
        rds_path = os.path.join(data_path, file)
        csv_name = os.path.splitext(file)[0] + '.csv'
        csv_path = os.path.join(csv_path, csv_name)
        rds_to_csv(rds_path, csv_path)
    print(f"{rds_path} has been converted to {csv_path}")
# 定义函数：PCA降维
def pca_reduction(data, n_components=50):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return pca, pca_data
# 定义函数：t-SNE降维
def tsne_reduction(data, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_data = tsne.fit_transform(data)
    return tsne, tsne_data
# 定义函数：将数据分成已知值和缺失值
def split_known_missing(data, missing_rate=0.3):
    known = data.copy()
    missing = data.copy()
    n_samples, n_features = data.shape
    missing_mask = np.random.choice([True, False], size=(n_samples, n_features), p=[missing_rate, 1-missing_rate])
    missing[missing_mask] = np.nan
    known[~missing_mask] = np.nan
    return known, missing
# 定义函数：使用线性回归方法进行插补
def linear_regression_impute(known, missing):
    lr = LinearRegression()
    lr.fit(known, missing)
    imputed = lr.predict(missing)
    return imputed
# 定义函数：使用神经网络方法进行插补
def neural_network_impute(known, missing):
    scaler = StandardScaler()
    known_scaled = scaler.fit_transform(known)
    missing_scaled = scaler.transform(missing)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam')
    mlp.fit(known_scaled, missing_scaled)
    imputed_scaled = mlp.predict(missing_scaled)
    imputed = scaler.inverse_transform(imputed_scaled)
    return imputed
# 定义函数：使用子空间回归进行插补
def subspace_regression_impute(data, missing_rate=0.3, impute_method='lr'):
    known, missing = split_known_missing(data, missing_rate)
    if impute_method == 'lr':
        imputed = linear_regression_impute(known, missing)
    elif impute_method == 'nn':
        imputed = neural_network_impute(known, missing)
    else:
        raise ValueError('Invalid impute method')
    data_imputed = data.copy()
    data_imputed[np.isnan(missing)] = imputed[np.isnan(missing)]
    return data_imputed
# 加载数据
data_list = []
for file in os.listdir(csv_path):
    if file.endswith('.csv'):
        csv_path = os.path.join(csv_path, file)
        data = pd.read_csv(csv_path)
        data_list.append(data.values)
        print(f"{csv_path} has been loaded")
# PCA降维
n_components = 50
pca_list = []
pca_data_list = []
for data in data_list:
    pca, pca_data = pca_reduction(data, n_components=n_components)
    pca_list.append(pca)
    pca_data_list.append(pca_data)
# t-SNE降维
tsne_list = []
tsne_data_list = []
for data in data_list:
    tsne, tsne_data = tsne_reduction(data, n_components=2, perplexity=30)
    tsne_list.append(tsne)
    tsne_data_list.append(tsne_data)
# 使用线性回归进行插补
impute_method = 'lr'
imputed_list = []
for data in data_list:
    imputed = subspace_regression_impute(data, missing_rate=0.3, impute_method=impute_method)
    imputed_list.append(imputed)
    print(f"Imputed data using {impute_method} method")
# 自编码器训练
ae_list = []
ae_imputed_list = []
for i, data in enumerate(imputed_list):
    ae = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam')
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    ae.fit(data_scaled, data_scaled)
    ae_list.append(ae)
    ae_imputed_scaled = ae.predict(scaler.transform(data_list[i]))
    ae_imputed = scaler.inverse_transform(ae_imputed_scaled)
    ae_imputed_list.append(ae_imputed)
    print(f"AE-imputed data using {impute_method} method")
# 评估插补效果
mse_list = []
for i, data in enumerate(data_list):
    mse = mean_squared_error(ae_imputed_list[i], data)
    mse_list.append(mse)
    print(f"Mean squared error for data {i}: {mse}")
# 输出插补后的数据for i, data in enumerate(ae_imputed_list):
    csv_name = f"{chr(ord('a')+i)}_imputed.csv"
    csv_path = os.path.join(csv_path, csv_name)
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"Imputed data for {chr(ord('a')+i)} has been saved to {csv_path}")