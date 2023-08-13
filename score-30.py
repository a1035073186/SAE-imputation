import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

# 读取插补后的矩阵C和缺失值矩阵B
B_1 = pd.read_csv("/root/SDCA-imputation/output1/A-30.CSV")
B_1 = B_1.iloc[1:]
C_1 = pd.read_csv("/root/SDCA-imputation/output1/imputed_data_A_30.csv")
C_1 = C_1.iloc[1:]

C_1 = C_1.values
B_1 = B_1.values

C_1[C_1 > 0] = 1
B_1[B_1 > 0] = 1

C_1[C_1 < 0] = 0
B_1[B_1 < 0] = 0
df1 = pd.DataFrame(C_1)
df2 = pd.DataFrame(B_1)

df2.to_csv("/root/SDCA-imputation/data/30data/A1.csv", index=False)
df1.to_csv("/root/SDCA-imputation/data/30data/A2.csv", index=False)

B = pd.read_csv("/root/SDCA-imputation/data/30data/A1.csv")
C = pd.read_csv("/root/SDCA-imputation/data/30data/A2.csv")
# 读取矩阵A和矩阵B作为输入数据
C1 = pd.concat([B, C], axis=1, join='inner')

# 计算F1分数
y_true = C1.iloc[:, -1]
y_pred = C1.iloc[:, -2]
f1 = f1_score(y_true, y_pred, average='binary')

# 输出F1分数
print("F1 score:", f1)
