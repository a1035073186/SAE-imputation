import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# 加载原始数据和插补后的数据
df_original = pd.read_csv('/root/SDCA-imputation/suboutput/b_out.csv')
df_imputed = pd.read_csv('/root/SDCA-imputation/output/b_out_dca_out.csv')

# 提取数据
X_original = df_original.values
X_imputed = df_imputed.values

# 计算均方根误差和相关系数
rmse = np.sqrt(mean_squared_error(X_original, X_imputed))
r2 = r2_score(X_original, X_imputed)
print('RMSE:', rmse)
print('R2 score:', r2)