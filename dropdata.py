import pandas as pd
import numpy as np
import os

# Create the output folder
os.makedirs("/root/SDCA-imputation/data/30data", exist_ok=True)
os.makedirs("/root/SDCA-imputation/data/40data", exist_ok=True)
os.makedirs("/root/SDCA-imputation/data/50data", exist_ok=True)

# 读取矩阵A
A = pd.read_csv("/root/SDCA-imputation/csvdata/baron_human_samp.csv")

# 随机替换30%的值为0
B = A.mask(np.random.random(A.shape) < 0.3, 0)

# 将缺失后的矩阵B保存为CSV文件
B.to_csv("/root/SDCA-imputation/data/30data/A.csv", index=False)
print("30写入成功")

# 随机替换40%的值为0
C = A.mask(np.random.random(A.shape) < 0.4, 0)

# 将缺失后的矩阵B保存为CSV文件
C.to_csv("/root/SDCA-imputation/data/40data/A.csv", index=False)
print("40写入成功")

# 随机替换50%的值为0
D = A.mask(np.random.random(A.shape) < 0.5, 0)

# 将缺失后的矩阵B保存为CSV文件
D.to_csv("/root/SDCA-imputation/data/50data/A.csv", index=False)
print("50写入成功")