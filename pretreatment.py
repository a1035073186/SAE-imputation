import os
import pandas as pd
import rpy2.robjects as robjects

# 创建文件夹
os.makedirs("csvdata", exist_ok=True)

# 获取data文件夹中所有的rds文件
rds_files = [file for file in os.listdir("data") if file.endswith(".rds")]

# 遍历每个rds文件，进行转换为csv
for rds_file in rds_files:
    # 构造rds文件的路径和csv文件的路径
    rds_path = os.path.join("data", rds_file)
    csv_filename = os.path.splitext(rds_file)[0] + ".csv"
    csv_path = os.path.join("csvdata", csv_filename)
    
    # 将rds文件读取为DataFrame
    rdata = robjects.r['readRDS'](rds_path)
    
    # 将DataFrame保存为csv文件
    data.to_csv(csv_path, index=False)

# 打印转换完成的信息
print("转换完成！")