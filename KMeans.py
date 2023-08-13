import pandas as pd
from sklearn.cluster import KMeans

# 加载插补后的数据
df_imputed = pd.read_csv('A.csv')

# 进行聚类分析
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_imputed)
labels = kmeans.labels_
print('Cluster labels:', labels)