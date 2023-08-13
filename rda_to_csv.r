# 首先安装所需的包
install.packages("readr")

# 加载包
library(readr)
# 读取rds文件
data <- readRDS("data/baron_human_samp.rds")

# 将数据写入csv文件
write.table(data, file="csvdata/baron_human_samp.csv", sep=",", row.names = FALSE)
# 读取rds文件
data <- readRDS("data/chen_samp.rds")

# 将数据写入csv文件
write.table(data, file="csvdata/chen_samp.csv", sep=",", row.names = FALSE)
# 读取rds文件
data <- readRDS("data/hrvatin.rds")

# 将数据写入csv文件
write.table(data, file="csvdata/hrvatin.csv", sep=",", row.names = FALSE)
# 读取rds文件
data <- readRDS("data/manno_human_samp.rds")

# 将数据写入csv文件
write.table(data, file="csvdata/manno_human_samp.csv", sep=",", row.names = FALSE)
# 读取rds文件
data <- readRDS("data/melanoma_dropseq.rds")

# 将数据写入csv文件
write.table(data, file="csvdata/melanoma_dropseq.csv", sep=",", row.names = FALSE)

# 读取rds文件
data <- readRDS("data/zeisel_samp.rds")

# 将数据写入csv文件
write.table(data, file="csvdata/zeisel_samp.csv", sep=",", row.names = FALSE)
