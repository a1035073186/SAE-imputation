# SAE-imputation
SAE-Impute: A novel method for single-cell data imputation combining subspace regression and autoencoders
# Step 1. Install the devtools package. Invoke R and then type
install.packages("devtools") 
# Step 2. Load the devtools package.
library("devtools") 
# Step 3. Install the SAE-imputation package from GitHub.
install_github("a1035073186/SAE-imputation", subdir="pkg")
# Example
Load the Goolam dataset and perform imputation
Load Goolam dataset: data('Goolam'); raw <- Goolam$data; label <- Goolam$label
Perform the imputation: imputed <- scISR(data = raw)
Result assessment
library(irlba)
library(mclust)
set.seed(1)
Filter genes that have only zeros from raw data
raw_filer <- raw[rowSums(raw != 0) > 0, ]
pca_raw <- irlba::prcomp_irlba(t(raw_filer), n = 50)$x
cluster_raw <- kmeans(pca_raw, length(unique(label)),
                      nstart = 2000, iter.max = 2000)$cluster
print(paste('ARI of clusters using raw data:', round(adjustedRandIndex(cluster_raw, label),3)))
set.seed(1)
pca_imputed <- irlba::prcomp_irlba(t(imputed), n = 50)$x
cluster_imputed <- kmeans(pca_imputed, length(unique(label)),
                          nstart = 2000, iter.max = 2000)$cluster
print(paste('ARI of clusters using imputed data:', round(adjustedRandIndex(cluster_imputed, label),3)))
# Step 4. convert the downloaded rds data into csv data to facilitate Python operation
# Step 5. put the predicted data as the weight label into the original data for autoencoder operation, and finally get the prediction result. 
