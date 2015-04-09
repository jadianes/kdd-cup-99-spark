# read clustering results
clustering <- read.csv("clustering_results.csv", header=FALSE)

# perform PCA
clustering_pca <- prcomp(clustering)

# Extract components 1 and 2
clustering_pca_1_2 <- clustering_pca$x[, 1:2]

# plot - asp = 1 gives equal scaling to x and y axes
library(ggplot2)
pca_df = data.frame(clustering_pca_1_2)
pca_df$cluster <- as.factor(clustering$V1)
ggplot(data=pca_df, aes(x=PC1, y=PC2, color=cluster)) +
    geom_point()
