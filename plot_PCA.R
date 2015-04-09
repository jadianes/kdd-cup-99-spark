# read clustering results
clustering <- read.csv("clustering_results.csv", header=FALSE)

# perform PCA
clustering_pca <- prcomp(clustering)

# Extract components 1 and 2
clustering_pca_1_2 <- clustering_pca$x[, 1:2]

# plot - asp = 1 gives equal scaling to x and y axes
plot(clustering_pca_1_2, col=clustering$V1, type = 'p',pch = 16, asp = 1) 
