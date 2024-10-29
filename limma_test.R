library(Seurat)
library(SeuratData)
library(SeuratWrappers)
library(Azimuth)
library(ggplot2)
library(patchwork)
library(Matrix)

library(scater)
library(cowplot)
devtools::install_github("zhangyuqing/sva-devel")
library(sva)

install.packages('edgeR')
library('edgeR')


count_matrix = as.matrix(read.table("C:/Faks/Magistrska/data/limma_log1p_data.tsv", header=TRUE, sep = "\t",
                  row.names = 1,
                  as.is=TRUE))

meta <- as.matrix(read.table("C:/Faks/Magistrska/data/metadata_limma_test.tsv", header=TRUE, sep = "\t",
                  row.names = NULL,
                  as.is=TRUE))
head(meta)

batch <- meta[, 4]

head(batch)
batch <- t(batch)
dim(batch)
dim(count_matrix)

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("limma")

library(limma)

count_matrix <- t(count_matrix)

adjusted <- removeBatchEffect(count_matrix, batch=batch)

adjusted_t <- t(adjusted)

write.table(adjusted_t, file="C:/Faks/Magistrska/data/limma.tsv", quote=FALSE, sep='\t', col.names = NA)
