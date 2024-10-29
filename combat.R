if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

library(sva)
BiocManager::install("limma")

library(limma)

count_matrix <- as.matrix(read.table("C:/Faks/Magistrska/data/grouped_tpm.tsv", header=TRUE, sep = "\t",
                  row.names = 1,
                  as.is=TRUE))


meta <- count_matrix[, 48360:48362]
count_matrix <- count_matrix[, 1:48359]

dim(meta)
dim(count_matrix)
batch <- meta[, 3]
head(meta)
count_matrix <- t(count_matrix)
matrix2 <- apply(count_matrix, 2, as.numeric)

head(matrix2)
dim(matrix2)
matrix3 = t(matrix2)
batch = t(batch)
head(batch)
adjusted_combat = ComBat(matrix2, batch=batch)

adjusted_combat_seq = ComBat_seq(matrix2, batch=batch)

write.table(adjusted_combat, file="C:/Faks/Magistrska/data/combat.tsv", quote=FALSE, sep='\t', col.names = NA)
write.table(adjusted_combat_seq, file="C:/Faks/Magistrska/data/combat_seq.tsv", quote=FALSE, sep='\t', col.names = NA)