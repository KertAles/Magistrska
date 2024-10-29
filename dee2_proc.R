library(reshape2)

x <- read.table("arabidopsis_ke.tsv.bz2")
w <- t(acast(x, V1~V2, value.var="V3",fun.aggregate=sum))
tpm <- apply(w,2,function(x) { x/sum(x) *1e6 } )

write.table(tpm _t, file="C:/Faks/Magistrska/data/athaliana_tpm.tsv", quote=FALSE, sep='\t', col.names = NA)

