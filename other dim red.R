

rm(list = ls())

setwd("F:/meeta/Sem Wiki Study Material/M.Mgt/Project/DATA - USa")

# install.packages("tsfa")
# install.packages("neldermead")

library("neldermead")
library("tsfa")
library("dplyr")
library("openxlsx")



########### data ################################################################

# dat = read.csv("all_vars.csv", header = TRUE)

# dat = read.csv("US_Raw_Data_FM.csv", header = TRUE)

dat = read.csv("test_tsfa.csv", header = TRUE)

# dat$year = substr(dat$year_month,1,4)
# dat$month = substr(dat$year_month,6,7)

dat = dat[,-1]
# dat = dat[1:204,9:17]

d = ncol(dat)
n = nrow(dat)



##########################################################################################


# tsfa gilbert
# svd
# pca
# regularization - lasso, ridge
# tsfa pan yao

# manifold non-linear dimension reduction




# dwt
# svd
# dft
# paa

##########################################################################################

# tsfa gilbert


z = as.ts(dat)
z
z <-tfwindow(z, start=1, end=300)
z


tfplot(z, graphs.per.page=3)
tfplot(diff(z), graphs.per.page=3)


start(z)
end(z)
Tobs(z)

DX <- diff(z, lag=1)
nseries(z)

colMeans(DX)

sqrt(diag(cov(DX)))

######################


zz <- eigen(cor(diff(z, lag=1)), symmetric=TRUE)[["values"]]
print(zz)

par(omi=c(0.1,0.1,0.1,0.1),mar=c(4.1,4.1,0.6,0.1))
plot(zz, ylab="Value", xlab="Eigenvalue Number", pch=20:20,cex=1,type="o")

z1 <- FAfitStats(z)
print(z1, digits=3)

c1withML <- estTSF.ML(z, 1)
c2withML <- estTSF.ML(z, 2)
c3withML <- estTSF.ML(z, 3)
c4withML <- estTSF.ML(z, 4)
c5withML <- estTSF.ML(z, 5)


print(DstandardizedLoadings(c1withML) )
print(c1withML$Phi, digits=3)

print(DstandardizedLoadings(c3withML) )
print(c3withML$Phi, digits=3)

print(DstandardizedLoadings(c4withML) )
print(c4withML$Phi, digits=3)

print(DstandardizedLoadings(c5withML) )
print(c5withML$Phi, digits=3)


#########################################################

# without rotation

print(DstandardizedLoadings(c2withML) )
print(c2withML$Phi, digits=3)
print(1 - c2withML$stats$uniquenesses)
print(loadings(c2withML) )

tfplot(ytoypc(factors(c2withML)),
       Title= "Factors from 2 factor model (year-to-year growth rate)",
       lty=c("solid"),
       col=c("black"),
       xlab=c(""),ylab=c("factor 1","factor 2"),
       par=list(mar=c(2.1, 4.1, 1.1, 0.1)),
       reset.screen=TRUE)

tfplot(factors(c2withML),
       Title="Factors from 2 factor model",
       lty=c("solid"),
       col=c("black"),
       xlab=c(""),ylab=c("factor 1","factor 2"),
       par=list(mar=c(2.1, 4.1, 1.1, 0.1)),
       reset.screen=TRUE)

z2 <- explained(c2withML)

tfplot(ytoypc(z), ytoypc(explained(c2withML)), series=1:5,
       graphs.per.page=5,
       lty=c("solid", "dashed"),
       col=c("black", "red"),
       ylab=c("y1","y2","y3","y4","y5"),
       Title=
         "Explained indicators 1-5 (year-to-year growth rate)using 2 factors",
       par=list(mar=c(2.1, 4.1, 1.1, 0.1)),
       reset.screen=TRUE)

tfplot( z, explained(c2withML), series=1:5, graphs.per.page=5,
        lty=c("solid", "dashed"),
        col=c("black", "red"),
        Title= "Explained indicators 1-5 using 2 factors",
        par=list(mar=c(2.1, 4.1, 1.1, 0.1)),
        reset.screen=TRUE)

tfplot( diff(z), diff(explained(c2withML)), graphs.per.page=2)




# with rotation















