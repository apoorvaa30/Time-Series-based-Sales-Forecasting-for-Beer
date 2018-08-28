rm(list = ls())

setwd("F:/meeta/Sem Wiki Study Material/M.Mgt/Project/DATA - USa")

# install.packages("tsfa")
# install.packages("neldermead")

library("neldermead")
library("tsfa")
library("dplyr")
library("openxlsx")





############ constants ###############################

# level of significance test
alpha = 0.05

# number of lags considered, arbitrary integer >= 1
p = 10





########### data #####################################

dat = read.csv("all_vars.csv", header = TRUE)
# dat$year = substr(dat$year_month,1,4)
# dat$month = substr(dat$year_month,6,7)

dat = dat[1:208,-1]
# sapply(dat, class)

d = ncol(dat)

n = nrow(dat)

########### lagged data ##################################

# dat = dat[,-1]
mean_vector = as.matrix(colMeans(dat))


Sk <- array(rep(0, p*d*d), dim=c(p, d, d))

for(k in 1:p){
  sn = matrix(data = 0, nrow = d, ncol = d)
  for(t in (k+1):n){
    sn = sn + as.matrix(t(dat[t,]) - mean_vector)%*%t(as.matrix(t(dat[(t-k),]) - mean_vector)) 
  }
  # print(sn/n)
  Sk[k,,] = sn/n
  print(Sk[k,,])
}

write.csv(Sk, "Sk.csv")







