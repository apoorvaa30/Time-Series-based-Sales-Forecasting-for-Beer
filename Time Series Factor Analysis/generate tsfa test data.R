
rm(list = ls())

setwd("F:/meeta/Sem Wiki Study Material/M.Mgt/Project/DATA - USa")

set.seed(123)

nums = read.csv("ran_num.csv")

e1 = rnorm(400, 0, 4)

e2 = rnorm(400, 0, 7)

e3 = rnorm(400, 0, 13)

x1 = as.matrix(NA, ncol = 1, nrow = 3)
x1[1] = 2
for(i in 2:300){
  x1[i] = 0.8*x1[i-1] + e1[i]
}


x2 = as.matrix(NA, ncol = 1, nrow = 3)
x2[1] = 19
x2[2] = 14
for(i in 3:300){
  x2[i] = 0.3*e2[i-2] + 0.9*e2[i-1] +e2[i]
}


x3 = as.matrix(NA, ncol = 1, nrow = 3)
x3[1] = 30
for(i in 2:300){
  x3[i] = 0.8*e3[i-1] + e3[i] - 0.5*x3[i-1]
}


y1 = as.matrix(NA, ncol = 1, nrow = 300)
for(i in 1:300){
  y1[i] = x1[i] + e1[i]
}

y2 = as.matrix(NA, ncol = 1, nrow = 300)
for(i in 1:300){
  y2[i] = x2[i] + e2[i]
}

y3 = as.matrix(NA, ncol = 1, nrow = 300)
for(i in 1:300){
  y3[i] = x3[i] + e3[i]
}

y4 = rnorm(300, 0, 98)

y5 = rnorm(300, 0, 1)






dat = as.data.frame(cbind(y1,y2,y3,y4,y5))
write.csv(dat, "test_tsfa.csv")



