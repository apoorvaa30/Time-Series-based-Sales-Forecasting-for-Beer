setwd("F:/meeta/Sem Wiki Study Material/M.Mgt/Project/DATA - USa")
# memory.size(max=TRUE);    memory.limit(size=9999999999999)

library(ggplot2)
library(tseries)
library(forecast)
library(TSA)
library(mlr)
library(Matrix)
library(xgboost)
library(car)



rm(list=ls())
beer.data = read.csv("volumes_xgboost.csv")

# original
# new_tr = beer.data[c(1:156), c(1,2,3,4)]
# new_ts = beer.data[c(157:168), c(1,2,3,4)]

# first diff
# new_tr = beer.data[2:156, c(1,2,3,5)]
# new_ts = beer.data[157:168, c(1,2,3,5)]

# seasonal diff
# new_tr = beer.data[13:156, c(1,2,3,6)]
# new_ts = beer.data[157:168, c(1,2,3,6)]

# double diff
new_tr = beer.data[14:156, c(1,2,3,7)]
new_ts = beer.data[157:168, c(1,2,3,7)]

tr = as.matrix(new_tr)
ts = as.matrix(new_ts)

dtrain <- xgb.DMatrix(data = tr[,-4], label = tr[,4]) 
dtest <- xgb.DMatrix(data = ts[,-4], label = ts[,4])


#default parameters
params <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, gamma=0, 
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, print_every_n = 10, 
                 early_stopping_rounds = 20, maximize = F)

##best iteration = 77, 13, 1
xgbcv$best_iteration

#first default - model training
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 1, 
                     watchlist = list(val=dtest,train=dtrain), print_every_n = 10, 
                     early_stopping_rounds = 10, maximize = F , eval_metric = "error")
#model prediction
xgbpred <- predict (xgb1, dtest)
xgbpred
new_ts[,4]


