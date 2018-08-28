setwd("F:/meeta/Sem Wiki Study Material/M.Mgt/Project/DATA - USa")
memory.size(max=TRUE);    memory.limit(size=9999999999999)

library(ggplot2)
library(tseries)
library(forecast)
library(TSA)

rm(list=ls())
beer.data = read.csv("US_Raw_Data_FM.csv")

volume = log10( beer.data$Industry[1:168] )

vol.ts = ts(volume, frequency = 12)

plot(vol.ts)


############################# Checking plots and seasonality #################################
             
# plot(Industry ~ Year+Month, beer.data[1:168,], xaxt = "n", type = "l")
# axis(1, beer.data$Month[1:168], format(beer.data$Month[1:168], "%d/%m/%Y"), cex.axis = .7)

# ggplot(beer.data, aes(Month, Industry)) + geom_line() 
# + scale_x_date(format = "%b/%Y") 
# + xlab("") + ylab("Monthly Beer Sales")


# diff5 = diff(diff(vol.ts, lag = 12))
# acf(diff5)
# pacf(diff5)
# adf.test(diff5)
# kpss.test(diff5)
# pp.test(diff5)
# 
# 
# d=1; D=1;
# 
# # choosing diff = d1 and D1 at lag 12.
# 
# spec.pgram(vol.ts)
# spec.pgram(10^(vol.ts))
# spec.pgram(log(10^(vol.ts)))
# x = locator()$x
# round(1/x)
# round(x)

############################################################################
############################################################################
############################################################################

# train, test

start_year = 2004
end_year = 2016
pred_year = 2017

# train till 2014 : 1 to 132
# train till 2015 : 1 to 144
# train till 2016 : 1 to 156
# train till 2017 : 1 to 168

train_end = (end_year-start_year+1)*12
test_end = (pred_year-start_year+1)*12

train = ts(vol.ts[1:train_end], frequency = 12)
test = ts(vol.ts[(train_end+1):test_end], frequency = 12)

# d = diff( diff(vol.ts, lag = 12) )
# train1 = ts(d[1:(length(d)-12)], frequency = 12)
# test1 = ts(d[(length(d)-11):length(d)], frequency = 12)




######################## auto.arima ###############################################

# v = diff( diff(vol.ts,12))
# train = v[1:143]
# test = v[144:155]
# mod.arima = auto.arima(train, d = 0, D = 0,
#                        max.p = 10,
#                        max.q = 10,
#                        max.P = 5,
#                        max.Q = 5,
#                        stepwise = FALSE, approximation = FALSE,
#                        test = c("kpss","adf","pp"),
#                        ic = c("aicc", "aic", "bic"),
#                        num.cores = 10   )
#                        # ,include.mean = FALSE)
# 
# summary(mod.arima)

# small = c(3,	1,	1 ); big = c(2,	1,	3)
# small = c(3,	1,	0 ); big = c(2,	1,	3) #AIC=-796.89
# small = c(3,	1,	4 ); big = c(2,	1,	3) #---
# small = c(2,	1,	3 ); big = c(3,	1,	3) #AIC=-817.17 --
# small = c(3,	1,	1 ); big = c(2,	1,	3) #---
# small = c(1,	1,	3 ); big = c(2,	1,	2) #AIC=-780.63
# small = c(1,	1,	3 ); big = c(2,	1,	3) #not
# small = c(2,	1,	2 ); big = c(0,	1,	2) #not
small = c(2,	1,	2 ); big = c(2,	1,	2) #final

train = vol.ts[1:156]
test = vol.ts[157:168]

mod.arima = Arima(train, order = small,
             seasonal = list(order = big, period = 12) )

summary(mod.arima)


pred = forecast(mod.arima, h=12)
pred
auto.predict = pred$mean

plot(forecast(mod.arima))

auto.predict = pred$mean
10^auto.predict

actuals = as.numeric(test)
10^actuals

accuracy(10^auto.predict, 10^actuals)

# portmanteau test - white noise
LB.test(mod.arima) #null - independent


# unit root tests
kpss.test(mod.arima$residuals)
adf.test(mod.arima$residuals)
pp.test(mod.arima$residuals)


# plots
plot(mod.arima$residuals)
res<-residuals(mod.arima)
par(mfrow = c(1,2))
boxplot(res)
qqnorm(res)
hist(res)

acf(res)
pacf(res)

# tsdiag(mod.arima)


# spectral analysis
spectrum(res)
spectrum(res,spans=c(3,3))
spectrum(res,spans=c(5,5))

u<-cumsum(spectrum(res)$spec)/sum(spectrum(res)$spec)
ks.test(u,"punif") # null : drawn from the uniform distribution, p-value = 0.171


# normality
library(normtest)
# normtest(res)
normtest::jb.norm.test(res) #p-value = 0.0295, not normal
normtest::ajb.norm.test(res) #p-value = 0.0235, not normal


# autocorrelation - independence
# null : the data are independently distributed
Box.test(res,type="L")                                         #0.7298
Box.test(res, lag=1, type = "Box-Pierce", fitdf = 0)$p.value   #0.7324505
Box.test(res, lag=12, type = "Ljung-Box", fitdf = 0)$p.value    #0.7297673
Box.test(res, lag=140, type = "Ljung-Box", fitdf = 0)$p.value  #4.983483e-09


# homoscedasticity
# library(lmtest)
# bptest(mod.arima$residuals)



# prediction
pr<-predict(mod.arima,n.ahead=12)
pr$pred
pr$pred-2*pr$se
pr$pred+2*pr$se






# # monthly and yearly accuracies
## i = 1
## monthly_accuracy = vector()
## while (i <= 12) {
##   # actuals[i] = actuals[i]^(1/num)
##   predict[i] = predict[i]
##   monthly_accuracy[i] <- (1- (abs(actuals[i]-predict[i])/actuals[i]))*100
##   i <- i+1
## }
## list(monthly_accuracy)
## sum_acc = sum(actuals)
## forecast = sum(predict)
## accuracy <- (1-abs((forecast-sum_acc)/sum_acc))*100
## accuracy




###############################################################
# sarima

# P=2, p=0, Q=9, q=2
d = 1; D = 1;

models_frame = as.data.frame(rbind(rep(NA,9)))
colnames(models_frame) = c("p","d","q","P","D","Q","AIC","BIC","AICc")


for(p in 0:7){
  #for(d in 0:1){
    for(q in 0:7){
      
      for(P in 0:3){
        #for(D in 0:5){
          for(Q in 0:3){
            # mod = NA
            rm(list="mod")
            mod = tryCatch({
              Arima(train, order = c(p,d,q), 
                        seasonal = list(order = c(P,D,Q), period = 12) ) },
              error=function(e) {
                mod=NA
              }
              )
            
            if (!is.na(mod)){
              things = as.data.frame(cbind(p,d,q, P,D,Q, mod$aic, mod$bic, mod$aicc) )
              colnames(things) = c("p","d","q","P","D","Q","AIC","BIC","AICc" )
              models_frame = rbind.data.frame(models_frame, things)
              }
            }
          }
        }
      }
   # }
  #}


write.csv(models_frame, "run arima loop results apr 19.csv")

# a = 0
# try( print(3/a) )
# 3/a
