# seasonality modelling

setwd("F:/meeta/Sem Wiki Study Material/M.Mgt/Project/DATA - USa")
# memory.size(max=TRUE);    memory.limit(size=9999999999999)

library(ggplot2)
library(tseries)
library(forecast)
library(TSA)

rm(list=ls())
beer.data = read.csv("US_Raw_Data_FM.csv")

beer.data = log10(beer.data$Industry)


########################### seasonality check ###################

p = periodogram(beer.data[1:168])
p
dd = data.frame(freq=p$freq, spec=p$spec)
order = dd[order(-dd$spec),]
top6 = head(order, 6)

# display the 2 highest "power" frequencies
top6

# convert frequency to time periods
time = 1/top2$f
time
# 12, 3


########################### monthly seasonality check ###################


gdpia<-ts(beer.data[1:168])

pp.test(gdpia)$p.value
kpss.test(gdpia)$p.value
adf.test(gdpia)$p.value

q1<-gdpia[seq(1,168,12)]
q2<-gdpia[seq(2,168,12)]
q3<-gdpia[seq(3,168,12)]
q4<-gdpia[seq(4,168,12)]
q5<-gdpia[seq(5,168,12)]
q6<-gdpia[seq(6,168,12)]
q7<-gdpia[seq(7,168,12)]
q8<-gdpia[seq(8,168,12)]
q9<-gdpia[seq(9,168,12)]
q10<-gdpia[seq(10,168,12)]
q11<-gdpia[seq(11,168,12)]
q12<-gdpia[seq(12,168,12)]
  
plot(gdpia,xlab="Time",ylab="Monthly Beer Shipment Volumes in HL", main=
         "Monthly Beer Shipment Volume \n of USA from 2004 to 2017")



plot(c(2004:2017),seq(12400986, 21460988, 696923.2),type="n",xlab="Year",ylab="Monthly Beer Shipment 
        Volumes in HL",main="Beer Shipment Volume \n of USA from 2004 to 2017")
lines(2004:2017,q1)
lines(2004:2017,q2,lty=2)
lines(2004:2017,q3,lty=3)
legend(2004,400000,legend=c("Q1","Q2","Q3"),lty=1:12)

plot(c(2004:2017),seq(12400986, 21460988, 696923.2),type="n",xlab="Year",ylab="Monthly Beer Shipment 
        Volumes in HL",main="Beer Shipment Volume \n of USA from 2004 to 2017")
lines(2004:2017,q4,lty=4)
lines(2004:2017,q5,lty=5)
lines(2004:2017,q6,lty=6)
legend(2004,400000,legend=c("Q4","Q5","Q6"),lty=1:12)

plot(c(2004:2017),seq(12400986, 21460988, 696923.2),type="n",xlab="Year",ylab="Monthly Beer Shipment 
        Volumes in HL",main="Beer Shipment Volume \n of USA from 2004 to 2017")
lines(2004:2017,q7,lty=7)
lines(2004:2017,q8,lty=8)
lines(2004:2017,q9,lty=9)
legend(2004,400000,legend=c("Q7","Q8","Q9"),lty=1:12)

plot(c(2004:2017),seq(12400986, 21460988, 696923.2),type="n",xlab="Year",ylab="Monthly Beer Shipment 
        Volumes in HL",main="Beer Shipment Volume \n of USA from 2004 to 2017")
lines(2004:2017,q10,lty=10)
lines(2004:2017,q11,lty=11)
lines(2004:2017,q12,lty=12)
legend(2004,400000,legend=c("Q9","Q10","Q11","Q12"),lty=1:12)




legend(2004,400000,legend=c("Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10","Q11","Q12"),lty=1:12)

  pp.test(q1)$p.value
  pp.test(q2)$p.value
  pp.test(q3)$p.value
  pp.test(q4)$p.value
  pp.test(q5)$p.value
  pp.test(q6)$p.value
  pp.test(q7)$p.value
  pp.test(q8)$p.value
  pp.test(q9)$p.value
  pp.test(q10)$p.value
  pp.test(q11)$p.value
  pp.test(q12)$p.value

  #############################################################
  
  
  
  q1<-gdpia[seq(1,168,4)]
  q2<-gdpia[seq(3,168,4)]
  q3<-gdpia[seq(6,168,4)]
  q4<-gdpia[seq(9,168,4)]
  
  pp.test(q1)$p.value
  pp.test(q2)$p.value
  pp.test(q3)$p.value
  pp.test(q4)$p.value
  

  ################## usual 1 difference stationarity check ##################
  
  ts_1 = diff(gdpia,lag=1)
  
  plot(diff(gdpia,lag=1),ylab="Difference 1 Usual",
       main="Time Series Plot of the Differenced Series")
  
  adf.test(diff(gdpia,lag=1))$p.value
  pp.test(diff(gdpia,lag=1))$p.value
  kpss.test(diff(gdpia,lag=1))$p.value
  
  acf(diff(gdpia,lag=1))  # looks like 14, or just 1
  pacf(diff(gdpia,lag=1)) # 2 spikes at 4
  spectrum(diff(gdpia,lag=1),spans=c(5,5))
  
  
  p = periodogram(diff(gdpia,lag=1))
  p
  dd = data.frame(freq=p$freq, spec=p$spec)
  order = dd[order(-dd$spec),]
  top2 = head(order, 2)
  
  # display the 2 highest "power" frequencies
  top2
  
  # convert frequency to time periods
  time = 1/top2$f
  time
  # > time
  # [1] 2.318841 2.191781
  
##################### seasonal first difference stationarity check #########################
  
plot(diff(gdpia,lag=12),ylab="Seasonal Difference",main="Time Series Plot of
Seasonal Difference")

  adf.test(diff(gdpia,lag=12))$p.value
  pp.test(diff(gdpia,lag=12))$p.value
  kpss.test(diff(gdpia,lag=12))$p.value
  
  acf(diff(gdpia,lag=12))
  pacf(diff(gdpia,lag=12))
  
  spectrum(diff(gdpia,lag=12),spans=c(14,14))
  
  
  
  p = periodogram(diff(gdpia,lag=12))
  p
  dd = data.frame(freq=p$freq, spec=p$spec)
  order = dd[order(-dd$spec),]
  top2 = head(order, 2)
  
  # display the 2 highest "power" frequencies
  top2
  
  # convert frequency to time periods
  time = 1/top2$f
  time
  # > time
  # [1] 2.857143 2.318841
  
  
  
############### BEST ### usual and seasonal first difference stationarity check ###########

  ts_s_1 = diff(diff(gdpia,lag=12))
  
  beer = gdpia
  acf(beer)
  pacf(beer)

plot(diff(diff(gdpia,lag=12)),ylab="First Difference (Seasonal+Usual)",
       main="Time Series Plot of the Differenced Series")

  adf.test(diff(diff(gdpia,lag=12)))$p.value
  pp.test(diff(diff(gdpia,lag=12)))$p.value
  kpss.test(diff(diff(gdpia, lag = 12)))$p.value
  
par(mfrow = c(1,1))
acf(diff(diff(gdpia,lag=12)))  # looks random
pacf(diff(diff(gdpia,lag=12))) # 2 spikes
spectrum(diff(diff(gdpia,lag=12)),spans=c(5,5))



p = periodogram(diff(diff(gdpia,lag=12)))
p
dd = data.frame(freq=p$freq, spec=p$spec)
order = dd[order(-dd$spec),]
top2 = head(order, 2)

# display the 2 highest "power" frequencies
top2

# convert frequency to time periods
time = 1/top2$f
time
# > time
# [1] 2.857143 2.318841


################## seasonal 2 differences stationarity check ##################

ts_2sea = diff(diff(gdpia,lag=12), lag=3)


plot(diff(diff(gdpia,lag=12), lag=3),ylab="Second Difference (Seasonal)",
     main="Time Series Plot of the Differenced Series")

adf.test(diff(diff(gdpia,lag=12), lag=3))$p.value
# 0.01
pp.test(diff(diff(gdpia,lag=12), lag=3))$p.value
# 0.01
kpss.test(diff(diff(gdpia, lag = 12), lag=3))$p.value
# 0.1

acf(diff(diff(gdpia,lag=12), lag=3))  # looks 14
pacf(diff(diff(gdpia,lag=12), lag=3)) # 2 spikes at 4
spectrum(diff(diff(gdpia,lag=12), lag=3),spans=c(5,5))


p = periodogram(diff(diff(gdpia,lag=12), lag=3))
p
dd = data.frame(freq=p$freq, spec=p$spec)
order = dd[order(-dd$spec),]
top2 = head(order, 2)

# display the 2 highest "power" frequencies
top2

# convert frequency to time periods
time = 1/top2$f
time
# > time
# [1] 2.318841 2.191781




################## usual and seasonal 2 differences stationarity check ##################

ts_2sea_1 = diff(diff(diff(gdpia,lag=12)))

plot(diff(diff(diff(gdpia,lag=12), lag=3)),ylab="Difference (2 Seasonal + 1Usual)",
     main="Time Series Plot of the Differenced Series")

adf.test(diff(diff(diff(gdpia,lag=12), lag=3)))$p.value
# 0.01
pp.test(diff(diff(diff(gdpia,lag=12), lag=3)))$p.value
# 0.01
kpss.test(diff(diff(diff(gdpia, lag = 12), lag=3)))$p.value
# 0.1

acf(diff(diff(diff(gdpia,lag=12), lag=3)) )  # looks like 14, or just 1
pacf(diff(diff(diff(gdpia,lag=12), lag=3))) # 2 spikes at 4
spectrum(diff(diff(diff(gdpia,lag=12), lag=3),spans=c(5,5)))


p = periodogram(diff(diff(gdpia,lag=12), lag=3))
p
dd = data.frame(freq=p$freq, spec=p$spec)
order = dd[order(-dd$spec),]
top2 = head(order, 2)

# display the 2 highest "power" frequencies
top2

# convert frequency to time periods
time = 1/top2$f
time
# > time
# [1] 2.318841 2.191781









##########################################################################################
##########################      Model Building     #######################################
##########################################################################################

# train = ts_2sea[1:141]
# test = ts_2sea[142:153]
# 
# mod.arima = auto.arima(ts_2sea
#                        , d = 0, D = 0,
#                        max.p = 16,
#                        max.q = 16,
#                        max.P = 10,
#                        max.Q = 10,
#                        stepwise = TRUE, approximation = FALSE,
#                        test = c("kpss","adf","pp"),
#                        ic = c("aicc", "aic", "bic"),
#                        num.cores = 10)
# 
# summary(mod.arima)
# 
# 
# 
# # prediction
# pred = forecast(mod.arima, h=12)
# auto.predict = pred$mean
# actuals = test
# 
# plot(ts_2sea,type = "l",color = "blue")
# points(auto.predict,type = "l",col = "red")
# 
# 
# 
# ########################### Residual Analysis #########################################
# 
# # portmanteau test - white noise
# LB.test(mod.arima)
# 
# 
# # unit root tests
# kpss.test(mod.arima$residuals)
# adf.test(mod.arima$residuals)
# pp.test(mod.arima$residuals)
# 
# 
# # plots
# plot(mod.arima)
# res<-residuals(mod.arima)
# boxplot(res)
# qqnorm(res)
# 
# acf(res)
# pacf(res)
# 
# tsdiag(mod.arima)
# 
# 
# # spectral analysis
# spectrum(res)
# spectrum(res,spans=c(3,3))
# spectrum(res,spans=c(5,5))
# 
# u<-cumsum(spectrum(res)$spec)/sum(spectrum(res)$spec)
# ks.test(u,"punif")
# 
# 
# # normality
# library(normtest)
# # normtest(res)
# normtest::jb.norm.test(res) #p-value = 0.566, normal
# normtest::ajb.norm.test(res) #p-value = 0.481, normal
# 
# 
# # autocorrelation - independence
# Box.test(res,lag=20,type="L")
# Box.test(res, lag=1, type = "Box-Pierce", fitdf = 0)$p.value   #0.8706662
# Box.test(res, lag=1, type = "Ljung-Box", fitdf = 0)$p.value    #0.8693002
# Box.test(res, lag=140, type = "Ljung-Box", fitdf = 0)$p.value  #0.2455635
# 
# 
# # homoscedasticity
# library(lmtest)
# bptest(mod.arima)
# 
# 
# 
# # prediction
# pr<-predict(mod.arima,n.ahead=12)
# pr$pred-2*pr$se
# pr$pred+2*pr$se




















# 
      
      
      
      
      
      
