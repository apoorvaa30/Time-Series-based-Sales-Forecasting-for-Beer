# descriptive time series modelling

setwd("F:/meeta/Sem Wiki Study Material/M.Mgt/Project/DATA - USa")
memory.size(max=TRUE);    memory.limit(size=9999999999999)

library(ggplot2)
library(tseries)
library(forecast)
library(TSA)

rm(list=ls())
beer.data = read.csv("US_Raw_Data_FM.csv")



########################### monthly seasonality check ###################


gdpcur<-ts(beer.data$Industry[1:168])

gdpcon<-ts(beer.data$Industry[1:168],deltat=1/12)

t = 1:168
plot(c(1,168),range(c(gdpcur,gdpcon)),type="n",main="Monthly Beer Shipment of USA
from 2004 to 2017",xlab="Months",ylab="Volume in HL")

lines(t,gdpcon)
points(t,gdpcon,pch=20)

lines(t,gdpcur,col=2)
points(t,gdpcur,pch=20,col=2)

legend(2,1000000,legend=c("no deltat","used deltat"),
         col=c(1,2),cex=1.5)


###############################################
decomposed.gdpcon<-decompose(gdpcon)

points(t,decomposed.gdpcon$trend)
lines(t,decomposed.gdpcon$trend)

# points(t,decompose(gdpcur)$trend,col=2)
# lines(t,decompose(gdpcur)$trend,col=2)

#########################################################################

decomposed.gdpcon

plot(decomposed.gdpcon)


# Checking the Independent Error Assumption
res1<-decomposed.gdpcon$random

plot(t,res1,type="l")

res1 = na.contiguous(res1)

acf(res1)
pacf(res1)


res1 = diff(diff(gdpia, lag = 12))

# independence tests

library(car)
dwt(as.vector(res1))

Box.test(res1, type = "Box-Pierce")
Box.test(res1, type = "Ljung-Box")

turning_point.test<-function(x)
  {
    n<-length(x)
    turns<-0
    for(i in 2:(n-1))
      if((x[i-1]<x[i] && x[i]>x[i+1]) || (x[i-1]>x[i] && x[i]<x[i+1]))
        turns<-turns+1
        z<-(turns-2*(n-2)/3)/sqrt((16*n-29)/90)
        return(list(nturns=turns,z=z,p=2*(1-pnorm(abs(z)))))
        }

turning_point.test(res1)



diff_sign.test<-function(x)
   {
     n<-length(x)
     y<-diff(x)
     t<-length(y[y>0])
     z<-(t-(n-1)/2)/sqrt((n+1)/12)
     return(list(t=t,z=z,p=2*(1-pnorm(abs(z)))))
     }
diff_sign.test(res1)



linear_trend.test<-function(x)
   {
     n<-length(x)
     t<-0
     for(i in 1:(n-1))
       for(j in (i+1):n)
         if(x[j]>x[i]) t<-t+1
         z<-(t-n*(n-1)/4)/sqrt((n*(n-1)*(2*n+5)/72))
         return(list(t=t,z=z,p=2*(1-pnorm(abs(z)))))
         }
linear_trend.test(res1)




# heteroscedasticity test
library(lmtest)
# ols_test_bartlett(res1)
bptest(as.vector(res1))



# normality test
library(normtest)
normtest::jb.norm.test(res1) #p-value = 0.0925, normal
normtest::ajb.norm.test(res1) #p-value = 0.0965, normal

