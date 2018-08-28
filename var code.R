

rm(list = ls())

setwd("F:/meeta/Sem Wiki Study Material/M.Mgt/Project/DATA - USa")

memory.size(max=TRUE);    memory.limit(size=9999999999999)

# packages = c("VAR")
# sapply(packages, library)

library(tsfa)
library(vars)
library(dplyr)
library(forecast)
library(car)


all_data = read.csv("all_vars.csv")
#####################################################################################
# HHT, FFT

# Var, tsfa


#####################################################################################
Final_data = all_data

# taking log of data
a = Final_data[ , c(1:3)]
b = Final_data[ , c(4:90)]
b = lapply(b, log10)
Final_data = as.data.frame(cbind(a,b))

# selecting variables
start_year <- 2004
end_year = 2016
pred_year = 2017

# train till 2014 : 1 to 132
# train till 2015 : 1 to 144
# train till 2016 : 1 to 156
# train till 2017 : 1 to 168

train_end = (end_year-2004+1)*12
test_end = (pred_year-2004+1)*12

Final_data$Month <- as.Date(Final_data$Month,"%d-%m-%y")

plot(Industry ~ Month, Final_data[1:168,], xaxt = "n", type = "l")
axis(1, Final_data$Month[1:168], format(Final_data$Month[1:168], "%b %y"), cex.axis = .7)

# sapply(Final_data, class)
# colnames(Final_data)

train_all <- Final_data[Final_data$Year<= end_year,]
test_all <- Final_data[Final_data$Year>end_year & Final_data$Year<=pred_year,]

variable_list = c(
  # "Month",
  # "Number",
  # "Year",
  # "Emp_HS_2",
  # "CPI.Beer",
  # "CPI_Sub_Final",
  # "UnemploymentRate",
  # "GDP.Bil.",
  # "CPI",
  # "RealGDP",
  # "GDPperCapita",
  # "RealGDPperCapita",
  # "PCE_WH",
  # "RPCE_WH",
  # "PCEPC_WH",
  # "RPCEPC_WH",
  # "GDP",
  # "RealGdp_FRED",
  # "IndustrialProductionIndex",
  # "PCE_Revised",
  # "PCPI_IX",
  # "AIP_IX",
  # "AIP_SA_IX",
  # "LE_PE_NUM",
  # "LLF_PE_NUM",
  # "LU_PE_NUM",
  # "LUR_PT",
  # "EREER_IX",
  # "ENDA_XDC_USD_RATE",
  # "ENDE_XDC_USD_RATE",
  # "ENEER_IX",
  # "ENSA_XDC_XDR_RATE",
  # "ENSE_XDC_XDR_RATE",
  # "FASLFOIL_USD",
  # "RAFAGOLDV_OZT",
  # "RAFAIMF_USD",
  # "RAFASDR_USD",
  # "RAXG_USD",
  # "RAXGFX_USD",
  # "TMG_CIF_USD",
  # "TMG_CIF_XDC",
  # "TXG_FOB_USD",
  # "TXG_FOB_XDC",
  # "Aging_rate",
  # "ageing_ratio",
  # "LDA_Population",
  # "LDA_Ratio",
  # "X20_64_Ratio",
  # "X25_64_Ratio",
  # "total_population",
  # "total_pop_yoy",
  # "pop_15_44", # --
  # "pop_15_44_yoy",
  # "per_pop_15_44",
  # "pop_15_64_yoy",
  # "per_pop_15_64",
  # "pop_15.",
  # "per_pop_15.",
  # "pop_15._yoy",
  # "pop_65.",
  # "per_pop_65.",
  # "pop_65._yoy",
  # "working_retired",
  # "BARLEY...ACRES.HARVESTED",
  # "BARLEY...PRICE.RECEIVED..MEASURED.IN.....BU..",
  # "BARLEY...PRODUCTION..MEASURED.IN....",
  # "BARLEY...PRODUCTION..MEASURED.IN.BU..",
  # "HOPS...PRODUCTION..MEASURED.IN....",
  # "HOPS...PRODUCTION..MEASURED.IN.LB..",
  # "HOPS...YIELD..MEASURED.IN.LB...ACRE..",
  # "HOPS...PRICE.RECEIVED..MEASURED.IN.....BU..",
  # "CORN..FOR.BEVERAGE.ALCOHOL...USAGE..MEASURED.IN.BU..",
  # "CORN..GRAIN...PRICE.RECEIVED..MEASURED.IN.....BU..",
  # "CORN..GRAIN...PRODUCTION..MEASURED.IN....",
  # "CORN..GRAIN...PRODUCTION..MEASURED.IN.BU..",
  # "SORGHUM..FOR.FUEL.ALCOHOL...USAGE..MEASURED.IN.CWT..",
  # "SORGHUM..GRAIN...PRICE.RECEIVED..MEASURED.IN.....CWT..",
  # "SORGHUM..GRAIN...PRODUCTION..MEASURED.IN....",
  
  
  "Precipitation", #
  "Max_temp_1", #
  # "SB.Pre", #
  # "Indep.Pre", #
  # "Labor.Day.Pre", #
  # "FASAFOIL_USD", #
  # "RAFAGOLDNV_USD", #
  # "LDA", #
  # "X20_64", #
  # "X25_64", #
  # "pop_15_64", #
  
  "Industry"
)


# train and test partition
data = Final_data[,which(colnames(Final_data) %in% variable_list)]

train <- train_all[,which(names(Final_data) %in% variable_list)]
test <- test_all[,which(names(Final_data) %in% variable_list)]

actuals <- test_all$Industry
list(actuals)

# data_st = as.data.frame(cbind(diff(data$Industry,12), 
#                               diff(data$Max_temp_1,12), 
#                               diff(data$Precipitation,12) 
#                               ))

data_st_three = as.data.frame(cbind(diff(diff(data$Industry,12),1)
                              , diff(diff(data$Max_temp_1,12),1)
                              , diff(diff(data$Precipitation,12),1)
                              ))

data_st_train = as.data.frame(cbind(diff(diff(train$Industry,12),1) 
                                    , diff(diff(train$Max_temp_1,12),1)
                                    , diff(diff(train$Precipitation,12),1)
))

data_st_test = as.data.frame(cbind(diff(diff(test$Industry,12),1) 
                                    , diff(diff(test$Max_temp_1,12),1)
                                    , diff(diff(test$Precipitation,12),1)
))

# pp.test(ts(data_st[1]))
# adf.test(ts(data_st[1]))
# kpss.test(ts(data_st[1]))
# acf(ts(data_st[1]))
# pacf(ts(data_st[1]))
# 
# pp.test(ts(data_st[2]))
# adf.test(ts(data_st[2]))
# kpss.test(ts(data_st[2]))
# acf(ts(data_st[2]))
# pacf(ts(data_st[2]))
# 
# pp.test(ts(data_st[3]))
# adf.test(ts(data_st[3]))
# kpss.test(ts(data_st[3]))
# acf(ts(data_st[3]))
# pacf(ts(data_st[3]))


###########################################################################################################
###########################################################################################################

data_st = data_st_three[,c(1,2)]

VARselect(data_st, lag.max=12, type = "const"
          # , season = 12
          )
VARselect(data_st, lag.max=12, type = "trend"
          # , season = 12
          )
VARselect(data_st, lag.max=12, type = "both"
          # , season = 12
          )
VARselect(data_st, lag.max=12, type = "none"
          # , season = 12
          )


data_st = data_st_train[,c(1,2)]
plot(data_st[,1], type='l')

VarA <- VAR(data_st, type="both", 
            # season=12, 
            ic=c("AIC","HQ","FPE"))
summary(VarA)
serial.test(VarA)
Box.test(VarA$varresult$V1$residuals)
Box.test(VarA$varresult$V2$residuals)
ResVarA <- restrict(VarA)
summary(ResVarA)
serial.test(ResVarA) #null no correlation
# V1 = V1.l1
# Residual standard error: 1109000 on 153 degrees of freedom
# Multiple R-Squared: 0.4364,	Adjusted R-squared: 0.4327 
# F-statistic: 118.5 on 1 and 153 DF,  p-value: < 2.2e-16



# VarA2 <- VAR(data_st, p = 2)
VarA2 <- VAR(data_st, p = 2, type="both"
             # , season=12
             , ic=c("AIC","HQ","FPE"))
summary(VarA2)
serial.test(VarA2)
VarA2 <- restrict(VarA2)
summary(VarA2)
serial.test(VarA2)
# V1 = V1.l1 + V3.l1 + V1.l2 
# Residual standard error: 681900 on 150 degrees of freedom
# Multiple R-Squared: 0.7909,	Adjusted R-squared: 0.7867 
# F-statistic: 189.1 on 3 and 150 DF,  p-value: < 2.2e-16 

# VarA3 <- VAR(data_st, p = 3)
colnames(data_st) = c("V1", "V2")
VarA3 <- VAR(data_st, p = 3, type="both"
             # , season=12
             , ic=c("AIC","HQ","FPE"))
summary(VarA3)
serial.test(VarA3)
VarA3 <- restrict(VarA3)
summary(VarA3)
serial.test(VarA3)
# V1 = V1.l1 + V2.l1 + V1.l2 + V2.l2 + V1.l3
# Residual standard error: 0.01811 on 135 degrees of freedom
# Multiple R-Squared: 0.7811,	Adjusted R-squared: 0.773 
# F-statistic: 96.33 on 5 and 135 DF,  p-value: < 2.2e-16 

# VarA4 <- VAR(data_st, p = 4)
VarA4 <- VAR(data_st, p = 4, type="both"
             # , season=12
             , ic=c("AIC","HQ","FPE"))
summary(VarA4)
serial.test(VarA4)
VarA4 <- restrict(VarA4)
summary(VarA4)
serial.test(VarA4)
# V1 = V1.l1 + V1.l2 + V1.l3
# Residual standard error: 0.01848 on 136 degrees of freedom
# Multiple R-Squared: 0.7694,	Adjusted R-squared: 0.7643 
# F-statistic: 151.2 on 3 and 136 DF,  p-value: < 2.2e-16 

# VarA5 <- VAR(data_st, p = 5)
VarA5 <- VAR(data_st, p = 5, type="both"
             # , season=12
             , ic=c("AIC","HQ","FPE"))
summary(VarA5)
serial.test(VarA5)
VarA5 <- restrict(VarA5)
summary(VarA5)
# V1 = V1.l1 + V2.l1 + V1.l2 + V2.l2 + V1.l3 + V2.l3 + V1.l4 + V1.l5
# Residual standard error: 0.01694 on 130 degrees of freedom
# Multiple R-Squared: 0.8146,	Adjusted R-squared: 0.8032 
# F-statistic: 71.41 on 8 and 130 DF,  p-value: < 2.2e-16 

# VarA6 <- VAR(data_st, p = 6)
VarA6 <- VAR(data_st, p = 6, type="both"
             # , season=12
             , ic=c("AIC","HQ","FPE"))
summary(VarA6)
serial.test(VarA6)
VarA6 <- restrict(VarA6)
summary(VarA6)
# V1 = V1.l1 + V2.l1 + V1.l2 + V2.l2 + V1.l3 + V2.l3 + V1.l4 + V1.l5 + V1.l6 
# Residual standard error: 0.01699 on 129 degrees of freedom
# Multiple R-Squared: 0.814,	Adjusted R-squared: 0.8024 
# F-statistic: 70.55 on 8 and 129 DF,  p-value: < 2.2e-16

# VarA7 <- VAR(data_st, p = 7)
VarA7 <- VAR(data_st, p = 7, type="both"
             # , season=12
             , ic=c("AIC","HQ","FPE"))
summary(VarA7)
serial.test(VarA7)
VarA7 <- restrict(VarA7)
summary(VarA7)
# V1 = V1.l1 + V2.l1 + V1.l2 + V2.l2 + V1.l3 + V2.l3 + V1.l4 + V1.l5 
# Residual standard error: 0.01705 on 128 degrees of freedom
# Multiple R-Squared: 0.8122,	Adjusted R-squared: 0.8004 
# F-statistic: 69.19 on 8 and 128 DF,  p-value: < 2.2e-16 

# VarA8 <- VAR(data_st, p = 8)
VarA8 <- VAR(data_st, p = 8, type="both"
             # , season=12
             , ic=c("AIC","HQ","FPE"))
summary(VarA8)
serial.test(VarA8)
VarA8 <- restrict(VarA8)
summary(VarA8)
# V1 = V1.l1 + V1.l2 + V1.l3 + V1.l4 + V1.l5 + V1.l6 + V1.l7 + V1.l8 
# Residual standard error: 0.01678 on 127 degrees of freedom
# Multiple R-Squared: 0.8189,	Adjusted R-squared: 0.8075 
# F-statistic: 71.78 on 8 and 127 DF,  p-value: < 2.2e-16

# VarA9 <- VAR(data_st, p = 9)
VarA9 <- VAR(data_st, p = 9, type="both"
             # , season=12
             , ic=c("AIC","HQ","FPE"))
summary(VarA9)
serial.test(VarA9)
VarA9 <- restrict(VarA9)
summary(VarA9)
# V1 = V1.l1 + V1.l2 + V2.l2 + V1.l3 + V1.l4 + V1.l5 + V1.l6 + V1.l9 
# Residual standard error: 0.01637 on 127 degrees of freedom
# Multiple R-Squared: 0.8274,	Adjusted R-squared: 0.8179 
# F-statistic: 86.99 on 7 and 127 DF,  p-value: < 2.2e-16 

# VarA10 <- VAR(data_st[,c(1,2)], p = 10) ################ also 11
VarA10 <- VAR(data_st, p = 10, type="both"
              # , season=12
              , ic=c("AIC","HQ","FPE"))
summary(VarA10)
serial.test(VarA10)
VarA10 <- restrict(VarA10)
summary(VarA10)
# V1 = V1.l1 + V1.l2 + V1.l3 + V1.l4 + V1.l5 + V1.l8 + V1.l10 + V2.l10 
# Residual standard error: 0.01595 on 125 degrees of freedom
# Multiple R-Squared: 0.8388,	Adjusted R-squared: 0.8284 
# F-statistic: 81.28 on 8 and 125 DF,  p-value: < 2.2e-16 

serial.test(VarA10)

plot(VarA10$varresult$V1$residuals)
qqnorm(VarA10$varresult$V1$residuals)
normtest::jb.norm.test(VarA10$varresult$V1$residuals)
normtest::ajb.norm.test(VarA10$varresult$V1$residuals)

Box.test(VarA10$varresult$V1$residuals, lag=1)
Box.test(VarA10$varresult$V1$residuals, lag=12)

# V1 = V1.l1 + V1.l2 + V2.l2 + V1.l3 + V1.l4 + V1.l8 + V1.l10 + V2.l10
# Residual standard error: 570200 on 137 degrees of freedom
# Multiple R-Squared: 0.8622,	Adjusted R-squared: 0.8542 
# F-statistic: 107.2 on 8 and 137 DF,  p-value: < 2.2e-16

# VarA11 <- VAR(data_st, p = 11)
VarA11 <- VAR(data_st, p = 11, type="both"
              # , season=12
              # , exogen = data_st[,2]
              , ic=c("AIC","HQ","FPE"))
summary(VarA11)
serial.test(VarA11)
VarA11 <- restrict(VarA11)
summary(VarA11)
# V1 = V1.l1 + V1.l2 + V2.l2 + V1.l3 + V1.l4 + V1.l8 + V1.l10 + V2.l10 
# Residual standard error: 0.01601 on 124 degrees of freedom
# Multiple R-Squared: 0.8385,	Adjusted R-squared: 0.8281 
# F-statistic:  80.5 on 8 and 124 DF,  p-value: < 2.2e-16 


# VarA12 <- VAR(data_st, p = 12)
VarA12 <- VAR(data_st, p = 12, type="both"
              # , season=12
              , ic=c("AIC","HQ","FPE"))
summary(VarA12)
serial.test(VarA12)
VarA12 <- restrict(VarA12)
summary(VarA12)
# V1 = V1.l1 + V1.l2 + V1.l3 + V1.l4 + V1.l5 + V1.l8 + V2.l10 + V1.l11 + V1.l12 
# Residual standard error: 0.0157 on 122 degrees of freedom
# Multiple R-Squared: 0.8463,	Adjusted R-squared: 0.835 
# F-statistic: 74.65 on 9 and 122 DF,  p-value: < 2.2e-16






######################################################################################







# # Restricted VAR.
# causality(VarA, cause = c("TempMax", "TempMin", "RelativeHumidity", "RainFall", "Sunshine")) # Causal
# causality(VarA3, cause = c("TempDiff")) # Causal
# causality(VarA3, cause = c("TempMin")) # Causal
# causality(VarA3, cause = c("RelativeHumidity")) # Not causal
# causality(VarA3, cause = c("RainFall")) # causal
# causality(VarA3, cause = c("Sunshine")) # Not causal
# 
# causality(VarA3, cause = c("TempMax", "TempMin")) # causal
# causality(VarA3, cause = c("TempMax", "TempMin", "RelativeHumidity")) # Not causal
# causality(VarA3, cause = c("TempMax", "TempMin", "RelativeHumidity", "RainFall")) # Causal
# causality(VarA3, cause = c("TempMax", "TempMin", "RelativeHumidity", "Sunshine"))
# 
# RHIRFA <- irf(VarA3, impulse = "RelativeHumidity",response = c("RustIncidence", "Severity") )
# plot(RHIRFA, main = "Impulse response from relative humidity for variety A", xlab = "Periods")
# 
# RFIRFA <- irf(VarA3,impulse = "RainFall", response = c("RustIncidence", "Severity") )
# plot(RFIRFA, main = "Impulse response from rainfall for variety A", xlab = "Periods")
# 
# SSIRFA <- irf(VarA3, impulse = "Sunshine",response = c("RustIncidence", "Severity") )
# plot(SSIRFA, main = "Impulse response from Sunshine for variety A", xlab = "Periods")
# 
# 
# TDIRFA <- irf(VarA, impulse = "TempDiff",response = c("RustIncidence", "Severity") )
# plot(TDIRFA, main = "Impulse response from temperature difference for variety A", xlab = "Periods")
# 
# 





VarA3 <- VAR(data_st[,], p = 3, type="both"
             # , season=12
             , ic=c("AIC","HQ","FPE"))
summary(VarA3)
serial.test(VarA3)
VarA3 <- restrict(VarA3)
summary(VarA3)
serial.test(VarA3)

#Test for serial autocorrelation using the Portmanteau test
#Rerun var model with other suggested lags if H0 can be rejected at 0.05
serial.test(VarA3, lags.pt = 10, type = "PT.asymptotic")
#ARCH test (Autoregressive conditional heteroscedasdicity)
arch.test(VarA3, lags.multi = 10)

plot(VarA3$varresult$V1$residuals)
qqnorm(VarA3$varresult$V1$residuals)
hist(VarA3$varresult$V1$residuals)
normtest::jb.norm.test(VarA3$varresult$V1$residuals)
normtest::ajb.norm.test(VarA3$varresult$V1$residuals)

Box.test(VarA3$varresult$V1$residuals, lag=1)
Box.test(VarA3$varresult$V1$residuals, lag=5)
# V1 = V1.l1 + V2.l1 + V1.l2 + V2.l2 + V1.l3
# Residual standard error: 0.01811 on 135 degrees of freedom
# Multiple R-Squared: 0.7811,	Adjusted R-squared: 0.773 
# F-statistic: 96.33 on 5 and 135 DF,  p-value: < 2.2e-16 

# causality(VarA3, cause = c("V2", "V3")) # Causal
causality(VarA3, cause = c("V2")) # Causal
# causality(VarA3, cause = c("V3")) # Causal

TmaxIRFA <- irf(VarA3, impulse = "V2", response = "V1")
plot(TmaxIRFA, main = "Impulse response from Max Monthly Temperature for Beer Sales")

grangertest(data_st$V1 ~ data_st$V2, order = 3) #yes
grangertest(data_st$V2 ~ data_st$V1, order = 3) #no

#Forecasting
prd <- predict(VarA3, n.ahead = 12, ci = 0.95, dumvar = NULL)
print(prd)
plot(prd, "single")







VarA12 <- VAR(data_st, p = 12, type="both"
              # , season=12
              , ic=c("AIC","HQ","FPE"))
summary(VarA12)
serial.test(VarA12)
VarA12 <- restrict(VarA12)
summary(VarA12)
serial.test(VarA12)

plot(VarA12$varresult$V1$residuals)
qqnorm(VarA12$varresult$V1$residuals)
hist(VarA12$varresult$V1$residuals)
normtest::jb.norm.test(VarA12$varresult$V1$residuals)
normtest::ajb.norm.test(VarA12$varresult$V1$residuals)

Box.test(VarA12$varresult$V1$residuals, lag=1)
Box.test(VarA12$varresult$V1$residuals, lag=12)
# V1 = V1.l1 + V1.l2 + V1.l3 + V1.l4 + V1.l5 + V1.l8 + V2.l10 + V1.l11 + V1.l12 
# Residual standard error: 0.0157 on 122 degrees of freedom
# Multiple R-Squared: 0.8463,	Adjusted R-squared: 0.835 
# F-statistic: 74.65 on 9 and 122 DF,  p-value: < 2.2e-16


causality(VarA12, cause = c("V2", "V3")) # Causal
causality(VarA12, cause = c("V2")) # Causal
causality(VarA12, cause = c("V3")) # Causal



