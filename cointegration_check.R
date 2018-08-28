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
library(quantmod)
library(tseries)
library(urca)
library(CADFtest)
library(tseries)


data = read.csv("all_vars.csv")


###########################################################################################################

set.seed(123)

back_up <- data
start_year <- 2004
data$Month <- as.Date(data$Month,"%d-%m-%y")
plot(Industry ~ Month, data[1:168,], xaxt = "n", type = "l")
axis(1, data$Month[1:168], format(data$Month[1:168], "%b %y"), cex.axis = .7)

Final_data <- data
sapply(Final_data, class)
colnames(Final_data)
# Final_data[,c(5:143)] <- lapply(Final_data[,c(5:143)], as.numeric)
# Final_data[,c(24:35,77:78,105:118,133,135:136,140)] <- lapply(Final_data[,c(24:35,77:78,105:118,133,135:136,140)], factor)

# data cleanliness check
sapply(Final_data, class)
# which(colSums(is.na(Final_data))>0)  
# lapply(Final_data, function(x) x[is.infinite(x)])  #

end_year = 2016
pred_year = 2017

# train till 2014 : 1 to 132
# train till 2015 : 1 to 144
# train till 2016 : 1 to 156
# train till 2017 : 1 to 168

train_end = (end_year-2004+1)*12
test_end = (pred_year-2004+1)*12

# train_all <- Final_data[Final_data$Year<= end_year,]
# test_all <- Final_data[Final_data$Year>end_year & Final_data$Year<=pred_year,]
# 
# actuals <- test_all$Industry
# list(actuals)

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
  # "Max_temp_1", #
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

selected_vars <- c(
  "Precipitation", #
  "Max_temp_1", #
  "SB.Pre", #
  "Indep.Pre", #
  "Labor.Day.Pre", #
  "FASAFOIL_USD", #
  "RAFAGOLDNV_USD", #
  "LDA", #
  "X20_64", #
  "X25_64", #
  "pop_15_64", #
  
  "Industry"
)


vals <- Final_data[,which(names(Final_data) %in% variable_list)]

all_vals <- Final_data[, which(names(Final_data) %in% selected_vars)]




###########################################################################################################
################################### Stationarity Check ####################################################
###########################################################################################################

# df = all_vals
# # checking for unit roots
# p = df[2]
# 
# 
# # stationarity check for temp
# gdpia<-ts(p)
# 
# adf.test(gdpia) #null unit root
# pp.test(gdpia) #null unit root
# kpss.test(gdpia, null = c("Level", "Trend")) #null stationary
# kpss.test(gdpia)$p.value
# 
# 
# 
# 
# 
# r1<-gdpia[seq(1,168,4)]
# r2<-gdpia[seq(2,168,4)]
# r3<-gdpia[seq(3,168,4)]
# r4<-gdpia[seq(4,168,4)]
# 
# pp.test(r1)$p.value
# pp.test(r2)$p.value
# pp.test(r3)$p.value
# pp.test(r4)$p.value
# 
# 
# 
# 
# 
# 
# q1<-gdpia[seq(1,168,12)]
# q2<-gdpia[seq(2,168,12)]
# q3<-gdpia[seq(3,168,12)]
# q4<-gdpia[seq(4,168,12)]
# q5<-gdpia[seq(5,168,12)]
# q6<-gdpia[seq(6,168,12)]
# q7<-gdpia[seq(7,168,12)]
# q8<-gdpia[seq(8,168,12)]
# q9<-gdpia[seq(9,168,12)]
# q10<-gdpia[seq(10,168,12)]
# q11<-gdpia[seq(11,168,12)]
# q12<-gdpia[seq(12,168,12)]
# 
# pp.test(q1)$p.value
# pp.test(q2)$p.value
# pp.test(q3)$p.value
# pp.test(q4)$p.value #
# pp.test(q5)$p.value
# pp.test(q6)$p.value
# pp.test(q7)$p.value
# pp.test(q8)$p.value
# pp.test(q9)$p.value #
# pp.test(q10)$p.value
# pp.test(q11)$p.value
# pp.test(q12)$p.value
# 
# # MAX_TEMP : seasonal 12 is non-stationary, whole and quarterly is stationary.






###########################################################################################################
################################### Paired Cointegration Check ############################################
###########################################################################################################

# https://www.quantshare.com/blog-542-introduction-to-cadf-and-johansen-statistical-tests-for-cointegration









################################### Johansen's Test #############################
#  null : no cointegration
# if test stat >> level value, then p is small and null is rejected
# https://www.quantstart.com/articles/Johansen-Test-for-Cointegrating-Time-Series-Analysis-in-R
# http://denizstij.blogspot.com/2013/11/cointegration-tests-adf-and-johansen.html

t = c(1:length(df$Max_temp_1))


df = all_vals[, c(1,12)]



jotest=ca.jo(df, type="eigen", K=2, ecdet="trend", spec="longrun"
             # , season = 12
             )
summary(jotest)
# s = 1.000*df$Industry -16017.040*df$Max_temp_1 + 8139.001*t    #w season
# s = 1.000*df$Industry -130113.130*df$Max_temp_1 + 8449.904*t #wo season
# acf(s)
# pacf(s)
# plot(s, type="l")
# adf.test(s)
# pp.test(s)
# s1 = s[seq(1,168,12)]
# adf.test(s1)
# s2 = s[seq(2,168,12)]
# adf.test(s2)
# s3 = s[seq(3,168,12)]
# adf.test(s3)


jotest=ca.jo(df, type="eigen", K=6, ecdet="trend", spec="longrun"
             # , season = 12
             ) #--
summary(jotest)
# s = 1.000*df$Industry +347743.524*df$Max_temp_1 + 6234.216*t   #how to interpret t
# # s = 1.000*df$Industry +347743.524*df$Max_temp_1 + 6234.216   #no t
# acf(s)
# pacf(s)
# plot(s, type="l")
# adf.test(s)
# pp.test(s)
# s1 = s[seq(1,168,12)]
# adf.test(s1)
# s2 = s[seq(2,168,12)]
# adf.test(s2)
# s3 = s[seq(3,168,12)]
# adf.test(s3)



jotest=ca.jo(df, type="eigen", K=3, ecdet="trend", spec="longrun"
             # , season = 12
             )
summary(jotest)
# s = 1.000*df$Industry -381676.91*df$Max_temp_1 +10294.04*t
# acf(s)
# plot(s, type="l")
# adf.test(s)

jotest=ca.jo(df, type="trace", K=4, ecdet="trend", spec="longrun") ##
summary(jotest)
# s = 1.000*df$Industry + 1908192.78*df$Max_temp_1 -2955.07*t
# acf(s)
# plot(s, type="l")
# adf.test(s)

jotest=ca.jo(df, type="eigen", K=5, ecdet="trend", spec="longrun")
summary(jotest)
# s = 1.000*df$Industry + 463449.090*df$Max_temp_1 + 5507.439*t
# acf(s)
# plot(s, type="l")
# adf.test(s)






jotest=ca.jo(df, type="eigen", K=7, ecdet="trend", spec="longrun") #--
summary(jotest)

jotest=ca.jo(df, type="trace", K=8, ecdet="trend", spec="longrun") #--
summary(jotest)

jotest=ca.jo(df, type="eigen", K=9, ecdet="trend", spec="longrun") #-
summary(jotest)

jotest=ca.jo(df, type="trace", K=10, ecdet="trend", spec="longrun") 
summary(jotest)

jotest=ca.jo(df, type="eigen", K=11, ecdet="trend", spec="longrun")
summary(jotest)

jotest=ca.jo(df, type="trace", K=12, ecdet="trend", spec="longrun")
summary(jotest)



# for( i in c(2:12)){
#   # jotest = NA
#   jotest = tryCatch ( { ca.jo(df, type="eigen", K=i, ecdet="none", spec="longrun") },
#                       error = function(e){
#                         print("not applicable")
#                         return(NA)
#                       }
#                       )
#   if(!is.na(jotest)){
#     summary(jotest)
#   }
# }







################################### ARDL Test #############################

# FoR MAX TEMPERATURE #####################################################

library(dLagM)

df = all_vals[, c(1,2)]
d = ardlDlm(Industry~Max_temp_1,
            data = df,
            p = 12,
            q = 12,
            x = df$Max_temp_1,
            y = df$Industry)
d = dlm(Industry~Max_temp_1,
            data = df,
            # p = 12,
            q = 12,
            x = df$Max_temp_1,
            y = df$Industry,
        show.summary = TRUE)



################################### CADF Test #############################

# FoR MAX TEMPERATURE #####################################################

# fixing model
ct = CADFtest(all_vals$Industry, X = all_vals$Max_temp_1, 
              type = c("trend", "drift", "none"),
         # data = list(),
         max.lag.y = 0, min.lag.X = 0, max.lag.X = 12,
         dname = NULL, 
         criterion = c("none", "BIC", "AIC", "HQC", "MAIC") )
summary(ct)

new_df = as.data.frame(cbind(all_vals$Industry, lag(all_vals$Max_temp_1, 2),
                             lag(all_vals$Max_temp_1, 6),
                             lag(all_vals$Max_temp_1, 7),
                             lag(all_vals$Max_temp_1, 9)
                             ))
new_df = na.omit(new_df)

new_model = lm(new_df$V1~.+0, data = new_df)
summary(new_model)

new_model = lm(new_df$V1~.+0, data = new_df[,-c(4)]) #less v2, which is lag 0
summary(new_model)
plot(new_model$residuals)
adf.test(new_model$residuals, k=1)
pp.test(new_model$residuals)
qqnorm(new_model$residuals)
jarque.bera.test(new_model$residuals)

# regression equation
s =  144764*new_df$V2 - 28614*new_df$V3 + 144448*new_df$V5
plot(s, type="l")

# residual analysis
adf.test(s)
pp.test(s)
kpss.test(s)




########################################

plot(all_vals$Industry, type="l", xlim=c(0, 168), xlab="Jan 2004 to Dec 2017", ylab="Beer Sales",
     col="blue")
par(new=T)
plot(all_vals$Precipitation, type="l", xlim=c(0, 168), axes=F, xlab="", ylab="", col="red")
par(new=F)
# 
plot(all_vals$Industry, all_vals$Precipitation, xlab="Industry Sales", ylab="Precipitation")
# 
# CADFtest(all_vals$Industry)
# CADFtest(all_vals$Max_temp_1)
# 
# comb1 = lm(all_vals$Industry~all_vals$Max_temp_1)
# plot(comb1$residuals, type="l", xlab="Jan 2004 to Dec 2017", 
#      ylab="Residuals of Industry Sales and Max Temp regression")
# adf.test(comb1$residuals, k=1)
# 
# 
# comb2 = lm(diff(all_vals$Industry,12)~diff(all_vals$Max_temp_1,12))
# plot(comb2$residuals, type="l", xlab="Jan 2004 to Dec 2017", 
#      ylab="Residuals of Industry Sales and Max Temp regression")
# adf.test(comb2$residuals, k=1)




















################################## ADF Test ############################################

# colnames(all_vals)
# 
# m <- lm(Industry ~ Max_temp_1 + 0, data = all_vals)
# beta <- coef(m)[1]
# 
# cat("Assumed hedge ratio is", beta, "\n")
# 
# sprd <- all_vals$Industry - beta*all_vals$Max_temp_1
# ht <- adf.test(sprd, alternative="stationary", k=0)
# 
# cat("ADF p-value is", ht$p.value, "\n")
# 
# if (ht$p.value < 0.05) {
#   cat("The spread is likely mean-reverting\n")
# } else {
#   cat("The spread is not mean-reverting.\n")
# }
