setwd("D:/ABI/Bolivia/Data/IFS/trade of goods")
rm(list=ls())

library(openxlsx)
library(xlsx)
library(dplyr)
library(lubridate)
library(reshape2)

library(imfr)

database_id = 'IFS'
# IFS

# Trade of Goods selected indicators
rm(list=ls())
country = c('BO')
ind = 
# 'TXG_FOB_USD'
# 'TXG_FOB_XDC'
# 'TMG_CIF_USD'
# 'TMG_CIF_XDC'
# 'TMG_FOB_XDC'
# 'TMG_FOB_USD'
# 'TXG_R_FOB_IX'
# 'TMG_R_CIF_IX'



# for(ind in ind_list){
# extracting data
ind_A <- imf_data(database_id = 'IFS', indicator = ind,
                  country = country, freq = 'A',
                  start = 2000, end = current_year(),
                  # return_raw = TRUE,
                  print_url = TRUE)

ind_Q <- imf_data(database_id = 'IFS', indicator = ind,
                  country = country, freq = 'Q',
                  start = 2000, end = current_year(),
                  # return_raw = TRUE,
                  print_url = TRUE)

ind_M <- imf_data(database_id = 'IFS', indicator = ind,
                  country = country, freq = 'M',
                  start = 2000, end = current_year(),
                  # return_raw = TRUE,
                  print_url = TRUE)

write.xlsx(ind_A, file=paste0("Trade of Goods_",ind,".xlsx"), sheetName=paste0(ind,"_Annual"), row.names=FALSE)
write.xlsx(ind_Q, file=paste0("Trade of Goods_",ind,".xlsx"), sheetName=paste0(ind,"_Quarterly"), append=TRUE, row.names=FALSE)
write.xlsx(ind_M, file=paste0("Trade of Goods_",ind,".xlsx"), sheetName=paste0(ind,"_Monthly"), append=TRUE, row.names=FALSE)

# }
