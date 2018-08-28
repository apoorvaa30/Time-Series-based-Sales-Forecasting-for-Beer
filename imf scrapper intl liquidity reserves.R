setwd("D:/ABI/Bolivia/Data/IFS/intl liquidity rates")
rm(list=ls())

library(openxlsx)
library(xlsx)
library(dplyr)
library(lubridate)
library(reshape2)

library(imfr)

# database_id >>
# db_list = imf_ids(return_raw = FALSE, times = 30)
# code list
# code_list <- imf_codelist(database_id, return_raw = FALSE, times = 3)


database_id = 'IFS'
# IFS

# Exchange Rates selected indicators
rm(list=ls())
country = c('BO')
ind = 
# 'RAXG_USD'
# 'RAFASDR_USD'
# 'RAFAIMF_USD'
# 'RAXGFX_USD'
# 'RAFAGOLDV_OZT'
# 'RAFAGOLDNV_USD'
# 'FASAFOIL_USD'
# 'FASLFOIL_USD'
# 'FOSAF_USD'     #not
# 'FOSAFIL_USD'
# 'FOSLF_USD'     #not
# 'FOSLFIL_USD'
# 'FFSAF_USD'     #Error: No data found.
# 'FFSLF_USD'     #Error: No data found.




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

write.xlsx(ind_A, file=paste0("International Liquidity - Reserves_",ind,".xlsx"), sheetName=paste0(ind,"_Annual"), row.names=FALSE)
write.xlsx(ind_Q, file=paste0("International Liquidity - Reserves_",ind,".xlsx"), sheetName=paste0(ind,"_Quarterly"), append=TRUE, row.names=FALSE)
write.xlsx(ind_M, file=paste0("International Liquidity - Reserves_",ind,".xlsx"), sheetName=paste0(ind,"_Monthly"), append=TRUE, row.names=FALSE)

# }

