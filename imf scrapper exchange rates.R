setwd("D:/ABI/Bolivia/IFS/exchange rates")
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



# IFS

# Exchange Rates selected indicators
rm(list=ls())
database_id = 'IFS'
country = c('BO')
ind = 
  # c(
# 'ENSE_XDC_XDR_RATE'
# "ENSA_XDC_XDR_RATE"
# "ENDE_XDC_USD_RATE"
# "ENDA_XDC_USD_RATE"
# "ENEER_IX"
"EREER_IX"
# )

# for(ind in ind_list){
  # for(country in country_list){
        # extracting data
        ind_A <- imf_data(database_id = 'IFS', indicator = ind,
                            country = country, freq = 'A',
                            start = 2000, end = current_year(),
                            # return_raw = TRUE,
                            print_url = TRUE)
        write.xlsx(ind_A, file=paste0(country,"_Exchange Rates",ind,".xlsx"), sheetName=paste0(ind,"_Annual"), row.names=FALSE)
        
        
        ind_Q <- imf_data(database_id = 'IFS', indicator = ind,
                            country = country, freq = 'Q',
                            start = 2000, end = current_year(),
                            # return_raw = TRUE,
                            print_url = TRUE)
        write.xlsx(ind_Q, file=paste0(country,"_Exchange Rates",ind,".xlsx"), sheetName=paste0(ind,"_Quarterly"), append=TRUE, row.names=FALSE)
        
        
        ind_M <- imf_data(database_id = 'IFS', indicator = ind,
                            country = country, freq = 'M',
                            start = 2000, end = current_year(),
                            # return_raw = TRUE,
                            print_url = TRUE)
        write.xlsx(ind_M, file=paste0(country,"_Exchange Rates",ind,".xlsx"), sheetName=paste0(ind,"_Monthly"), append=TRUE, row.names=FALSE)
      
        # }
# }

        