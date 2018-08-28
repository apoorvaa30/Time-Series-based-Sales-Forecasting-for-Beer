setwd("D:/ABI/Bolivia/Data/IFS/interest rates")
rm(list=ls())

library(openxlsx)
library(xlsx)
library(dplyr)
library(lubridate)
library(reshape2)

library(imfr)

rm(list = ls())
database_id = 'IFS'
# IFS
# Interest Rates selected indicators
country = c('BO')
ind =
# 'FPOLM_PA'
# 'FID_PA'
# 'FIR_PA' #not
# 'FIRA_PA'
# 'FIMM_PA'
# 'FIMM_FX_PA' #not able to download
# 'FITB_PA'
# 'FISR_PA'
# 'FISR_FX_PA' #not
# 'FIDR_PA'
# 'FIDR_FX_PA' #not
# 'FILR_PA'
# 'FILR_FX_PA' #not
'FIGB_PA' #not

# )


# for(ind in ind){
# extracting data
  # try(
    ind_A <- imf_data(database_id = 'IFS', indicator = ind,
                      country = country, freq = 'A',
                      start = 2000, end = current_year(),
                      # return_raw = TRUE,
                      print_url = TRUE)
    
    write.xlsx(ind_A, file=paste0("Interest Rates_",ind,".xlsx"), sheetName=paste0(ind,"_Annual"), row.names=FALSE)
  # )
  # try(
    ind_Q <- imf_data(database_id = 'IFS', indicator = ind,
                      country = country, freq = 'Q',
                      start = 2000, end = current_year(),
                      # return_raw = TRUE,
                      print_url = TRUE)
    
    write.xlsx(ind_Q, file=paste0("Interest Rates_",ind,".xlsx"), sheetName=paste0(ind,"_Quarterly"), append=TRUE, row.names=FALSE)
  # )
  # try(
  ind_M <- imf_data(database_id = 'IFS', indicator = ind,
                    country = country, freq = 'M',
                    start = 2000, end = current_year(),
                    # return_raw = TRUE,
                    print_url = TRUE)
  
  write.xlsx(ind_M, file=paste0("Interest Rates_",ind,".xlsx"), sheetName=paste0(ind,"_Monthly"), append=TRUE, row.names=FALSE)
  # )

# }
