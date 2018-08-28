setwd("D:/ABI/Bolivia/Data/IFS")
rm(list=ls())

library(openxlsx)
library(xlsx)
library(dplyr)
library(lubridate)
library(reshape2)

library(imfr)



# IFS

# Trade of Goods selected indicators
rm(list=ls())
database_id = 'IFS'
country = 
  'BO'
# 'FR'
# 'RU'


ind = 
          # 'FPE_IX'
          # 'FPE_EOP_IX'
          # 'PCPI_IX'
          # 'AIP_IX'
          # 'AIP_SA_IX'
          # 'LLF_PE_NUM'
          # 'LE_PE_NUM'
          # 'LU_PE_NUM'
          'LUR_PT'




# for(ind in ind_list){
  # for(country in country_list){
      # extracting data
      ind_A <- 
        # tryCatch({
        imf_data(database_id = 'IFS', indicator = ind,
                        country = country, freq = 'A',
                        start = 2000, end = current_year(),
                        # return_raw = TRUE,
                        print_url = TRUE)
        write.xlsx(ind_A, file=paste0(country,"_CPP Indices_",ind,".xlsx"), sheetName=paste0(ind,"_Annual"), row.names=FALSE)
        
      # },
      # error = function(e){
      #   # return(NA)
      #   # do nothing
      # })
      
      
      ind_Q <- 
        # tryCatch({
        imf_data(database_id = 'IFS', indicator = ind,
                        country = country, freq = 'Q',
                        start = 2000, end = current_year(),
                        # return_raw = TRUE,
                        print_url = TRUE)
      # write.xlsx(ind_A, file=paste0(country,"_CPP Indices_",ind,".xlsx"), sheetName=paste0(ind,"_Annual"), row.names=FALSE)
      write.xlsx(ind_Q, file=paste0(country,"_CPP Indices_",ind,".xlsx"), sheetName=paste0(ind,"_Quarterly"), append=TRUE, row.names=FALSE)
      # },
      # error = function(e){
      #   # return(NA)
      #   # do nothing
      # })
      
      ind_M <- 
        # tryCatch({
        imf_data(database_id = 'IFS', indicator = ind,
                        country = country, freq = 'M',
                        start = 2000, end = current_year(),
                        # return_raw = TRUE,
                        print_url = TRUE)
        # write.xlsx(ind_A, file=paste0(country,"_CPP Indices_",ind,".xlsx"), sheetName=paste0(ind,"_Annual"), row.names=FALSE)
        write.xlsx(ind_M, file=paste0(country,"_CPP Indices_",ind,".xlsx"), sheetName=paste0(ind,"_Monthly"), append=TRUE, row.names=FALSE)
#       },
#       error = function(e){
#         # return(NA)
#         # do nothing
#       })
#       
#       rm(ind_A,ind_Q,ind_M)
#       
#   } #end country loop
# } #end indices loop


