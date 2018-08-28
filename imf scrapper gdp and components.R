setwd("D:/ABI/Bolivia/IFS/gdp and components")
rm(list=ls())

library(openxlsx)
library(xlsx)
library(dplyr)
library(lubridate)
library(reshape2)

library(imfr)

rm(list=ls())
database_id = 'IFS'
# IFS
country = c('BO')
# Trade of Goods selected indicators
ind = 
        # 'NGDP_XDC' #no monthly
        # 'NCP_XDC'  #no monthly
        # 'NCGG_XDC' #no monthly
        # 'NFI_XDC'  #no monthly
        # 'NINV_XDC' #no monthly
        # 'NX_XDC'   #no monthly
        # 'NM_XDC'   #no monthly
        # 'NSDGDP_XDC' #no quarterly
        # 'NGDP_R_K_IX' #base=2010 #no monthly
        'NGDP_D_IX'   #base=2010 #no monthly




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

write.xlsx(ind_A, file=paste0("GDP_",ind,".xlsx"), sheetName=paste0(ind,"_Annual"), row.names=FALSE)
write.xlsx(ind_Q, file=paste0("GDP_",ind,".xlsx"), sheetName=paste0(ind,"_Quarterly"), append=TRUE, row.names=FALSE)
write.xlsx(ind_M, file=paste0("GDP_",ind,".xlsx"), sheetName=paste0(ind,"_Monthly"), append=TRUE, row.names=FALSE)

# }
