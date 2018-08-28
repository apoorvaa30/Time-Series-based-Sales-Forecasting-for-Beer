rm(list = ls())

setwd("F:/meeta/Sem Wiki Study Material/M.Mgt/Project/DATA - USa")

# install.packages("tsfa")
# install.packages("neldermead")

library("neldermead")
library("tsfa")
library("dplyr")
library("openxlsx")



########### data ################################################################

# dat = read.csv("all_vars.csv", header = TRUE)
# dat = read.csv("US_Raw_Data_FM.csv", header = TRUE)
dat = read.csv("test_tsfa.csv", header = TRUE)

# dat$year = substr(dat$year_month,1,4)
# dat$month = substr(dat$year_month,6,7)

dat = dat[,-1]
# dat = dat[1:204,9:17]

d = ncol(dat)
n = nrow(dat)





############ constants ##########################################################

# level of significance test
alpha = 0.7

# number of lags considered, arbitrary integer >= 1
p = 10

# p=4,20 gives r=8,

# initialize 
m = 1



########### formulae #############################################################

########## functions for optimization of B #######################################

########################################## 1

 get_u <- function(k, x){
  u = array()
  
  xcos = cos(x)
  xsin = sin(x)
  
  u[1] = prod(xcos[1:(k-1)])
  i = 2
  if(k>2){
    for(i in 2:(k-1)){
      u[i] = xsin[i-1]*prod(xcos[i:(k-1)])
    }
  }
  if(k>1){
    u[k] = xsin[k-1]
  }
  
  u = as.matrix(u)
  return(u)
}



########################################## 3

get_so <- function(){
  mean_vector = as.matrix(colMeans(dat))
  
  So <- array(rep(0, d*d), dim=c(d, d))
  
  sn = matrix(data = 0, nrow = d, ncol = d)
  # j=2
  for(j in 1:n){
      sn = sn + as.matrix(t(dat[j,]) - mean_vector)%*%t(as.matrix(t(dat[j,]) - mean_vector)) 
    }
  
  So[,] = sn/n
  
  return(So)
  
}


########################################## 3

Sk_calc <- function(){
    mean_vector = as.matrix(colMeans(dat))
    
    Sk <- array(rep(0, p*d*d), dim=c(d, d, p))
    
    for(j in 1:p){
      sn = matrix(data = 0, nrow = d, ncol = d)
      for(o in (j+1):n){
        sn = sn + as.matrix(t(dat[o,]) - mean_vector)%*%t(as.matrix(t(dat[(o-j),]) - mean_vector)) 
      }
      Sk[,,j] = sn/n
    }
    
    return(Sk)
  
}

########################################## 2

optim_b <- function(m, d, B){
  
  if(m==1){
    
    # finding Sk
    # Sk = Sk_calc()
    
    # finding dm
    Id = diag(x=1, nrow = d, ncol = d)
    eigens = eigen(Id)
    
    vectors= (as.matrix(eigens$vectors))
    values = matrix(eigens$values)
    values
    
    norm_list = matrix(NA, nrow=d, ncol=(d-m+1)) #list()
    j = 1; vec_count = 0
    for(i in 1:d){
      if(as.character(unlist(values[i,1])+1)=="2"){
        norm_list[,j] = vectors[,i]
        j = j+1
        vec_count = vec_count + 1
      }
    }
    
    Dm = norm_list
    
    # finding theta
    k = d-m+1
    
    theta_func = function(x){
      # x = as.matrix(rep(3,(k-1))) # size 1x(k-1), list of theta free params
      x = as.matrix(x)
      u = get_u(k, x)
      f.x.1 = 0
      for(a in 1:p){
        f.x.1 = f.x.1 + (t(Dm%*%u)%*%Sk[,,a]%*%(Dm%*%u))^2
      }
      return(f.x.1)
    }
    
    out.sphere <- optim(c(1,rep(0,(k-2))), theta_func, method = "Nelder-Mead")
      
    # finding b (Dm is dxk, u is kx1, b is dx1)
    u = get_u(k, out.sphere$par)
    b = Dm%*%u
      
    return(b)
    
  }else{
    k = d - m + 1
    k
    # finding Sk
    # Sk = Sk_calc()
    # Sk
    
    # finding dm
    Id = diag(x=1, nrow = d, ncol = d)
    Idbm = Id - ( B[,(m-1)]%*%t(B[,(m-1)]) )
    
    eigens = eigen(Idbm)
    vectors= (as.matrix(eigens$vectors)) #columns are eigen vectors
    vectors
    values = matrix(eigens$values)
    values
    
    dot_prods = t(vectors[,1:5]) %*% vectors[,1:5]
    
    dot_prods_abs = round(dot_prods, digits = 5)
    dot_prods_abs
    # dot_prods = matrix(NA, ncol = d, nrow = d)
    # for(i in 1:d){
    #  for(j in 1:d){
    #    dot_prods[i,j] = t(vectors[,i]) %*% vectors[,j]
    #  } 
    # }
    
    
    norm_list = matrix(NA, nrow=d, ncol=(d-m+1)) #list()
    j = 1; vec_count = 0
    
    for(i in 1:d){
      # print(i)
      #print(values[i,1])
      if(j==(d-m+2)){
        break
      }
      if(as.character(unlist(values[i,1])+1)=="2"){
        # print("yes")
        norm_list[,j] = vectors[,i]
        # print("in loop")
        j = j+1
        vec_count = vec_count + 1
        # print(vec_count)
      }
      
    }
    vec_count
    norm_list
    # norm_vectors = do.call(cbind, norm_list) 
    # each column is an eigen vector!!!
    Dm = norm_list
    
    # finding b and bi ##################### b is the first!!!
    # b = B[,1]
    # q = m-1
    # q
    # bi = B[,q]
    
    # finding theta
    k = d-m+1
    
    theta_func = function(x){
      x = matrix(x, ncol = (k-1) ) # size 1x(k-1), list of theta free params
      u = get_u(k, x)
      f.x.m = 0
      for(a in 1:p)
        for(i in 1:(m-1)){
          f.x.m = f.x.m + ( t(Dm%*%u)%*%Sk[,,a]%*%B[,i])^2 + (t(B[,i])%*%Sk[,,a]%*% (Dm%*%u) )^2
        }
      return(f.x.m)
    }
    
    # the optim
    if(m==d){
      
      theta_func2 = function(x){
        x = matrix(x, ncol = 1 ) # size 1x(k-1), list of theta free params
        u = get_u(k, x)
        f.x.m = 0
        for(a in 1:p)
          for(i in 1:(m-1)){
            f.x.m = f.x.m + ( t(Dm%*%u)%*%Sk[,,a]%*%B[,i])^2 + (t(B[,i])%*%Sk[,,a]%*% (Dm%*%u) )^2
          }
        return(f.x.m)
      }
      
      out.sphere <- optim(1, theta_func2, method = "Nelder-Mead")
      
    }else{
      out.sphere <- optim(c(1,rep(0,(k-2))), theta_func, method = "Nelder-Mead")
    }
    
    
    # finding bj (Dm is dxk, u is kx1, b is dx1)
    u = get_u(k, out.sphere$par)
    bj = Dm%*%u
    
    return(bj)
  }
    
} #optim_b end







########################################## 4

lb_test_stat <- function(B, m){
  lpm = 0
  bm = B[,m]
  bm
  # hu = 1
  for(a in 1:p){
    # print(lpm)
    lpm = lpm + ( (( t(bm)%*%Sk[,,a]%*%(bm) )^2)/(n-a) )
    # hu = hu*(n-a)
    # print(hu)
  }
  lpm = lpm *( n*(n+2) )
  # lpm
  return(lpm)
}




########### white noise expansion algorithm #####################################

# So
So = get_so()

# standardized dat
so_inverse = as.matrix( sqrt( abs(solve(So)) ) )
dat_pre = so_inverse %*% t(dat)
dat = as.data.frame( t(dat_pre) )


###################### loop

B = matrix(NA, nrow = d, ncol = 1)

# for m = 1
# b is first!!
m=1
# p number of sample correlation matrices - dxd
Sk <- Sk_calc()

b = optim_b(1, d, NA)
B = as.matrix(cbind(B,b))
B = as.matrix(B[,-1])
B

r = NA

test_stat = lb_test_stat(B, m)
test_stat
chi = qchisq(p = 1-alpha, df = p, lower.tail = TRUE)
# chi = qchisq(p = 1-alpha, df = p, lower.tail = FALSE)
chi

# for m > 1
if(test_stat <= chi){ #next step
  for(m in 2:d){
    # bi = as.matrix(NA, ncol = 1, nrow = d)
    print(m)
    bi = optim_b(m,d,B)
    bi
    B = as.matrix(cbind(B,bi))
    B
    test_stat = lb_test_stat(B, m)
    print(test_stat)
    # chi = qchisq(p = 1-alpha, df = (p*(2*m - 1)), lower.tail = TRUE)
    chi = qchisq(p = 1-alpha, df = p, lower.tail = TRUE)
    # chi = qchisq(p = 1-alpha, df = p, lower.tail = FALSE)
    print(chi)
    
    if(test_stat > chi){
      r = d-m+1
      B = as.matrix(B[,-m])
      break
    }else {
      if(m==d){
        r = 0
        B = diag(x=1, nrow = d, ncol = d)
      }else{
        # if(test_stat <= chi & m<d)
        # continue
      }
    }
  }

}else{ #is greater than, then terminate
  r = d - m + 1
  B = 0
}


print(r)
B



#########################################################################################











