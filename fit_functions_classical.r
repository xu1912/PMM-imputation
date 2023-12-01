

# All Dr. Chen functions ----
# function1
# function2_5_14_2019
# gedata_test
# respoint


# Libraries ----
library(tidyverse)
pacman::p_load(
  tidyverse,
  tidymodels,
  tidyquant,
  mgcv,
  np,
  survey,
  sampling
  # torch,  # Uncomment to install! Once installed, the mlverse tabnet demo doesn't call the torch library (it's like those other libraries used in tidymodels engines)
  # tabnet
)

# library(tidyverse)
# library(tidymodels)
# library(tidyquant)
# 
# library(mgcv)     # function1
# library(np)       # function1
# 
# library(survey)   # function2_2019
# library(sampling) # gedata_test


# <><> function1 (the 6 estimators) ----
# fM_A ... A mean
# fM_B ... B mean
# fP_A ... PMIE (from sample A; uses lm())
# fKS  ... NPMIEK
# fKS1 ... (not used)
# fGAM ... NPMIEG
# fPW  ... PWE
###1. Naive estimator:
fNaive<-function(dat){
y<-dat[,1]
r<-dat[,2]
x1<-dat[,3]
x2<-dat[,4]
x3<-dat[,5]
x4<-dat[,6]
w<-dat[,7]
ry<-y[r==1]
rw<-w[r==1]

etheta<-sum(ry*rw)/sum(rw)
return(etheta)
}

###2. Regression estimator:
fReg<-function(dat){
y<-dat[,1]
r<-dat[,2]
x1<-dat[,3]
x2<-dat[,4]
x3<-dat[,5]
x4<-dat[,6]
w<-dat[,7]
ry<-y[r==1]
rx1<-x1[r==1]
rx2<-x2[r==1]
rx3<-x3[r==1]
rx4<-x4[r==1]
rw<-w[r==1]
mw<-w[r==0]

rclu<-1:length(rw)
rdat<-cbind(rclu,rx1,rx2,rx3,rx4,ry,rw)
rdat<-as.data.frame(rdat)
rdat.design <-svydesign(id = ~rclu,data = rdat,weights = ~rw)

glm1<-svyglm(ry~rx1+rx2+rx3+rx4,design=rdat.design)
ebeta<-glm1$coefficient
esm<-ebeta[1]+ebeta[2]*x1+ebeta[3]*x2+ebeta[4]*x3+ebeta[5]*x4

eN<-sum(w)
etheta<-(sum(rw*ry)+sum(mw*esm[r==0]))/eN
return(etheta)
}

###3.PMM1 estimator:
fPMM1<-function(dat){
y<-dat[,1]
r<-dat[,2]
x1<-dat[,3]
x2<-dat[,4]
x3<-dat[,5]
x4<-dat[,6]
w<-dat[,7]
ry<-y[r==1]
rx1<-x1[r==1]
rx2<-x2[r==1]
rx3<-x3[r==1]
rx4<-x4[r==1]
rw<-w[r==1]
mw<-w[r==0]

rclu<-1:length(rw)
rdat<-cbind(rclu,rx1,rx2,rx3,rx4,ry,rw)
rdat<-as.data.frame(rdat)
rdat.design <-svydesign(id = ~rclu,data = rdat,weights = ~rw)

glm1<-svyglm(ry~rx1+rx2+rx3+rx4,design=rdat.design)
ebeta<-glm1$coefficient
esm<-ebeta[1]+ebeta[2]*x1+ebeta[3]*x2+ebeta[4]*x3+ebeta[5]*x4
resm<-esm[r==1]
mesm<-esm[r==0]
nr<-length(resm)
nm<-length(mesm)

d10_0<-kronecker(mesm,resm,FUN='-')
d10_1<-matrix(d10_0,nm,nr,byrow=T)
d10_2<-abs(d10_1)
r_id<-apply(d10_2,1,order)[1,]

eN<-sum(w)
etheta<-(sum(rw*ry)+sum(mw*ry[r_id]))/eN
return(etheta)
}


# 
# # 2.0 The naive estimator from sample B ----
# # (sample mean)
fM_B <- function(dat){

  # M   <- cbind(My,  # will become dat[,1]
  #              Mx1, # will become dat[,2]
  #              Mx2, # will become dat[,3]
  #
  #              Mx3, # will become dat[,4]
  #
  #              Mx4, # will become dat[,5]
  #
  #              MsIA,# will become dat[,6]
  #              MsIB,# will become dat[,7]
  #              Mw)  # will become dat[,8]
  y       <- dat[,1]
  x1      <- dat[,2]
  sIB     <- dat[,7]
  syB     <- y[sIB==1]
  sx1B    <- x1[sIB==1]
  
  etheta1 <- mean(syB)
  etheta2 <- median(syB)
  etheta  <- c(etheta1,etheta2)

  return(etheta)
}



FRES <- function(indat,modeling_method = "GAM"){
  
  dat    <- indat[[1]]
  theta0 <- indat[[2]]
  
  if(modeling_method =="Naive"){
	res<-apply(dat,3,fNaive)
	bias<-mean(res)-theta0
	se<-sqrt(var(res))
	rmse<-sqrt(bias^{2}+se^{2})
	rbias<-bias/theta0
	rse<-se/theta0
	rrmse<-rmse/theta0
	RES<-data.frame(rbias,rse,rrmse)
  	colnames(RES) <- c('RB','RSE','RRMSE')
	return(RES)

  }

  if(modeling_method =="Reg"){
        res<-apply(dat,3,fReg)
        bias<-mean(res)-theta0
        se<-sqrt(var(res))
        rmse<-sqrt(bias^{2}+se^{2})
        rbias<-bias/theta0
        rse<-se/theta0
        rrmse<-rmse/theta0
        RES<-data.frame(rbias,rse,rrmse)
        colnames(RES) <- c('RB','RSE','RRMSE')
        return(RES)

  }

  if(modeling_method =="PMM1"){
        res<-apply(dat,3,fPMM1)
        bias<-mean(res)-theta0
        se<-sqrt(var(res))
        rmse<-sqrt(bias^{2}+se^{2})
        rbias<-bias/theta0
        rse<-se/theta0
        rrmse<-rmse/theta0
        RES<-data.frame(rbias,rse,rrmse)
        colnames(RES) <- c('RB','RSE','RRMSE')
        return(RES)
  }


  # 1. The sample mean from sample A (Mean A) ----
  
  # 6. The PMIEG ----
  # * ADD f_ML instead of fGAM ----
  res_GAM    <- apply(dat,3,f_ML,
                      modeling_method = modeling_method)
  #bias_GAM   <- res_GAM - theta0
  #m_bias_GAM <- apply(bias_GAM,1,mean)
  #rb_GAM     <- m_bias_GAM / theta0             # rb_GAM
  #var_GAM    <- apply(res_GAM,1,var)
  #se_GAM     <- sqrt(var_GAM)
  #rse_GAM    <- se_GAM / theta0                 # rse_GAM
  #mse_GAM    <- m_bias_GAM^2 + var_GAM
  #rrmse_GAM  <- sqrt(mse_GAM) / theta0          # rrmse_GAM
  #Res_GAM    <- cbind(rb_GAM,rse_GAM,rrmse_GAM) # Res_GAM

  res<-res_GAM
        bias<-mean(res)-theta0
        se<-sqrt(var(res))
        rmse<-sqrt(bias^{2}+se^{2})
        rbias<-bias/theta0
        rse<-se/theta0
        rrmse<-rmse/theta0
        RES<-data.frame(rbias,rse,rrmse)
        colnames(RES) <- c('RB','RSE','RRMSE')
  
  return(RES)
}
