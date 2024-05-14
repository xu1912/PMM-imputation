rpath <- "./";
setwd (rpath);

pacman::p_load(
  tidyverse,
  tidyquant,
  # tidymodels,
  plotly,
  gt
)

# Source scripts ----
source("./fit.r")
source("./est_function.r")

# Model number
mn=2

# Sample size 
smpn=500

# Load data
fn=paste("dat_n",smpn,"_m", mn,".rds", sep="")
dat <- read_rds(fn)

# Run each method
start_time <- Sys.time()
res=FRES(dat, modeling_method="Naive")
end_time <- Sys.time()
rt=difftime(end_time, start_time, units="mins")
# Save running time
write.table(rt, paste("./m",mn,"_n",smpn,"_naive.csv", sep=""),row.name=F,col.names=F,quote=F,sep=",")
# Save result
res %>% write_rds(paste("./m",mn,"_n",smpn,"_naive.rds", sep=""))
res=NULL

start_time <- Sys.time()
res=FRES(dat, modeling_method="Reg")
end_time <- Sys.time()
rt=difftime(end_time, start_time, units="mins")
write.table(rt, paste("./m",mn,"_n",smpn,"_reg.csv", sep=""),row.name=F,col.names=F,quote=F,sep=",")
res %>% write_rds(paste("./m",mn,"_n",smpn,"_reg.rds", sep=""))
res=NULL

start_time <- Sys.time()
res=FRES(dat, modeling_method="PMM1")
end_time <- Sys.time()
rt=difftime(end_time, start_time, units="mins")
write.table(rt, paste("./m",mn,"_n",smpn,"_pmm1.csv", sep=""),row.name=F,col.names=F,quote=F,sep=",")
res %>% write_rds(paste("./m",mn,"_n",smpn,"_pmm1.rds", sep=""))
res=NULL

start_time <- Sys.time()
res <- FRES(dat, modeling_method = "GAM")
end_time <- Sys.time()
rt=difftime(end_time, start_time, units="mins")
write.table(rt, paste("./m",mn,"_n",smpn,"_gam.csv", sep=""),row.name=F,col.names=F,quote=F,sep=",")
res %>% write_rds(paste("./m",mn,"_n",smpn,"_gam.rds", sep=""))
res=NULL

start_time <- Sys.time()
res <- FRES(dat, modeling_method = "XGBOOST")
end_time <- Sys.time()
rt=difftime(end_time, start_time, units="mins")
write.table(rt, paste("./m",mn,"_n",smpn,"_xgboost.csv", sep=""),row.name=F,col.names=F,quote=F,sep=",")
res %>% write_rds(paste("./m",mn,"_n",smpn,"_xgboost.rds", sep=""))
res=NULL

start_time <- Sys.time()
res <- FRES(dat, modeling_method = "SVM")
end_time <- Sys.time()
rt=difftime(end_time, start_time, units="mins")
write.table(rt, paste("./m",mn,"_n",smpn,"_svm.csv", sep=""),row.name=F,col.names=F,quote=F,sep=",")
res %>% write_rds(paste("./m",mn,"_n",smpn,"_svm.rds", sep=""))
res=NULL

start_time <- Sys.time()
res <- FRES(dat, modeling_method = "KNN")
end_time <- Sys.time()
rt=difftime(end_time, start_time, units="mins")
write.table(rt, paste("./m",mn,"_n",smpn,"_knn.csv", sep=""),row.name=F,col.names=F,quote=F,sep=",")
res %>% write_rds(paste("./m",mn,"_n",smpn,"_knn.rds", sep=""))
res=NULL
