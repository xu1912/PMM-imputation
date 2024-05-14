library(readr)
# For model 1, create a folder to save files
dat1 <- read_rds("dat_n500_m1.rds")
if (!file.exists("dat_n500_m1")){
    dir.create("dat_n500_m1")  
}
# One file for each simulation
for(i in 1:200){
	dat=dat1[[1]][,,i]
	write.csv(dat,paste("dat_n500_m1/dt_", i, ".csv", sep=""), row.names=F)
}
# Show the true value of VOI
dat1[[2]]

# For model 2, create a folder to save files
dat2 <- read_rds("dat_n500_m2.rds")
if (!file.exists("dat_n500_m2")){
    dir.create("dat_n500_m2")  
}
for(i in 1:200){
	dat=dat2[[1]][,,i]
	write.csv(dat,paste("dat_n500_m2/dt_", i, ".csv", sep=""), row.names=F)
}
dat2[[2]]

# For model 3, create a folder to save files
dat3 <- read_rds("dat_n500_m3.rds")
if (!file.exists("dat_n500_m3")){
    dir.create("dat_n500_m3")  
}
for(i in 1:200){
	dat=dat3[[1]][,,i]
	write.csv(dat,paste("dat_n500_m3/dt_", i, ".csv", sep=""), row.names=F)
}
dat3[[2]]