pacman::p_load(
  tidyverse,
  tidyquant,
  # tidymodels,
  plotly,
  gt
)


# Source scripts ----
source("getdata.r")

# 0.0 - Generate/load sim. data ----
## NUmber of Simulation: B = 200 ----
dat1_500 <- gedata(B = 200,N = 20000,n = 500, id_m = 1, id_resp=1, id_design=1, rou=0) # linear
dat1_500 %>% write_rds("dat_n500_m1.rds")

dat2_500 <- gedata(B = 200,N = 20000,n = 500, id_m = 2, id_resp=1, id_design=1, rou=0) # interactions
dat2_500 %>% write_rds("dat_n500_m2.rds")

dat3_500 <- gedata(B = 200,N = 20000,n = 500, id_m = 3, id_resp=1, id_design=1, rou=0) # multi-layer
dat3_500 %>% write_rds("dat_n500_m3.rds")
