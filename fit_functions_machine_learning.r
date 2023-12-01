

# f_ML with 4 and with 21 ----

# Recall that in the xgboost package manual they mention there are 'Linear Booster'
# parameters, called lambda, lambda_bias, and alpha. These are the hyperparameters for
# L2 regularization on weights, L2 regularization on bias, and L1 regularization on 
# weights, respectively (L1 regularization on bias is not important). Dr. Xu calls 
# them penalties. 
# There is not currently support for tuning these in tidymodels. Apparently Julia Silge 
# is waiting for more interest in adding that. Darn.
# The defaults for all of these is 0. These three hyperparameters are not listed in 
# the options for xgboost in tidymodels:
# show_model_info("boost_tree")


f_ML <- function(dat,modeling_method){
  
    
    # M   <- cbind(My,  # will become dat[,1]
    #              Mx1, # will become dat[,2]
    #              Mx2, # will become dat[,3]
    #              
    #              # ADD Mx3 ----
    #              Mx3, # will become dat[,4]
    #              
    #              # ... ADD Mx4  ----
    #              Mx4, # will become dat[,5]
    #              
    #              MsIA,# will become dat[,6]
    #              MsIB,# will become dat[,7]
    #              Mw)  # will become dat[,8] 
  
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

mx1<-x1[r==0]
mx2<-x2[r==0]
mx3<-x3[r==0]
mx4<-x4[r==0]
mw<-w[r==0]

datB   <- cbind(ry,rx1,rx2,rx3,rx4)
datB2  <- as.data.frame(datB)
datB2_tbl <- datB2 %>% as_tibble()

datXA  <- cbind(x1,x2,x3,x4)
datXA2 <- as.data.frame(datXA)
colnames(datXA2) <- c('rx1','rx2','rx3','rx4')
datXA2_tbl <- datXA2 %>% as_tibble()

	common_recipe <- recipe(
      ry ~ ., data = datB2_tbl
    )

	if (modeling_method == "GAM") {
      fit <- gam(ry ~ s(rx1) + s(rx2) + s(rx3) + s(rx4), # ... GAM's x4 ----
                 data = datB2)
    }
    
        
    if (modeling_method == "XGBOOST") {
      
      # * 1) model spec xgboost ----
      boost_spec <- boost_tree(
        tree_depth = tune(),
        trees = tune(),              # add tune() here as well - 20, 50, 100, 200, 300, etc.
        learn_rate = tune(),
        mtry = tune(),
        min_n = tune(),
        #loss_reduction = tune(),
        sample_size = 0.8
      ) %>% 
        set_mode("regression") %>% 
        set_engine("xgboost")
      
      # * 2) workflow ----
      boost_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(boost_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)
      
      # * 4) tune_grid() ----
      boost_cube <- grid_latin_hypercube(
        tree_depth(),
        learn_rate(),
	  trees(),
        finalize(mtry(), datB2_tbl),
        min_n(),        
        size = 30
      )
      
      set.seed(123)
      boost_grid <- tune_grid(
        boost_workflow,
        resamples = dat_folds,
        grid = boost_cube,
        control = control_grid(save_pred = T)
      )
      
      # * 5) finalize_workflow() ----
      final_boost_workflow <- boost_workflow %>% 
        finalize_workflow(select_best(boost_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_boost_workflow) %>% 
        fit(ry ~ ., data = datB2_tbl)
      
    }

    if (modeling_method == "KNN") {
      
      # * 1) model spec xgboost ----
      knn_spec <- nearest_neighbor(
        neighbors = tune(),
        weight_func = tune(),        
        dist_power = tune()
      ) %>% 
        set_mode("regression") %>% 
        set_engine("kknn")
      
      # * 2) workflow ----
      knn_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(knn_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)
      
      # * 4) tune_grid() ----
      knn_cube <- grid_latin_hypercube(
        neighbors(),
        dist_power(),
	  weight_func(),       
        size = 30
      )
      
      set.seed(123)
      knn_grid <- tune_grid(
        knn_workflow,
        resamples = dat_folds,
        grid = knn_cube,
        control = control_grid(save_pred = T)
      )
      
      # * 5) finalize_workflow() ----
      final_knn_workflow <- knn_workflow %>% 
        finalize_workflow(select_best(knn_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_knn_workflow) %>% 
        fit(ry ~ ., data = datB2_tbl)
      
    } 
    
    if (modeling_method == "SVM") {
      
      # ... ----
      # * 1) model spec SVM ----
      svm_spec <- svm_rbf(
        cost = tune(),
        rbf_sigma = tune(),
        margin = tune()
      ) %>% 
        set_mode("regression") %>% 
        set_engine("kernlab")

      
      # * 2) workflow ----
      svm_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(svm_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl) # default v=10
      
      # * 4) tune_grid() ----
      set.seed(123)
      svm_cube <- grid_latin_hypercube(
        cost(),
        rbf_sigma(),
        svm_margin(), 
        size = 30
      )

      svm_grid <- tune_grid(
        svm_workflow,
        resamples = dat_folds,
        grid = svm_cube # SVM doesn't take long at grid = 20 (just 4 min)
      )

      #autoplot(svm_grid, metric = "rmse") 

      # * 5) finalize_workflow() ----
      final_svm_workflow <- svm_workflow %>% 
        finalize_workflow(select_best(svm_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_svm_workflow) %>% 
        fit(ry ~ ., data = datB2_tbl)
      
    } 

esm<-predict(fit,datXA2)
if(modeling_method == "GAM"){
	resm=esm[r==1]
	mesm=esm[r==0]
}else{
	resm<-esm$.pred[r==1]
	mesm<-esm$.pred[r==0]
}
nr<-length(resm)
nm<-length(mesm)


d10_0<-kronecker(mesm,resm,FUN='-')
d10_1<-matrix(d10_0,nm,nr,byrow=T)
d10_2<-abs(d10_1)
r_id<-apply(d10_2,1,order)[1,]

eN<-sum(w)
etheta<-(sum(rw*ry)+sum(mw*ry[r_id]))/eN
return(list(etheta,esm))
}
