# PMM-imputation
Predictive mean matching imputation procedure based on machine learning models, using R and Python

## List of files
1. README.md - contains instructions and illustrations about everything.
2. fit_functions_classical.r - contains R code for imputation using classical methods, including Naive, Regression, PMM estimator.
3. fit_functions_deep_learning.py - contains Python code for imputation using deep learning.
4. fit_functions_machine_learning.r - contains R code for imputation using machine learning methods, including GAM, XGBOOST, KNN, and SVM.
5. toy_data.rds - contains an example dataset to test R code.

## Instruction to run the code
Here, we shared code related to our submission to JDS. A toy data was provided to test the codes.

To test the R code with the toy data, please download the r code files and toy_data.rds to a local folder.

In R:
>##Load required package, toy data, and R code files
> library(readr)
> 
> source("fit_functions_classical.r")
> 
>source("fit_functions_machine_learning.r")
> 
>fn="toy_data.rds"
>
>dat=read_rds(fn)
>
> 
>##Test classical functions, such as Regression estimator.
>Function returned two items: the estimate of variable of interest (VOI) and the individual value of VOI after imputation.
>For functions do not estimate individual value, only VOI will be returned.
>
>res=fReg(dat)
>
> 
>##Test machine learning functions, such as SVM estimator. Similarly, two items will be returned.
>
>res=f_ML(dat, "SVM")


A screenshot of running previous code is listed here:

![image](https://github.com/xu1912/PMM-imputation/assets/8320920/5e183d57-6eac-424f-8c84-18e27075a2be)
