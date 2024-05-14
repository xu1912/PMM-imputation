# PMM-imputation
Predictive mean matching imputation procedure based on machine learning models, using R and Python

## List of files
1. Simulaiton_data_used_in_paper - folder contains simulation datasets used in paper.
   - dat_n500_m1.rds - R data file contains generated data for 200 times simulation of Model 1 and the true value of interest.
   - dat_n500_m2.rds - R data file contains generated data for 200 times simulation of Model 2 and the true value of interest.
   - dat_n500_m3.rds - R data file contains generated data for 200 times simulation of Model 3 and the true value of interest.
   - dl_prep.r - R code to convert RDS files to csv files for deep learning using Python.
   - est_function.r - R code for imputation using machine learning methods, including GAM, XGBOOST, KNN, and SVM.
   - fit.r - R code for imputation using classical methods, including Naive, Regression, PMM estimator.
   - fit_functions_deep_learning_all_datasets.py - Python code for processing simulated data of a model.
   - getdata.r - R code contains functions to generate simulation data.
   - run_data_generation.r - R code to implment getdata.r and physically generate the data.
   - run_simulation.r - R code to implment est_function.r and fit.r to analyze generated rds file.
3. README.md - contains instructions and illustrations about everything.
4. fit_functions_classical.r - contains R code for imputation using classical methods, including Naive, Regression, PMM estimator.
5. fit_functions_deep_learning.py - contains Python code for imputation using deep learning.
6. fit_functions_machine_learning.r - contains R code for imputation using machine learning methods, including GAM, XGBOOST, KNN, and SVM.
7. toy_data.rds - contains an example dataset to test R code.
8. toy_data.csv - contains an example dataset to test Python code.

## Instruction to run the R code
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

## Instruction to run the Python code
To test the Python code with the toy data, please download the python code files and toy_data.csv to a local folder.

Open a command line with Python, Keras, and TensorFlow installed. 

Simply run:
> python.exe .\fit_functions_deep_learning.py

Screenshot of output:
>![image](https://github.com/xu1912/PMM-imputation/assets/8320920/57ee659f-0ad3-4349-82f2-21fbc40bee43)

After certain epochs:
> ![image](https://github.com/xu1912/PMM-imputation/assets/8320920/524b2c26-6c21-464c-bd41-77ff86138801)

The existing code was tested in PowerShell Win10 v1809, Python v3.8.3, and R v4.0.3.
