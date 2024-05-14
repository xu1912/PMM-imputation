## This folder contains code and data used in the manuscript.

### R code for classical and machine learning based methods
The three rds files were generated using run_data_generation.r, which called getdata.r. Please open run_data_generation.r to cinfigure and run. The generated data may be different from what we had, due to random generation procedure.

Open and run run_simulation.r to process the generated dataset using classical and machine learning based methods, which were implemented in R.

### Python code for deep learning methods
For deep learning method, we used python to implement. Before running python code, we needed to convert rds files to csv files, which was more convinent for running Python code. Please open and run dl_prep.r.

After that, simply open a command line and run:
> python.exe .\fit_functions_deep_learning_all_datasets.py

Or a python editor can be used to open and run.

