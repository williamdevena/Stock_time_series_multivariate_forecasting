[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=9481312&assignment_repo_type=AssignmentRepo)
# Data Acquisition and Processing Systems (DAPS) (ELEC0136)


![My Image](timeseries_forecast.gif)

This repo contains the code and report (**Report_DAPS.pdf**) produces for the assignment of Data Acquisition and Processing Systems (DAPS) module.
More details about the assignment in the file **"DAPS Assignment_2022_2023.pdf"**.

## How to run the code
To run the code, just execute the file main.py.

The code will create three folders:

- data: contains all the the raw data acquired
- plots: contains all the plots produced duting the execution
- project_log: contains the file assingment.log, that contains all the logs of the execution and the final metrics at the end

The plots folder is going to have 4 folders, one for every type of data acquired: Covid-19 data, technical indicators and economic indicators (fundamental and macro-economical indicators). Inside everyone of these folders there is going to be 4 images for every variable: one image contains the line plot and the distribution plot (histogram and kde), one contains the four plots of the seasonal decomposition(series, trend, seasonality and residuals), one contains two plots that show the outliers detected with two different methods (IQR and Z-score) and finally the last shows the autocorrelation plot. Furthermore, the lot folder contains also the inference plot (univariate and multivariate): one image shows the inference and one show the residuals.

## Repository structure
All the code is in the main.py file and in the src folder. The src folder contains the following files:

- **auth.py**: contains credentials for the MongoDB database and BLS API
- **costants.py**: contains costants that are been used inside the all repository
- **data_acquisition.py**: contains the functions used to acquire data both locally and remotely
- **data_inference.py**: contains all the function related to the inference stage
- **data_preparation.py**: contains all the functions used preprocess and prepare the data
- **data_storing.py**: contains the functions to store data both locally and
remotely into a MongoDB database
- **data_visualization.py**: contains the functions to visualize data in different forms
- **logging_utilities.py**: contains the functions used to log information
- **seeds.py**: contains all the functions used to set the seeds for the package, to make the experiments reproducible.


