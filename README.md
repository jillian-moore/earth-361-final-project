# Dengue Forecasting and Machine Learning Analysis

This project analyzes Dengue fever transmission dynamics using exploratory data analysis, numerical modeling, regression models, and machine learning approaches.  
The dataset originates from public Dengue surveillance records for San Juan and Iquitos.

## What's in the repo

#### Sub-folders/directories
-   [data/](./data/): contains all data for this project, including the raw CSV files and processed datasets  
-   [figures/](./figures/): contains all plots and visualizations used in the analysis and final report  
-   [results/](./results/): contains exported model outputs, evaluation metrics, and LaTeX-ready tables  

#### Scripts

-   `1_data_clean.py`: reads, cleans, and preprocesses data; produces processed datasets used throughout the project  
-   `2_numerical_modeling.py`: implements a SEI–SEIR numerical model to examine mechanistic transmission dynamics  
-   `3_data_analysis.py`: conducts exploratory data analysis (EDA), including distribution plots, correlations, lag analysis, and seasonality patterns  

###### Regression models (`4a`–`4d`)
-   `4a_regression.py`: simple linear regression model  
-   `4b_regression.py`: interaction-term regression  
-   `4c_regression.py`: autoregressive regression using lagged case counts  
-   `4d_regression.py`: 12-week lagged regression model  

All scripts generate performance metrics and diagnostic plots saved in `results/` and `figures/`.

###### Machine learning models (`5a`–`5e`)
-   `5a_ml_bt_reg.py`: boosted tree regression (XGBoost engine)  
-   `5b_ml_bt_class.py`: boosted tree classification  
-   `5c_ml_rf_reg.py`: random forest regression  
-   `5d_ml_rf_class.py`: random forest classification  
-   `5e_ml_en_reg.py`: elastic net regression  

Models include hyperparameter tuning with repeated cross-validation, feature importance analysis, and prediction output files saved in `results/` and `figures/`.

###### Model Analysis
-   `6a_model_eval.py`: regression models evaluation 
-   `6b_model_eval.py`: machine learning models evaluation  

#### Additional files
-   `bibliography.bib`: references for the final report  
-   `.gitignore`: excludes temporary files, Python cache, LaTeX build files, and large data  
-   `README.md`: documentation for the project
