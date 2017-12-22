# Udacity-ML

### Problem Statement
---
This project attempts to use Supervised Learning Classifier model to predict the price of Bitcoin. 

### Dataset
---
The datasets used for this project can be downloaded from [Blockchain](https://blockchain.info/stats)

The following individual csv files were downloaded housing the datasets: Volume, Volatility, Hash Rate, Mining Difficulty, Average Number of Transactions, Average Block Size, Market Capitalization, Bid/Ask Spread.

Training set has 1436 samples.
Testing set has 360 samples.

### Libraries

The libraries used for this project are os, time, functools, numpy, pandas, matplotlib, a multiplicity of sklearn modules, 

### Scripts
---
Several Python scripts are available to train the model:

* ```preprocess.py```: script containing the function that imports the csv files, formats and returns a dataframe.
* ```visuals.py```: script containing various plot functions
* ```walk_forward.py```: script that contains the walk_forward function
* ```pred.py```: script that contains the pred_dict, acc_test, f_test, and pred function 

* ```capstone_btc_project_121417```: All development Python code

Report
A final report explaining this project and the surrounding problem domain is available as * ```report.pdf```.

#### License---
---
The code for this project is open source and available under the terms of the license in this repository.
