# Udacity-ML

### Problem Statement
---
This project attempts to use Supervised Learning Classifier model to predict the price of Bitcoin. 

### Dataset
---
The dataset can be downloaded primarily from [Blockchain](https://blockchain.info/stats)

Download the various csv files for the following datasets: Volume, Volatility, Hash Rate, Mining Difficulty, Average Number of Transactions, Average Block Size, Market Capitalization, Bid/Ask Spread.

Training set has 1436 samples.
Testing set has 360 samples.

### Libraries

The libraries used for this project are os, time, functools, numpy, pandas, matplotlib, a multiplicity of sklearn modules, 

### Scripts
---
Several Python scripts are available to train the model:

* ```preprocess.py```: Use this script to preprocess each of the downloaded images. This will detect the bounding boxes around the house numbers, crop out the numbers, and resize the numbers to 64x64 images.
* ```split.py```: Use this script to split the data into training, validation, and test sets.
* ```train.py```: Use this script to train the model. Training with an NVIDIA Titan X (Pascal) GPU will take approximately three days to reach 95% accuracy.
* ```eval.py```: Use this script to periodically evaluate the validation set during training. After finishing training, run with -set=test to calculate accuracy and coverage on the test set.
* ```pred.py```: Use this script to make predictions on new images in the predict directory. Images should be 64x64 jpegs. Results will be available in predict.csv.
* ```export.py```: Use this script to extract the trained model in a format suitable for use in external applications.
Two Python utility modules are used for training and evaluation.

* ```model.py```: This defines the model, optimizer, and loss functions.
* ```input.py```: This defines the image input and preprocessing data pipelines.
Report
A final report explaining this project and the surrounding problem domain is available as * ```report.pdf```.

#### License---
---
The code for this project is open source and available under the terms of the license in this repository.
