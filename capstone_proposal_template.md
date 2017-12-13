# Machine Learning Engineer Nanodegree
## Capstone Proposal
Mark Black  
12/11/17

<img src="images/bitcoin.png" width="300"/>

## Is Bitcoin Price Predictable? ## 

### Domain Background

***What is a Bitcoin?***
Bitcoin is a cryptocurrency that was designed as a digital payments processing system that relies on crytography in order to secure transactions. There is much mythology built around Bitcoin's creation and it's creator Satoshi Nakamoto who in late 2008 authored the white paper "Bitcoin: A Peer-to-Peer Electronic Cash System"(1). Bitcoin's technological revolution is it's blockchain technology where all transactions including the creation of new bitcoins are stored and verified. Since 2008 countless cryptocurrencies known as [Altcoins](http://www.businessinsider.com/list-top-cryptocurrencies-analysis-comparison-2017-10) have emerged in the space such as Ethereum and Litecoin. Over the past couple of year Bitcoin has skyrocketed from obscurity into mainstream as it's price continues it's meteoric rise with many calling this rapid ascension a "bubble". Recently the CBOE and CME offered Bitcoin futures for trading signally the cryptocurrecncy's legitimacy as an tradeable asset. 

<img src="images/bitcoin_price_up.jpg" width="300"/>

There has been a lot of research dedicated to deciphering Bitcoin's price and trying to predict it's future value. Some interesting work has been done in trying to use sentiment analysis from social media as a predictive feature(2). Other studies have been devoted towards looking at possible tells in the technical indicators like volume or change in volume as a forecasting mechanism(3).

Is Bitcoin a "bubble"? Is it here to stay? Will global Central Banks and governments wipe out Bitcoin holders? These are the central questions that plague crytocurrencies today in their nascent stage. These questions are central for investors and potential investors who are trying to assess the long-term structural risks inherent in new asset classes. 

### Problem Statement
<img src="images/money_machine.jpg" width="300"/>

Much of finance and trading in particular is motivated by the crusade to find the next money machine and trading is no exception. So from a trading standpoint the above questions are not as interesting as trying to determine whether shorter-term opportunities exist in the price fluctuations of assets. This proposal is concerned with the questions:
<br><br>
***Is the price of Bitcoin predictable using a single or multiple factor model?*** 
<br><br>
***Can we use Machine Learning techniques to aid us in understanding and possibly forecasting the price of cryptocurrencies*** 

In financial markets trying to understand and predict the price of assets is the name of the game and the introduction of a volatile instrument like Bitcoin presents opportunity as well as excitement for traders. The report will explore the underlying potential factors such as that govern Bitcoin's price movements. Once we have identified the factors that exert the most influence we will try to train Machine Learning models that attempt to predict whether tomorrow's price will be higher "up" or "down" lower than the previous day's closing price. The best model for predicting a binary outcome is a classification algorithm and there are wide variety of different machine learning models at our disposal.

### Datasets and Inputs

Data for this project will be pulled from various data API sources. The primary source will be Quandl which offers mostly free economic and financial datasets, the St. Lious Federal Research database (FRED) and a few miscellaneous data sets pulled from a variety of sources.

**Datasets**
<br>
***Quandl:***
<br>
[Bitcoin daily price bitstampUSD](https://www.quandl.com/data/BCHARTS/BITSTAMPUSD-Bitcoin-Markets-bitstampUSD) 
<br>
[Gold Futures](https://www.quandl.com/data/CHRIS/CME_GC1-Gold-Futures-Continuous-Contract-1-GC1-Front-Month)
[M2 Money Supply (USD)](https://www.quandl.com/data/FRED/M2-M2-Money-Stock)
[Dollar Index](https://www.quandl.com/data/CHRIS/ICE_DX1-US-Dollar-Index-Futures-Continuous-Contract-1-DX1-Front-Month)

***St. Lious Fed (FRED)***
<br>
[Federal Reserve: Total Assets](https://fred.stlouisfed.org/series/WALCL)
<br>
[Bank of Japan: Total Assets](https://fred.stlouisfed.org/series/JPNASSETS)
<br>
[European Central Bank: Total Assets](https://fred.stlouisfed.org/series/ECBASSETS)
<br>

***Potential Additional Variables***
<br>
[Bitcoin trading volume breakdown:](http://data.bitcoinity.org/markets/volume/all?c=e&t=b)

Exchange Rates/Currency Cross Pairs
<br>

### Solution Statement

In order to solve the problem(s) proposed above various macro economic variables must be tested to determine if any exhibit some colinearity with the price of Bitcoin. This proposal will focus on fundamental asset data that are intuitively correlated to the price of bitcoin like other alternative stores of money (i.e. gold) to currencies (i.e. USD) to monetary indicators like "total assets" owned by the individual Central Banks of the developed world and the collective "total assets" by all 3. Once all the data has been properly vetted and prepared the separation of training and testing sets takes place. Then each model is trained, tested and evaluated according to their f-beta score.

### Benchmark Model
_(approximately 1-2 paragraphs)_

<img src="images/5050coin.jpg" width="300"/>

As mentioned above our baseline benchmark will be a 50/50 coin flip. We will build from there by testing various Supervised Learning models (listed in the Project Design) in order to discover which model or models perform best on the test set. 

Following the coin flip bar models will be ranked according to their the evaluation criteria outlined below.

### Evaluation Metrics

***F-beta Score***

<img src="images/fbeta.png" width="300"/>

***Precision, Recall, Accuracy***
<br>
<br>
**Accuracy** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).
<br>
<br>
<img src="images/accuracy.png" width="300"/>

**Precision** tells us what proportion of forecasts we classified as up, actually were up.
It is a ratio of true positives(words classified as up, and which are actually up) to all positives(all words classified as up, irrespective of whether that was the correct classification), in other words it is the ratio of

<img src="images/precision.png" width="300"/>
<br>


**Recall(sensitivity)** tells us what proportion of forecasts that actually were up were classified by us as up.
It is a ratio of true positives(words classified as up, and which are actually up) to all the words that were actually up, in other words it is the ratio of

<img src="images/recall.png" width="300"/>
<br>

### Project Design

<img src="images/architects.jpg" width="300"/>

***Data Preparation***
<br>
The first step is to import all the data from their respective API's or loaded locally. After all the data has been imported and collected some data exploratory analysis will be performed to detect any data that needs to be cleaned (i.e. NA's missing data) or if any variables need to be noramlized and convert any non-numeric variables into categorial dummy variables through the one-hot encoding process. The dependent variable (output) is then separated from the independent variables (input) variables. Once separated the dependent variable will need to be converted into a binary outcome (1 for "up" and 0 for "down").

***Model Selections and Implementation***
<br> 
Luckily there are a nice selection of Supervised Learning models at our disposable and we will select at least 3 different models from the list below:

- Gaussian Naive Bayes (GaussianNB)
- Decision Trees
- Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
- K-Nearest Neighbors (KNeighbors)
- Stochastic Gradient Descent Classifier (SGDC)
- Support Vector Machines (SVM)
- Logistic Regression

***Model Implentation***
<br>
During the model implementation phase the data is broken up into a training and a test set. The chosen models learn from the training set and tests it's predictions on the testing set.

- Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
 - Fit the learner to the sampled training data and record the training time.
 - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
 - Calculate the accuracy score for both the training subset and testing set.
 - Calculate the F-score for both the training subset and testing set.

***Model Evaluation***
<br>
And the winner is... here each model's f-beta score will be calculated and compared against the other models in order to ascertain which model performed the best.

***Tuning and Improving the Model***
<br>
In an effort to fine tune the parameters of the winning model grid search (GridSearchCV) will be employed. Using a parameter optimizer like scikitlearn's GridSearchCV will facilitate a better understanding of how the features affect or influence the decision boundary produced by the model. 

***Final Model Performance***
<br>
<img src="images/model_results.png" width="300"/>

Now that the final results have been tabulated a final performance summary report will summarize the key metrics of the best performing model and offer areas of improvement.

In addition some of the key features with the most predictive power will be explored in order to better understand their relationship wtih the target label. During this distillation process each selected features relevance will be assessed. Once the most significant features have been identified the winning model will be trained on this smaller feature subspace. This will simplify the model without hopefully compromising performance. After the smaller features subspace has been evaluated a comparison of the full set of features against the small subset of key features will be made.

---
References:
<br>
(1)[Wikipedia for Bitcoin](https://en.wikipedia.org/wiki/History_of_bitcoin)
<br>
(2)["Predicting Bitcoin price fluctuation with Twitter sentiment analysis"](http://www.diva-portal.org/smash/get/diva2:1110776/FULLTEXT01.pdf)
<br>
(2)[Can Volume Predict Bitcoin Returns and Volatility? A Quantiles-Based Approach](https://poseidon01.ssrn.com/delivery.php?ID=679064026081103078116107114017112078003010031014027056097107102011066114109071073068007100042121052122038095104004097122101106028042064064059127076021101095097005004018008075068009089064023108021085084125066117065107113028094007064026125117127091004094&EXT=pdf)
<br>

<br>
