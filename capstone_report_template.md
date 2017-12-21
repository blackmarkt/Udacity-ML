# Machine Learning Engineer Nanodegree
### Capstone Project
#### Predicting the Price of Bitcoin
Mark Black
<br>
December 19th, 2017
___

## I. Definition

### Project Overview
The field of finance is typically a stodgy tried and true industry where innovation is more evolutionary  than revolutionary. In the past few years the rapid rise in cryptocurrencies has ushered in a whole new digital financial ecosystem. Crytocurrencies have the potential to disrupt areas from payment processing to how we think about physical versus virtual assets. Bitcoin mania has swept the world as the price of the cryptocurrency continues to skyrocket. 

Finance professionals particularly those working on Wall Street have typically relied heavily on Excel spreadsheets, Econometrics and linear regression models for analysis. The rise of Machine Learning somewhat parallels the ascent of cryptocurriences as the businesses rapidly adopt and adjust to the era of "Big Data". This project explores the potential that Supervised Machine Learning in trying to predict the future price of Bitcoin.  

Even though it is still early innings there is already a ton of work being focused on trying to understand and predict the price of cryptoassets. Much of this project draws inspiration from the work Madan, ***et al***, where Supervised Machine Learning was used to forecast the price of Bitcoin but on shorter intraday time frames<sup>1</sup> 

This project deals with Bitcoin data on a daily time frame. Although it would have been nice and perhaps even more fruitful to use shorter time frame intervals unfortunately the availability of data for all variables played was the deciding factor. The datasets for all independent features can be found at [Blockchain.info](https://blockchain.info/) in the form of downloadable csv files. An alternative source to find the data is at [Quandl Blockchain](https://www.quandl.com/data/BCHAIN-Blockchain?keyword=)<sup>2</sup>.

### Problem Statement
Trying to predict the future price of any security or asset is central to Wall Street's ability to generate profitable trading and investment strategies. Bitcoin may have been originally intended to function as a digital payment processing system but participants have primarily been focused on it's speculative store of value. In the last year many crytocurrencies have risen more than tenfold in less than a year. Now with both the CBOE and CME introducing Bitcoin futures the motivation to forecast Bitcoin's daily price movements is a potentially lucrative endeavor<sup>3</sup>.

This project will employ a wide variety of Classification algorithms in order predict if Bitcon's price will the "up" or "down" each day in the test set. The data will be separated into sequential training and testing sets with each model being trained on the former and tested on the latter. 

From there the accuracy and f-scores for each individual model will be assessed along with it's performance as an automated trading system. Testing the profitability metrics of a trading system is known as alpha generation. At the end of the day the viability for any trading strategy is it's profiability above and beyond a buy and hold benchmark strategy (usually the annual return of the S&P500 index).

The following classifiers were selected:

- AdaBoost (Ensemble)
- Random Forest (Ensemble)
- Bagging (Ensemble)
- K-Nearest Neighbors (KNeighbors)
- Support Vector Machines (SVM)
- Logistic Regression

### Metrics
The preliminary metrics that the models will be judged against are precision, accuracy, recall and f-scores detailed below:

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

At the end of the day the name of the game in finance is to develop models or trading strategyies that beat their respective benchmarks. The final metric that the models will be judged against is whether or not employing the algorithm in a trading model beats a simle "buy and hold" strategy. The net positive excess return of an active strategy above and beyond a certain index is known as "alpha". For this project alpha the benchmark is just simply investing in Bitcoin from the start to the end of the testing period and the active trading strategies are the machine learning classifier models. The models will need to prove their superior profitability.   
___

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
As previously mentioned the source of the data for this project is blockain.info. The selection for the feature space is broken down below:

Input Feature | Continuous or Categorial | Description
--- | --- | ---
Trading Volume | Continuous | Total Daily volume in USD from all exchanges
Volaility | Continuous | Volatility is calculated as standard deviation from all market trades
Bid/Ask Spread | Continuous | Average daily spread between the Bid and Ask
Hashrate | Continuous | Average daily speed at which a computer is completing an operation in the Bitcoin code
Mining Difficulty | Continuous | How difficult it is to find a new block
Market Cap | Continuous | End of day total number of bitcoins times market value
Block Size | Continuous | Average block size (MB)
Number of Transactions | Continuous |  Total number of unique Bitcoin transactions per day

A dataframe of all the variables:

<img src="report_images/dfs.png" width="700"/>

The start and end dates were selected primarily because the dataset was wholly intact for this period with no corrupt or missing data points like NAN's or Inf's. All the data seemed to be clean and as reliable for this particular exercise. The dates for the datasets are: 

```start='2013-01-01'```
<br>
```end='2017-12-01'```
<br>

And the the training and testing sets were broken up using an 80/20 split:
<br>
```Training set has 1436 samples.```
<br>
```Testing set has 360 samples.```

### Exploratory Visualization
The datasets are fortunately intact meaning there are no missing or abnormal entries. However there are features that will need some sort of preprocessing adjustments.

Below is a plot of all the variables. Just from a cursory perspective the variables have positive slopes wtih the exception of spread. From this an expectation of data preprocssing is going to be necessary. As a result histogram plots are generated in order to analyze the distribution characteristics of each indepedent variable. 

<img src="report_images/general_plots.png" width="800"/>

From the distribution plots of the raw values it is evident that most of the variables exhibit right or positive skewness<sup>4</sup>. Skewness signals that a feature or features contain values that lie near a single number but also a smaller subset of values that are lie or are distant from the single cluster. Algorithms can be sensitive to skewed distributions and can be adversely affected to underperform if the range is not properly normalized.

<img src="report_images/dist_plots.png" width="800"/>

For the purposes of this model the independent variables will be converted according to differential or the percent change from each day to the next. This will help normalize the variables. The methods and proesses for dealing with data transformations will be detailed in the "Data Proprocessing" stage. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

### Algorithms and Techniques
Choosing the right model along with the right parameters is key to optimzing your performance metrics<sup>5</sup>. 

Below is list of the classifiers selected for this particular project along with any parameter designations and a short description which includes each algorithm's respective primary strength and weakeness:

Classifier | Parameters | Description
--- | --- | ---
AdaBoost | random_state=0 | Ensemble learner where each iteration improves prediction by weighting misclassified labels. AdaBoost Ensemble algorithms are relatively fast but can be susceptible to noise and outliers. 
Random Forest | | Ensemble learner that uses a decision tree structures to arrive at a classification determination. Random Forest modles are typically fast to train but slow to make predictions and can be computationally expensive/slow.
Bagging | random_state=0 | Ensemble learner that involves taking multiple samples from your training dataset (with replacement) and training a model for each sample. The final output prediction is averaged across the predictions of all of the sub-models. 
K-Nearest Neighbors (KNeighbors) | | Uses k surrounding labels to classify. For continuous variables Euclidean distance. KNN algorithms are east to understand and implement but at the cost of being computationally expensive
Support Vector Machines (SVM) | random_state = 0 | Iterarive algorithm that creates separation gap(s) as wide as possible. A primary advantage of SVM's is their ability to create non-linear decision boundaries and capture complex relationships in datasets but computationally suffer with complexity and as datasets become too large
Logistic Regression | random_state = 0 |Special type of regression model that uses probability to determine a categorical response. Logistic Regression models tend to be fast for small dataset with limited features but have difficulty interpreting complex relationships within the data

### Benchmark
The first performance hurdle for the classifier algorithms is to beat a Naive Bayes Predictor benchmark. A Naive predictor<sup>7</sup> is simply used to show what a base model without any intelligence or "naive" would look like. Since there is no clear benchmark or research paper to compare against the results will be benchmarked with random choice. The below Naive Predictor was generated in the report:

```Naive Predictor: [Accuracy score: 0.5579, F-score: 0.6120]```

The second hurdle will be ranking the top 3 models in terms of their accuracy and f-scores and testing their "alpha<sup>8</sup> generation" potential against the passive "buy and hold strategy". This is a higher bar for any trading model to overcome as it must not only achieve a high level of accuracy in terms of predicting the daily direction of an asset but must also exhibit a high level of precision on the days where the returns were significant.

```The benchmark for Bitcoin is 12.917352```

<img src="report_images/btc_growth.png" width="600"/>
___

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
Continuing from the Exploratory Visualization section the first preprocessing step will be to transform the raw values into differentials. This conversion involves taking the current day's value subtracting it from the previous day's value and dividing the difference by the previous day's value. In finance this percentage change is known as the "rate of return"<sup>8<sup>.
  
<img src="report_images/return.png" width="300"/>
  
The distribution plots of the features post differential conversion are below:

<img src="report_images/dist_plots.png" width="800"/>

When dealing with predicting future events it is necessary to "lag" the outcomes forward one day. This process is due to the fact that we are dealing with time and time series data sets where the features that are reported today are used to predict tomorrow's value. There is a whole branch of study for time series analysis which this project does not explore in-depth<sup9></sup>. 

The second preprocessin step is dealing with collinearity. In order to identify variable paris that exhibit a significant level of collinearity a the scatter matrix and table are generated below: 

<img src="report_images/corr_matrix_plot.png" width="700"/>

<img src="report_images/collinearity.png" width="700"/>

From these two visual cues it is evident that the pair hashrate and marketcap are highly correlated at 0.992431. There are 3 options when dealing with highly correlated variables:

1. Reduce variables
2. Combine them into a single variable
3. Do nothing

For this project the decision to remove the marketcap was chosen as highly correlated variable may overstate the effects of a single variable.

The third data proessing step is to address skewed variables. From the distribution plots (post differentialization) the 2 skewed variables are "volume" and "Bid/Ask Spread". In order to deal with skewed variables a logarithmic transformation is applied on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers.

<img src="report_images/skewed_norm_plot.png" width="800"/>

The final processing step it is good practice to normalize all the numberic continuous features. This transformation process will level the playing field for the feature space in proper prepartion for the upcoming "Implementation" phase.

Because our features only consisted of numeri continuous values there is no need for dummy variable conversions like "one-hot encoding" scheme.

Further feature reduction  will be addressed later in the project.  

Now it is time to split the data into training and testing sets. Because the data is time series is not adviseable to randomly shuffle the data but rather maintain a sequential order of division. Using the 80/20 split the dates for the training set are ```2013-01-01``` to ```2016-12-06``` and the testing set dates are ```2016-12-07``` through ```2017-12-01```. 

### Implementation
Now that the data is all prepped and split into their respective sets and the learner models all designated the implementation stage can commence.  Training, testing, predicting and acquiring the accuracy and fscores were all performed using the functions in the module ```preds.py```. Each classifier model is trained on the X_train independent features and y_train outcomes. Once each model is trained then the algorithms are each individually tested on the X_test independent feature dataset resulting in an array of binary predictions. 

Once the testing/prediction phase is complete each model's predictions are evaluated against the actual y_test outcomes. 2 hoirzontal dashed lines shows the baseline Naive Bayes Predictor threshold.  The evaluation consists of the accuracy and fscore and each model's performance is shown below:

<<img src="report_images/acc_fscore.png" width="800"/>

It may appear tht the Logistic Regression and SVM models performed best according to their accuracy and fscore but upon closer inspection this outperformance cannot be validated. The predictions for these 2 models were all "up" or 1's so we have to drop these two models as they were unable to differentiate outcomes. In fact these 2 models performance would mirror the baseline "buy and hold" strategy. For this reason the 2 algorithms must be disregarded.

Using the train_predict function a deeper analysis of the top 3 algorithms. In addition to accuracy and fscore the amount of computational time that each algorithm expended is displayed. Finally from the ```visuals.py``` module a plot of the top 3 performing classifiers is outlined:

<img src="report_images/perf_met_top3.png" width="800"/>

It is no real surprise that each model failed to generate any excess "alpha" as indicative of the across the board low accuracy and fscores. In order to calculate the return streams for each model and the Bitcoin benchmark a "walk-forward" function was created in the ```walkf_forward.py``` module. The function firstly evaluates if the prediction for the model was correct then if correct sums the day's returns with the day's starting value and if incorrect subtracts the day's returns from the starting value. Then the function iterates this process storing each end day's value until the terminal value is determined. Below is a plot of this sequence of return streams for each valid model compared against the Bitcoin benchmark:

<<img src="report_images/all_mods_wf.png" width="700"/>

### Refinement
The winning algorithm is AdaBoost Ensemble. 

The premiere step in the refinement optimization proces is model tuning. GridSearchCV will be used to help tune our Adaboost model. GridSearchCV is an exhaustive procedure that uses the parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

A number of variations of the following parameter ```parameters = {'n_estimators':[100,200,300],'learning_rate':[0.1,0.01,0.001]}``` were attempted resulting in the following best score:

<img src="report_images/tune_grid.png" width="400"/>

Ironically the tuned model's prediction was once again all "1" or "Buy" signals for every day in the testing set. This confirms that the optimal approach for Bitcoin given for this project appears to be the "Buy and Hold" strategy. 

Now that the winning model has been found the process of feature optimization can begin. The first step in improving the algorithm is feature reduction. This process involves ranking all the features in an attempt to rank their importance. From this ranked set the top number of features are selected. Below is the rankings of the top 4 significant features:

<img src="report_images/red_feat.png" width="600"/>

The primary benefit with feature reduction is to time cost savings in terms of computational efficiency at the expense of loss of performance. For this situation it appears that reducing the feature space down to only the 5 most relevant features not only increased time efficiency but also a slight increase in performance as well. Reducing the features any further only leads the model predicting all "1" and thus reverting back to the default "Buy and Hold" strategy.

With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*.

<img src="report_images/optimized_mod.png" width="320"/>

Finally the fruits of tuning and refining our model and training set are realized with a significantly improved performance from the AdaBoost Ensemble classifier model. While still underperforming the benchmark the model was still able to greatly enhance it's return stream to a final:

```Final Alpha Performance of the Optimzed Feature Set AdaBoost Classifier is 11.790559```

<<img src="report_images/fin_mod_red_wf.png" width="700"/>

___

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

AdaBoost Ensemble is a great classifier model because it is relatively fast, computationally efficient and has a intuitive iterative process that is more easily understood. 

Because of Bitcoin's inherent volatility during the time interval for this project small misclassifications in the outcomes can lead to significant under or over-performance of the model in terms of alpha. If this model were employed with a more seasoned and less historically volatile asset like US Treasuries the consequential results would not as greatly affect performance. 

### Justification
While the project fell short in overcoming the final hurdle of outperforming the "Buy and Hold" benchmark each model was able to achieve profitability. While it is a little deflating to spend so much time on to developing an underperforming model it is much worse to have built a model that loses money at the same the benchmark rises. In the past couple years the finance industry in it's race to employ data science and machine learning techniques has experienced this very tragedy<sup> </sup.

Developing unfit or underperforming models is par for the course in finance as the competition is fierce and quantitative strategies are arbitraged away as more and more players exploit the same edge. Not only would a trading strategy need to beat it's benchmark but it would need to clearly exceed it as there are cost of doing business (trading fees/commissions, infrastructure costs, data fees, etc..) that will quickly eat away profibaility. 

This is simply the first step in the "alpha" discovery process. Some obvious areas of potential improvement this project could build on are:

- more data or shorter time intervals with a lot more data
- more unique exotic datasets relating to Bitcoin
- playing with different parameter settings
- continued tuning and feature reduction 

AdaBoost was able to outperform the Naive Bayes Predictor baseline accuracy and fscore which is a win and the final tuned and reduced featured model almost kept up with the Bitcoin benchmark. This result was a little surprising given such that our accuracy and fscores were around 60%. While this model is not a certified money machine it is very promising and has definitely laid a promising foundation to build on.

___

## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_



### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

Unfortunately it appears that the strategies employed by this project are not viable as a trading system as it greatly underperformed a simple "buy and hold" approach. 

Some potential issues with this particular problem are that the data during the designated dates for Bitcoin only reflect a certain regime. Since Bitcoin's inception the price pattern trend has been only up meaning that there is an inherent positive bias to the data. If the crytocurrency space should enter a different regime the most obvious being a downtrend any model trained on data before 2018 will most likely still produce upwardly skewed predictions.

Another drawback to this project is the restrictive limitation of daily data. Unfortunately intraday data on shorter intervals for the features listed was not available or was not freely available. It would interesting to see how these classifiers perform on larger datasets. 

At the end of the day the return results were positive and quite unexpected. While very exciting and promising a more thorough stress test and vetting proess must follow in order to ensure that the algorithm is indeed legitimate. 

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

---
References:
<br>
<sup>1</sup>Madan, Saluja, Zhao"Automated Bitcoin Trading via Machine Learning Algorithms"
<br>
<sup>2</sup> A Quandl account is necessary in order to access the API or datasets
<br>
<sup>3</sup>["Duelling bitcoin futures go head-to-head as CME launches contract"](https://www.ft.com/content/877b867c-e18e-11e7-8f9f-de1c2175f5ce)
<br>
<sup>4</sup>[Skewed Distribution: Definition, Examples](http://www.statisticshowto.com/probability-and-statistics/skewed-distribution/)
<br>
<sup>5</sup>[Naive Bayes Classifier](http://www.statsoft.com/textbook/naive-bayes-classifier)
<br>
<sup>6</sup>[Machine Learning Roadmap](http://scikit-learn.org/stable/tutorial/machine_learning_map/
<br>
<sup>7</sup>[Investopedia Definition of "Alpha"](https://www.investopedia.com/terms/a/alpha.asp)
<br>
<sup>8</sup>[Rate of Return](https://en.wikipedia.org/wiki/Rate_of_return)
<br>
<sup>9</sup>[Introduction to Time Series Analysis](http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm)
<br>
<sup> </sup>[The Future Is Bumpy: High-Tech Hedge Fund Hits Limits of Robot Stock Picking](https://www.wsj.com/articles/the-future-is-bumpy-high-tech-hedge-fund-hits-limits-of-robot-stock-picking-1513007557)
