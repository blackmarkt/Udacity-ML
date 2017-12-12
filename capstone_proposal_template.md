# Machine Learning Engineer Nanodegree
## Capstone Proposal
Mark Black  
12/11/17

<img src="images/bitcoin.png" width="300"/>

## Is Bitcoin Price Predictable? ## 

### Domain Background
references: https://en.wikipedia.org/wiki/History_of_bitcoin

***What is a Bitcoin?***
Bitcoin is a cryptocurrency that was designed as a digital payments processing system that relies on crytography in order to secure transactions. There is much mythology built around Bitcoin's creation and it's creator Satoshi Nakamoto who in late 2008 authored the white paper "Bitcoin: A Peer-to-Peer Electronic Cash System". Bitcoin's technological revolution is it's blockchain technology where all transactions including the creation of new bitcoins are stored and verified. Since 2008 countless cryptocurrencies known as [Altcoins](http://www.businessinsider.com/list-top-cryptocurrencies-analysis-comparison-2017-10) have emerged in the space such as Ethereum and Litecoin. Over the past couple of year Bitcoin has skyrocketed from obscurity into mainstream as it's price continues it's meteoric rise with many calling this rapid ascension a "bubble". Recently the CBOE and CME offered Bitcoin futures for trading signally the cryptocurrecncy's legitimacy as an tradeable asset. 

<img src="images/bitcoin_price_up.jpg" width="300"/>

Is Bitcoin a "bubble"? Is it here to stay? Will global Central Banks and governments wipe out Bitcoin holders? These are the central questions that plague crytocurrencies today in their nascent stage. These questions are central for investors and potential investors who are trying to assess the long-term structural risks inherent in new asset classes. 
 
Misc Ideas:
Silk Road

### Problem Statement
<img src="images/money_machine.jpg" width="300"/>

Much of finance and trading in particular is motivated by the crusade to find the next money machine and trading is no exception. So from a trading standpoint the above questions are not as interesting as trying to determine whether shorter-term opportunities exist in the price movements of assets. This proposal is concerned with the questions:
<br><br>
***Is the price of Bitcoin predictable using a single or multiple factor model?*** 
<br><br>
***Can we use Machine Learning techniques to aid us in understanding and possibly forecasting the price of cryptocurrencies*** 

In financial markets trying to understand and predict the price of assets is the name of the game and the introduction of a volatile instrument like Bitcoin presents opportunity as well as excitement for traders. The report will explore the underlying potential factors that govern Bitcoin's price movements. Once we have identified the factors that exert the most influence we will try to train Machine Learning models that attempt to predict whether the tomorrow's price will be higher or lower than the previous day's closing price. In order to determine whether our model adds any informtional value we will compare our results against the  "coin flip test" with a benchmark of 50% or simply guessing.

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

**Datasets**
<br>
***Quandl:***
<br>
[Bitcoin daily price bitstampUSD](https://www.quandl.com/data/BCHARTS/BITSTAMPUSD-Bitcoin-Markets-bitstampUSD) 
<br>
[Gold Futures](https://www.quandl.com/data/CHRIS/CME_GC1-Gold-Futures-Continuous-Contract-1-GC1-Front-Month)
[M2 Money Supply (USD)](https://www.quandl.com/data/FRED/M2-M2-Money-Stock)
Velocity of Money (USD):

***St. Lious Fed (FRED)***
<br>
[Federal Reserve: Total Assets](https://fred.stlouisfed.org/series/WALCL)
<br>
[Bank of Japan: Total Assets](https://fred.stlouisfed.org/series/JPNASSETS)
<br>
[European Central Bank: Total Assets](https://fred.stlouisfed.org/series/ECBASSETS)
<br>
<br>
[Bitcoin trading volume breakdown:](http://data.bitcoinity.org/markets/volume/all?c=e&t=b)

Exchange Rates/Currency Cross Pairs
[Dollar Index](https://www.quandl.com/data/CHRIS/ICE_DX1-US-Dollar-Index-Futures-Continuous-Contract-1-DX1-Front-Month)

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement

In order to solve the problem(s) proposed above we must test various macro economic variables that may exhibit some colinearity with the price of Bitcoin. We will first look at assets or variables that would intuitively be correlated to the price of bitcoin like other alternative stores of money (i.e. gold) to currencies (i.e. USD) to monetary indicators like M2 Money supply. Once we have settled on a set of independent variables we can then train various models and test our model(s) on cross validated datasets. At the end of the day our model will be evaluated on it's ability to perform better than the 50/50 "coin flip" benchmark.

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
