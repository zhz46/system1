# Target Challenge - A Hybrid Recommender System 


## Background
Target wants to provide quality recommendations to guests for products they may be interested in. The task of this data science challenge is to build a model to provide product recommendations given guest id. 
- purchases.csv which contains guest item purchases (purchase date, guest id, item id, quantity purchased), and
- items.csv which contains catalog data that gives a list of anonymized discrete attributes for each item (item id, list of discrete attributes).

## Getting Started
The code was developed using Python 2.7. Before running the scripts, we first need to install python **implicit** library:
```
pip install implicit
```
Then download and put content_based.py and collaborative.py with input data sets into your work directory, and simply run:
```
python content_based.py items.csv purchases.csv
python collaborative.py purchases.csv cold_start.csv k
```
Here k is the number of product you want to recommend to guests. It has been set to less than or equal to 10. cold_start.csv 
is an intermediate output from the first command. **The prediction results are saved in recommendations.csv**.

## Problem Statement
In general, it is more difficult to implement a Recommender System than other common machine learning models. The main reason is that most of other machine learning models have a pretty unified framework: Data preprocessing/feature engineering, training and testing. For recommender system, the problems and solutions are more flexible. Many researchers have different evaluation methods to support their models. And the only golden criteria probably is to check whether your recommender systems bring more business to you or not.

This data challenge is trickier than it seems to be. The difficulty comes from three issues:
- The purchases data set only provides implicit feedback. Unlike the more extensively researched explicit feedback, we do not have any
direct input from the users regarding their preference. In fact, we are lack of solid evidence on which products guest dislike.
- The original purchases data set is very sparse. Its sparsity is around 99.8%. Generally, the maximum sparsity we could tolerate would probably be about 99.5% for collaborative filtering to work.
- The item profiles data set just contains around one half item_ids of purchased data while purchases data set also just contains around one half item_ids of item profiles data, i.e., the size of their common item_ids is not very large.

## Modelling
There are many recommender system approaches. Neither collaborative filtering nor content-based filtering could solve our problems well. As mentioned in the last section, the common item_ids in both two provided sets could only provide partial information. We do not have any feature information for some items in purchases data set. Thus, we could not calculate item feature similarities for those items. we are also not sure why some items are in the items data set but not in the purchases data set. They could be either out of stock/currently unavailable or lack of attraction. The single collaborative filtering does not work well neither due to cold start/high sparsity and implicit feedback.

Here I would use a hybrid recommender system that combines knowledge from both collaborative actions and item profiles.  The core approach I used is taken from the famous paper *Collaborative Filtering for Implicit Feedback Datasets* by *Hu et al*. Essentially the approach conducts a matrix factorization to estimate latent factors representation of guests and items. Moreover, they differentiate the concepts of unobserved observation and low preference. Instead of modelling the utility matrix directly, their approach treats the data as numbers representing the strength in observations of user actions. In our data set the transaction quantity are related to the level of confidence in observed user preferences, i.e., a multiplier of L2 loss function in optimization. 

The last issue is the cold start. Here I used a common solution. I calculated guest profiles and the cross-similarity matrix of guest profiles and item profiles. Those high similarity guest-item pairs are assigned initial values for later matrix factorization. By combining knowledge from two sources, the sparsity is lowered and the system could provide more reliable predictions.

## Evaluation
Performance evaluation for implicit feedback is not a trivial problem. We could not just randomly split data into training set and testing set as classification problems since we need all guests and items for matrix factorization. An extreme case is that your training set has half guests while your testing set has another half guests. Obviously, you could not use learned latent features for first half guests to predict preferences of remaining guests. 

A common evaluation method is to hide a certain percentage of the guest-item interactions during the training phase chosen at random. Then, check during the test phase how many of the items that were recommended the user ended up purchasing. This is the evaluation method that I used in this project.

The evaluation metric I used is expected percentile ranking suggested by the same paper mentioned above by *Hu et al* . Here each *r* is the true purchased in testing set and each *rank* is the percentile-ranking of item *i* within the predicted ordered list of all programs prepared for guest *u*. Lower values of the expected rank are more desirable, as they indicate ranking actually watched shows closer to the top of the recommendation lists. For random predictions, the expected value of this metric is 50%. Without initialization using content-based filtering, my expected rank is 43.5% on testing set and 12.5% on all set. With initialization, my expected rank becomes 7.9% on testing set and 3.6%.

![Expected Percentile Ranking](https://github.com/zhz46/target_challenge/blob/master/rank.png)

## Supplement
It takes me around one day to complete this project including literature review, coding and writing this README. If I have more time, there are many things I could improve. Firstly, due to content-based initialization, the model evaluation might be a little bit optimistic. However, this popular evaluation approach which hides part of original data is essentially biased. When we hide part of the data, those observations just become zeros in training set. Although *Hu et al*'s approach for implicit feedback reduces this biased to some extent, these zeros still influence the learned model in the training phase. I may need read more paper to find a better evaluation method. Secondly, there are a few tuning parameters in the model. Due to limited data and time I did not use any cross-validation method to tune those parameters. They were chosen by suggestions in the paper and my personal experience. Thirdly, the provided data set is not very large so I just used python to solve the problem. If we have larger data sets I would use *spark.mllib.recommendation* or optimize my codes better. Finally, there might be bugs in my codes. If it does not run well on your computer or the result is very absurd, please feel free to let me know. Thanks!
