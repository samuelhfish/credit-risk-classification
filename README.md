# credit-risk-classification
Credit risk classification exercise using supervised machine learning and a logistic regression model.


# Overview
The purpose of this analysis was to use factors such as "loan size", "interest rate", "borrower income", "income-to-debt ratio", "number of accounts" and "total debt" and train a model to predict whether the loan was more likely to be healthy or more likely to be high risk. In our sample data of 77,500 loans, roughly 97% were healthy and the remaining 3% deemed high-risk.

For the machine learning analysis, we imported the lending data, seperated the loan status which would be our labels/y variable and dropped that from our features/x variable dataframe. We then used sklearn and the train_test_split function to split the data into training and testing groups. Then we used a logistic regression model to fit the data and predict binary classification.

For our final step, we used an accuracy score, confusion matrix and classification report to judge the effectiveness of our model.

- Accuracy Sore: 95.2%

<img width="751" alt="Screenshot 2023-10-11 at 1 46 07 PM" src="https://github.com/samuelhfish/credit-risk-classification/assets/125224990/a060e348-2b1c-4a8e-a125-bd5d9076694b"><br>

With a total score of 95.2% this is how often the model was correct, or the ratio of the correctly predicted observations to the total observations.


- Prcesion Score
Healthy: 100%
High-Risk: 85%


The precision score relates to the number of correctly predicted positive observations among all positive instances, or false positive instances that were predicted as positive that should have been negative.


- Recall Score
Healthy: 99%
High-Risk: 91%

The recall score relates to the ratio of predicted positives and number of positive instances but takes into account the number of results that were false negatives, or predicted negative and should have been positive.

<img width="725" alt="Screenshot 2023-10-11 at 1 46 57 PM" src="https://github.com/samuelhfish/credit-risk-classification/assets/125224990/41a9960d-fbc4-40fa-b4d2-97a2af3420e7"><br>

<img width="724" alt="Screenshot 2023-10-11 at 1 47 32 PM" src="https://github.com/samuelhfish/credit-risk-classification/assets/125224990/091380b6-5be6-449c-ab7f-a0d58e7aa674"><br>


Because of the binary nature of the data, a logistic regression model makes sense to use. In practicee for this exercise, the logistic regression model did a good job of predicting the loan status. However, it seems to do a much better job at predicting a healthy loan from a high-risk loan. A possible explanation for this is that healthy loans frequently have many characteritics in common, whereas an at risk loan might have a variety of factors that contribute to it's riskiness making it harder to predict.