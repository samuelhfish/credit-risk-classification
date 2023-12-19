# credit-risk-classification
Credit risk classification exercise using supervised machine learning and a logistic regression model.


### Overview
The purpose of this analysis was to use factors such as "loan size", "interest rate", "borrower income", "income-to-debt ratio", "number of accounts" and "total debt" and train a model to predict whether the loan was more likely to be healthy or more likely to be high risk. In our sample data of 77,500 loans, roughly 97% were healthy and the remaining 3% deemed high-risk.

For the machine learning analysis, we imported the lending data, seperated the loan status which would be our labels/y variable and dropped that from our features/x variable dataframe. We then used sklearn and the train_test_split function to split the data into training and testing groups. Then we used a logistic regression model to fit the data and predict binary classification.

For our final step, we used an accuracy score, confusion matrix and classification report to judge the effectiveness of our model.

Full code exsists in 'credit_risk_classification.ipynb' notebook. Credit_Risk > credit_risk_classification.ipynb

Data exists in 'lending_data.csv' file. Credit_Risk > Resources > lending_data.csv

```
After neccesarry imports we have our loan data DataFramee, df_data.
```
![Screenshot 2023-12-18 at 11 26 02 PM](https://github.com/samuelhfish/credit-risk-classification/assets/125224990/d0d73fae-6c61-467e-ada0-a0b6cfe87a22)

```python
# Separate the data into labels and features
# Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.

# Separate the y variable, the labels
y = df_data["loan_status"]

# Separate the X variable, the features
X = df_data.drop(columns="loan_status")
```
```python
# Review the y variable Series
y
```
```
0        0
1        0
2        0
3        0
4        0
        ..
77531    1
77532    1
77533    1
77534    1
77535    1
Name: loan_status, Length: 77536, dtype: int64
```
```python
# Review the X variable DataFrame
X
```
![Screenshot 2023-12-18 at 11 30 46 PM](https://github.com/samuelhfish/credit-risk-classification/assets/125224990/08b6a3f5-36be-4feb-bc7a-45a593d64ee2)
```python
# Check the balance of our target values
df_data.loan_status.value_counts()
```
```
loan_status
0    75036
1     2500
Name: count, dtype: int64
```

```python
# Import the train_test_learn module
from sklearn.model_selection import train_test_split

# Split the data using train_test_split
# Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```
```python
# Create a Logistic Regression Model with the Original Data
# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
logistic_regression_model = LogisticRegression(random_state=1)

# Fit the model using training data
lr_model = logistic_regression_model.fit(X_train, y_train)
```
```python
# Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.
# Make a prediction using the testing data
testing_predictions = logistic_regression_model.predict(X_test)
```
Now we evaluate the models performance by doing the following:
 - Calculate the accuracy score of the model.
 - Generate a confusion matrix.
 - Print the classification report.

```python
# Print the balanced_accuracy score of the model
acc_score = balanced_accuracy_score(y_test, testing_predictions)
print(f"Accuracy Score : {acc_score}")
```
```
Accuracy Score : 0.9520479254722232
```
The accuracy score is 95.2% which is how often the model was correct, or the ratio of the correctly predicted observations to the total observations.
```python
# Generate a confusion matrix for the model
# Calculating the confusion matrix
cm = confusion_matrix(y_test, testing_predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)
print("Confusion Matrix")
display(cm_df)
```
![Screenshot 2023-12-18 at 11 38 18 PM](https://github.com/samuelhfish/credit-risk-classification/assets/125224990/d24dbb1b-92ac-4518-92cc-fcb5f2fbee02)

```python
# Print the classification report for the model
print("Classification Report")
print(classification_report(y_test, testing_predictions))
```
```
Classification Report
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384
```

- Prcesion Score
Healthy: 100%
High-Risk: 85%

The precision score relates to the number of correctly predicted positive observations among all positive instances, or false positive instances that were predicted as positive that should have been negative.


- Recall Score
Healthy: 99%
High-Risk: 91%

The recall score relates to the ratio of predicted positives and number of positive instances but takes into account the number of results that were false negatives, or predicted negative and should have been positive.

### Analysis:

Because of the binary nature of the data, a logistic regression model makes sense to use. In practice for this exercise, the logistic regression model did a good job of predicting the loan status. However, it seems to do a much better job at predicting a healthy loan from a high-risk loan. A possible explanation for this is that healthy loans frequently have many characteritics in common, whereas an at risk loan might have a variety of factors that contribute to it's riskiness making it harder to predict.
