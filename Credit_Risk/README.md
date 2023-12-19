## Code Cell 1

```python
# Import the modules
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
```

## Code Cell 2

```python
# Read the CSV file from the Resources folder into a Pandas DataFrame
file_path = Path('../Credit_Risk/Resources/lending_data.csv')
df_data = pd.read_csv(file_path)

# Review the DataFrame
df_data.head()
```

## Code Cell 3

```python
# Separate the data into labels and features

# Separate the y variable, the labels
y = df_data["loan_status"]

# Separate the X variable, the features
X = df_data.drop(columns="loan_status")
```

## Code Cell 4

```python
# Review the y variable Series
y
```

## Code Cell 5

```python
# Review the X variable DataFrame
X
```

## Code Cell 6

```python
# Check the balance of our target values
df_data.loan_status.value_counts()
```

## Code Cell 7

```python
# Import the train_test_learn module
from sklearn.model_selection import train_test_split

# Split the data using train_test_split
# Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

## Code Cell 8

```python
# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
logistic_regression_model = LogisticRegression(random_state=1)

# Fit the model using training data
lr_model = logistic_regression_model.fit(X_train, y_train)
```

## Code Cell 9

```python
# Make a prediction using the testing data
testing_predictions = logistic_regression_model.predict(X_test)
```

## Code Cell 10

```python
# Print the balanced_accuracy score of the model
acc_score = balanced_accuracy_score(y_test, testing_predictions)
print(f"Accuracy Score : {acc_score}")
```

## Code Cell 11

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

## Code Cell 12

```python
# Print the classification report for the model
print("Classification Report")
print(classification_report(y_test, testing_predictions))
```

