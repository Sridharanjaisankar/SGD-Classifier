# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Dataset:
Load the Iris dataset.

Convert the dataset into a pandas DataFrame.

The dataset includes both features (sepal length, width, petal length, width) and the target class (target).

2. Preprocess the Data:
Data Splitting: Split the data into:

Features (X): Columns representing the characteristics of the iris flowers.

Target (y): Column representing the species of the flower.

Train-Test Split:

Split the data into training (80%) and testing (20%) sets using train_test_split().

3. Model Initialization:
Initialize the SGDClassifier with the following parameters:

Max Iterations (max_iter): Set to 1000 iterations for the model to converge.

Tolerance (tol): Set to 1e-3 to define the stopping criterion (tolerance for the loss function to stop updating).

4. Train the Model:
Fit the SGDClassifier using the training data (X_train, y_train).

The classifier uses Stochastic Gradient Descent to minimize the loss function and find the best coefficients for the model.

5. Predict on Test Data:
Use the trained model to predict the classes for the test set (X_test).

6. Evaluate the Model:
Accuracy: Compute the model’s accuracy score using accuracy_score() based on the predictions (y_pred) and true values (y_test).

Confusion Matrix: Generate the confusion matrix to evaluate the performance of the model in terms of true positives, false positives, true negatives, and false negatives for each class.

Classification Report: Generate the classification report to get a detailed analysis of precision, recall, F1-score, and support for each class.

7. Make New Predictions:
Input a new sample (new flower data) and predict its class using the trained model.

8. Output:
The accuracy score, confusion matrix, and classification report of the model’s performance on the test set.

Predictions for new samples.



## Program:
```

Program to implement the prediction of iris species using SGD Classifier.
Developed by: SRIDHARAN J
RegisterNumber: 212222040158 


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

print(df.head())

x=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

print(df.head())

x=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm=confusion_matrix(y_test,y_pred)
print("Confufion Matrix:")
print(cm)

classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```

## Output:

![Screenshot 2025-03-29 190959](https://github.com/user-attachments/assets/b863dec5-3cf0-4e21-aa00-68a15d5f8556)

![Screenshot 2025-03-29 191006](https://github.com/user-attachments/assets/1097d428-7f41-4fc4-8a5c-d5961e3e11da)

![Screenshot 2025-03-29 191012](https://github.com/user-attachments/assets/fef2e7ae-fac2-43bc-a4bd-c2cd52bdb918)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
