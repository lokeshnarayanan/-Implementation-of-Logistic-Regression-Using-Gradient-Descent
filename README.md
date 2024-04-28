# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Lokesh N
RegisterNumber:  212222100023
*/
```
```python

import pandas as pd
import numpy as np
data=pd.read_csv('/content/Placement_Data.csv')
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1['gender'])
data1['ssc_b']=le.fit_transform(data1['ssc_b'])
data1['hsc_b']=le.fit_transform(data1['hsc_b'])
data1['hsc_s']=le.fit_transform(data1['hsc_s'])
data1['degree_t']=le.fit_transform(data1['degree_t'])
data1['workex']=le.fit_transform(data1['workex'])
data1['specialisation']=le.fit_transform(data1['specialisation'])
data1['status']=le.fit_transform(data1['status'])
X=data1.iloc[:,:-1]
Y=data1['status']
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,x,y):
  h=sigmoid(x.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=X.T.dot(h-y)
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)
```
## Output :

##  Dataset:

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/9a9e7802-e21e-4803-92b0-0f7463ada78b)

## Accuracy and actual,predicted values:

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/ceb49662-22c5-412a-9ac4-a3d5d432a2ae)

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/9f952b27-8389-40df-9a19-5e5b2c2fe867)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

