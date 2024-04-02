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

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:
# Array Value of x:


![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/e7cc2e23-16b8-4834-a667-611b1e16f137)
# Array Value of y:
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/0d9f888e-af11-4171-912d-e08b297323b3)

# Exam 1 - score graph:
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/ad13f585-9335-4eb2-9c7b-b1d1b2ea046e)

# Sigmoid function graph:
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/bcce4908-58bd-4f53-9310-683ed536bd3c)

# X_train_grad value:
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/df8729da-eb38-4f1f-820e-43fd03a19b8b)

# Y_train_grad value:
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/b56bd877-27a2-4281-aeea-dd3e04ced5d9)

# Print res.x:
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/dfa8ebaa-d840-4f26-b9fc-f411403d2c49)
# Decision boundary - graph for exam score:
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/168a151e-ae4d-404c-8ab7-1cd40c50789e)

# Proability value:
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/b8a992fb-4564-4032-b2b7-4c4c3c7d0f64)
# Prediction value of mean:
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/0ea1f482-073f-4d58-95cf-68ae32d3c001)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

