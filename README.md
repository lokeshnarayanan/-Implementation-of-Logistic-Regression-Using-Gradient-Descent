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

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
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
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output :
## Array Value of x

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/9e553339-47f1-491c-87c0-e5b3c91a68e4)

## Array Value of y

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/bc43c506-bbff-4c09-aa5b-e47c5f5258b2)

## Exam 1 - score graph

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/68a16c17-5455-478a-973c-d0aaaed590ec)

## Sigmoid function graph

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/79cae205-9d2e-43d7-b7bf-dcc8e288919e)

## X_train_grad value

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/e671a115-f37d-4b5b-aeae-effada34a5b0)

## Y_train_grad value

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/8f76cff4-a343-43d4-a4b5-0578420ca87d)

## Print res.x

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/ec23d841-e7e0-40e9-8f35-3da598f22f9b)

## Decision boundary - graph for exam score

![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/3cbb8666-6584-4407-9e1b-3c7ab859187c)

## Proability value
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/8791e4a0-d0b2-4221-841e-8e4171d4cf24)


## Prediction value of mean
![image](https://github.com/lokeshnarayanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393019/6a9bbb76-89d3-490a-a108-3564d2051168)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

