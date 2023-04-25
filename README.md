# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard python libraries for Gradient design.
2. Introduce the variables needed to execute the function.
3. Use function for the representation of the graph.
4. Using for loop apply the concept using the formulae.
5. Execute the program and plot the graph.
6. Predict and execute the values for the given conditions.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Andra likitha
RegisterNumber:  212221220006
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

print("Profit prediction graph:")
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,1000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function in a linear regression model
  """
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err) #returning ] 
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

print("Compute cost value:")
computeCost(X,y,theta) #Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  """
  Take in numpy array X,y and theta and update theta by taking number with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history  
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) value:")
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

print("Cost function using Gradient Descent graph:")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

print("Profit prediction graph:")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  """
  Take in numpy array of x and theta and return the predicted value of y based on theta
  """

  predictions= np.dot(theta.transpose(),x)

  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("Profit for the population 35,000:")
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("Profit for the population 70,000:")
print("For populat

ion = 70,000, we predict a profit of $"+str(round(predict2,0)))


```

## Output:
Profit prediction graph:
![229297745-35008d99-cc02-477b-a097-5efbcb0a30c8](https://user-images.githubusercontent.com/131592130/234219655-30b607df-b46d-48c5-8b47-48da405a487e.png)
Compute cost value:
![229297766-cdde2e4f-cf10-46d6-88b6-b8f1f04ab4d8](https://user-images.githubusercontent.com/131592130/234219988-526b63e4-0088-46c1-bbab-739e0786e2a5.png)
h(x) value:
![229529799-c33628fc-db12-48e8-88af-cfbbf83ea00e](https://user-images.githubusercontent.com/131592130/234226622-86c1d943-7513-4e98-94f4-ef84db32b762.png)
Cost function using Gradient Descent graph:
![229297805-9aec7927-1b4a-4631-a2dc-d0aec4b9ca50](https://user-images.githubusercontent.com/131592130/234226849-43ec1aa6-6351-40ce-9122-f0f5b238ec97.png)
Profit prediction graph:
![229297827-9f401d64-2648-4687-b7d6-1daefcd6bcc4](https://user-images.githubusercontent.com/131592130/234227107-b1f0db35-03e1-4b11-85e7-4a1b1585ef56.png)
Profit for the population 35,000:
![229297847-791ec19e-f63b-4c5d-b2d7-c497b05cf491](https://user-images.githubusercontent.com/131592130/234227322-81a5b03d-b22c-496c-8d7d-f063adc9326b.png)
Profit for the population 70,000:
![229297865-0cae4b12-9cbe-4d0b-bd4c-813e902f1842](https://user-images.githubusercontent.com/131592130/234227536-95e1e224-3ae6-4e80-9568-56d2904001f0.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
