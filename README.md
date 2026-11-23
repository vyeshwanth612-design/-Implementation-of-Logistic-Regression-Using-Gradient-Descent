# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import NumPy and initialize the input data (X) and output data (Y).
2. Initialize weight, bias, learning rate and number of iterations.
3. Calculate the predicted values using the sigmoid function and update weight and bias using gradient descent.
4. 1. Predict the result and display whether the student is placed or not placed.

## Program:
```

import numpy as np

X = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([0, 0, 0, 1, 1, 1])

w = 0
b = 0
alpha = 0.1
m = len(X)

for i in range(5000):
    z = w * X + b
    p = 1 / (1 + np.exp(-z))
    
    dw = np.sum((p - Y) * X) / m
    db = np.sum(p - Y) / m
    
    w = w - alpha * dw
    b = b - alpha * db

x = 4.5
z = w * x + b
prediction = 1 / (1 + np.exp(-z))

if prediction >= 0.5:
    print("Student is Placed")
else:
    print("Student is Not Placed")

```

## Output:
<img width="170" height="40" alt="image" src="https://github.com/user-attachments/assets/c0c92406-8921-4fb8-83c9-25cb7e348c2f" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

