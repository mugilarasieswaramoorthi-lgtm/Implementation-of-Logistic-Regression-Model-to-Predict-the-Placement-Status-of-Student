# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Mugilarasi E 
RegisterNumber: 25017644 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

data = {
    'Square_Feet': [1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000, 3200, 3500],
    'Bedrooms': [2, 3, 3, 4, 4, 4, 5, 5, 5, 6],
    'Age': [5, 10, 15, 20, 5, 10, 15, 20, 5, 10],
    'Price': [200000, 250000, 270000, 300000, 320000, 350000, 370000, 400000, 420000, 450000],
    'Occupants': [3, 4, 4, 5, 5, 5, 6, 6, 6, 7]
}

df = pd.DataFrame(data)

X = df[['Square_Feet', 'Bedrooms', 'Age']].values
Y = df[['Price', 'Occupants']].values  

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

sgd_price = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)
sgd_occupants = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)

sgd_price.fit(X_train_scaled, Y_train_scaled[:, 0])
sgd_occupants.fit(X_train_scaled, Y_train_scaled[:, 1])

Y_pred_price_scaled = sgd_price.predict(X_test_scaled)
Y_pred_occupants_scaled = sgd_occupants.predict(X_test_scaled)

Y_pred_price = scaler_Y.inverse_transform(
    np.column_stack((Y_pred_price_scaled, np.zeros(len(Y_pred_price_scaled))))
)[:, 0]

Y_pred_occupants = scaler_Y.inverse_transform(
    np.column_stack((np.zeros(len(Y_pred_occupants_scaled)), Y_pred_occupants_scaled))
)[:, 1]

mse_price = mean_squared_error(Y_test[:, 0], Y_pred_price)
mse_occupants = mean_squared_error(Y_test[:, 1], Y_pred_occupants)

print("Predicted Prices:", Y_pred_price)
print("Actual Prices:", Y_test[:, 0])
print("Mean Squared Error (Price):", mse_price)

print("\nPredicted Occupants:", Y_pred_occupants)
print("Actual Occupants:", Y_test[:, 1])
print("Mean Squared Error (Occupants):", mse_occupants)

```

## Output:
<img width="576" height="186" alt="Screenshot 2025-10-06 202326" src="https://github.com/user-attachments/assets/c9533c45-4a59-4a9b-89d0-4c4b3f5198c9" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
