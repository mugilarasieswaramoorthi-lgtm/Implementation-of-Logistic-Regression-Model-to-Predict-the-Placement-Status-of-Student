# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start and Import the Required Libraries

2.Load and Prepare the Dataset

3.Split the Dataset into Training and Testing Sets

4.Scale the Features and Train Separate SGD Regression Models

5.Predict, Evaluate, and Display the Results for Both Outputs 

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Mugilarasi E 
RegisterNumber: 25017644 


import pandas as pd 
data=pd.read_csv("Placement_Data.csv") 
print("first 5 elements:\n",data.head()) 
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1) 
data1.head() 
data1.isnull() 
data1.duplicated().sum() 
from sklearn .preprocessing import LabelEncoder 
le=LabelEncoder() 
data1["gender"]=le.fit_transform(data1["gender"]) 
data1["ssc_b"]=le.fit_transform(data1["ssc_b"]) 
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"]) 
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"]) 
data1["specialisation"]=le.fit_transform(data1["specialisation"]) 
data1["status"]=le.fit_transform(data1["status"]) 
data1 
x=data1.iloc[:,:-1] 
print("\n x:\n",x) 
y=data1["status"] 
print("\n y:\n",y)
 from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_sta
 from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train) 
y_pred=lr.predict(x_test) 
print("\n y_pred:\n",y_pred) 
from sklearn.metrics import accuracy_score 
accuracy=accuracy_score(y_test,y_pred) 
print("\n accuracy:\n",accuracy) 
from sklearn.metrics import confusion_matrix
 confusion=confusion_matrix(y_test,y_pred)
 print("\n confusion matrix:\n",confusion)
 from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
 print("\n classification_report1:\n",classification_report1) 
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
![5th 1](https://github.com/user-attachments/assets/b5ab9d0d-7939-4ced-b7b3-a07962d7f138)
![5th 2](https://github.com/user-attachments/assets/69fdb6ee-e2a0-476e-bcd6-06dcb282e0d8)
![5th 3](https://github.com/user-attachments/assets/eaae98fb-351a-40b9-b64c-abe5612d860e)
![5th 4](https://github.com/user-attachments/assets/bbae59fe-e8f6-4f6f-9ff1-be3840e6d424)
![5th 5](https://github.com/user-attachments/assets/20b9dfdb-26f7-452f-aab1-291908d15ae7)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
