# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:59:45 2025

@author: j
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#df = pd.read_csv('Salary_Data.csv', delimiter =';' , decimal = '.' )
df = pd.read_csv('Salary_Data.csv')
print(df.head())

#Divide the dataset into X et Y
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

#Divide the dataset into the training set andc the test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state= 0)

#Building the model
reg = LinearRegression()
reg.fit(X_train, Y_train) #cette fonction nous permet de passer les donnees au model au travers de la fonction fit

#Create new prediction
Y_pred = reg.predict(X_test)
print(reg.predict(np.array([[20]]))) #pour predire modifier juste cette valeur

#To show the result
plt.scatter(X_test, Y_test, color = 'blue')
plt.plot(X_train, reg.predict(X_train), color ='red')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.title('Salaire X Experience')
plt.show()

#Evaluate the model
print(reg.score(X, Y))