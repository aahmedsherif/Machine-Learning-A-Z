# Multiple Linear Regression

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values


# Spreading the dataset into Training set and Data set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)               # no fit for the test data
"""