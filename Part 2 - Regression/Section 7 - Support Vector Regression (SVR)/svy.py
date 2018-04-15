
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data
dataset = pd.read_csv("Position_Salaries.csv")
# The upper bound '2' is ignored, [:,1:2] is used to generate a matrix instead of a vector in case you write [:,1]
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

 


# Splitting the dataset into Training set and Data set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"""


 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))               

 
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)
 

# Predicting a new result 
y_pred = sc_y.inverse_transform( regressor.predict(sc_X.transform(np.array([[6.5]]))))

 
# Visualizing SVR 
plt.scatter(X, y , color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualizing the SVR results  (For higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y , color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
