# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# We dont need to split in this particular case ('Position_salaries.csv')
# because there are only 10 data sets and we want our model to be 
# more precise
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""


# Fitting  Regression Model to the dataset
# create your regressor here

# Predicting a new result with polynomial regression
y_pred = regressor.predict(([[6.5]]))


# visualising the polynomial regression results
plt.scatter(X, y, c='red')
plt.plot(X, regressor.predict((X)), c='blue')
plt.title('truth or bluff( Regression Model)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

# visualising the polynomial regression results( for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, c='red')
plt.plot(X, regressor.predict((X)), c='blue')
plt.title('truth or bluff( Regression Model)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()