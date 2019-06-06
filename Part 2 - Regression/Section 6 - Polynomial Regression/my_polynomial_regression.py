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

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
#try doing with more degrees, it helps in improving accuracy of the model
poly_reg = PolynomialFeatures(degree = 4) 
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the linear regression results
plt.scatter(X, y, c='red')
plt.plot(X, lin_reg.predict(X), c='blue')
plt.title('truth or bluff(Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

# visualising the polynomial regression results
plt.scatter(X, y, c='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), c='blue')
plt.title('truth or bluff(Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

# visualising with more resolution (i.e. no straight line bw plotted points)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, c='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), c='blue')
plt.title('truth or bluff(Polynomial Regression) with more resolution')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

# Predicting a new result with linear regression
lin_reg.predict([[6.5]])

# Predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
