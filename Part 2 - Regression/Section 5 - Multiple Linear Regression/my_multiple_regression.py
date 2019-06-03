# Multiple Linear Regression 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap
X = X[:, 1:] # we are avoiding first column of dummy variable, however we can skip this as our regression library is taking care for this

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (not applying in linear regression as library is taking care for feature scaling)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred =  regressor.predict(X_test)

#-------------------------------BACKWARD ELIMINATION---------------------- 
# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr =  np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # step 2 of Backward Elimination
regressor_OLS.summary()
# Step 3 :-Consider the predictor with highest p value
#......for P-value greater than our SL(0.05) we have to remove the respective Independent Variable
#......removing the predictor with highest P value
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # step 2 of Backward Elimination
regressor_OLS.summary()
# Step 3 :-Consider the predictor with highest p value
#......for P-value greater than our SL(0.05) we have to remove the respective Independent Variable
#......removing the predictor with highest P value
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # step 2 of Backward Elimination
regressor_OLS.summary()
# Step 3 :-Consider the predictor with highest p value
#......for P-value greater than our SL(0.05) we have to remove the respective Independent Variable
#......removing the predictor with highest P value
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # step 2 of Backward Elimination
regressor_OLS.summary()
# Step 3 :-Consider the predictor with highest p value
#......for P-value greater than our SL(0.05) we have to remove the respective Independent Variable
#......removing the predictor with highest P value
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # step 2 of Backward Elimination
regressor_OLS.summary()

#---------------------------AUTOMATED BACKWARD ELIMINATION----------------------
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)