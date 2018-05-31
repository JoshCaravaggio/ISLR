import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

#################################################
#### ISLR: Linear Regression Lab
#### 3.6.3 Multiple Linear Regression
#################################################


#Loading dataset
boston = pd.read_csv('Data/Boston.csv', index_col=0)
boston.head()

# Creating a linear model with a subset of the predictors from the data.
# Columns to be excluded are added to the "difference" list as an argument
model = sm.OLS.from_formula('medv ~ ' + '+'.join(boston.columns.difference(['medv', 'age', 'indus'])), boston)
result = model.fit()
#sns.regplot()



