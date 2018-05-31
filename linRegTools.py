import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


#################################################
#### ISLR: Linear Regression Lab
#### 3.6.4 Interaction Terms
#################################################

# We can build interaction terms into our model by altering the formula in the from_formula method arguments
model = sm.OLS.from_formula('medv ~ lstat*age', boston)
result = model.fit()
#print(result.summary())


#################################################
#### ISLR: Linear Regression Lab
#### 3.6.5 Non-linear Transformations of the Predictors
#################################################

# The from_formula() method can also accomodate non-linear transformations of the predictors. 
# Given a predictor like lstat, we can create a predictor lstat^2 using np.square(X). We 
# now perform a regression of medv onto lstat lstat^2
lmf1 = sm.OLS.from_formula('medv~lstat', boston).fit()
lmf2 = sm.OLS.from_formula('medv ~ lstat + np.square(lstat)', boston).fit()


print(lmf2.summary())

# The near-zero p-val for the quadratic term means that it leads to an improved model 

print (sm.stats.anova_lm(lmf1, lmf2))

# using the anova_lm method (as above) we can further quantify how much better the quadratic fit is
# by looking at the F stat and the near-zero p-value, we can quantify that our model is better with the quadratic term

# We can now check the residual plot to see if the residuals are more evenly distributed now 

lmf1_fitted_values = pd.Series(lmf1.fittedvalues, name = "Fitted Values")			#Residuals for simple fit
lmf1_residuals = pd.Series(lmf1.resid, name ="Residualss")
#sns.regplot(lmf1_fitted_values, lmf1_residuals, fit_reg = False)
#plt.show()

lmf2_fitted_values = pd.Series(lmf2.fittedvalues, name = "Fitted Values")			#Residuals for fit with quadratic term
lmf2_residuals = pd.Series(lmf2.resid, name ="Residuals")
#sns.regplot(lmf2_fitted_values, lmf2_residuals, fit_reg = False)
#plt.show()

#We can use more than just polynomial transformtions, such as a logarithm

lmf3 = sm.OLS.from_formula('medv ~ np.log(rm)', boston).fit()


#################################################
#### ISLR: Linear Regression Lab
#### 3.6.6 Qualitative Predictors 
#################################################

# Loading the Carseats dataset, we'll attempt to predict Sales (child carseat sales) in 400 locations 
# based on a number of predictor variables

carseats = pd.read_csv('Data/Carseats.csv')
carseats.head()

# The carseats dataset has some qualitative predictors in it such as Shelvloc which take on a finite number of values
# Python will generate dummy variables automatically that represent the different qualitative variables

# Heres a multiple regression model with some interaction terms 
model  = sm.OLS.from_formula('Sales ~ Income:Advertising+Price:Age + ' + "+".join(carseats.columns.difference(['Sales'])), carseats)
result = model.fit()
#print(result.summary())


