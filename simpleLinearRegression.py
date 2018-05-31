import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

#################################################
#### ISLR: Linear Regression Lab
#### 3.6.2 Simple Linear Regression
#################################################

# Defining a method for estimating responses for a list of predictions with the
# model, including the lower and upper bounds of the confidence interval 

def predict(model, new):

	#Get predicted valeus from model
	fit = pd.DataFrame(model.predict(new), columns =['fit'])

	#Get confidence interval for the model and renaming columns,
	#by default the level of confidence is 95% (alpha = 0.05)
	confidenceInterval = model.conf_int().rename(columns = {0:'lower', 1: 'upper'})	#Returns a 


	#Matrix multiplication to get the confidence intervals for the predictions
	confidenceInterval = confidenceInterval.T.dot(new.T).T	

	#Wrap up the confidence intervals with the predicted values
	return pd.concat([fit, confidenceInterval], axis=1)
	
	

#Loading dataset
boston = pd.read_csv('Data/Boston.csv', index_col=0)
boston.head()
#boston.info()

#Creating the regression lines with Ordinary Least Squares regression,
# 'medv ~ lstat' represents the formula and boston is the data object
linearModel = sm.OLS.from_formula('medv ~ lstat', boston)
result = linearModel.fit()			#Trains the model on the data added to it in the last line
print(result.summary())		

#You can grab individual statistics from the model by accessing the result attributes such as .resid and .fvalue


# Creating a dataframe of values for prediction, we will predict the medv
# with lstats of 5, 10 and 15, as well as computing our confidence interval
new = pd.DataFrame([[1,5], [1,10],[1,15]], columns = ['Intercept', 'lstat'])
print(predict(result, new));

# Plotting the lstat and medv data in boston. The regplot automatically produces
# an OLS fit. *fit_reg  = True produces an estimate of the regression line uncorrelated with our model *

#sns.regplot('lstat', 'medv', boston, line_kws = {"color": 'r'}, ci=100, fit_reg = True)		#Data plot with estimated regression line


# Pulling the fitted values and residuals and 
fitted_values = pd.Series(result.fittedvalues, name = "Fitted Values")
residuals = pd.Series(result.resid, name = "Residuals")					

#sns.regplot(fitted_values, residuals, fit_reg=False)					#Residuals Plot

# Looking for high leverage points
from statsmodels.stats.outliers_influence import OLSInfluence
s_residuals = pd.Series(result.resid_pearson, name = "S. Residuals")			#Normalized residuals can be retrieved with result.resid_pearson
leverage = pd.Series(OLSInfluence(result).influence, name = "Leverage")
sns.regplot(leverage, s_residuals, fit_reg = False)`


plt.show()


