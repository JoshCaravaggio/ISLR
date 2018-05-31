import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.linear_model as skl_lm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import preprocessing
from sklearn import neighbors

import statsmodels.api as sm
import statsmodels.formula.api as smf

#################################################
## Lab 4.6.4: Quadratic Discriminant Analysis  ## 
#################################################

#Loading the Dataset
mdata = pd.read_csv('Data/Smarket.csv', usecols=range(1,10), index_col=0, parse_dates=True)

x_train = mdata[:'2004'][['Lag1','Lag2']]					# Selecting data from  2004 and later to train on with Lag1 and Lag2 as predictors
y_train = mdata[:'2004']['Direction']						

x_test = mdata['2005':][['Lag1','Lag2']]					# Selecting data from 2005 & later to test on
y_test = mdata['2005':]['Direction']	


model = QuadraticDiscriminantAnalysis()
fitted_model = model.fit(x_train, y_train)
prediction = fitted_model.predict(x_test)

print(model.priors_)										# Access to prior probabilities of each response class (Up or down)
print(model.means_)										# Access to the group means for each predictor/class pairing 
print(confusion_matrix(y_test, prediction).T)					# Access to the confusion matrix for the model
print(classification_report(y_test, prediction, digits=3))	# Access to a report on the performance of the model	

pred_p = model.predict_proba(x_test)						# Predicting probabilities of each X_test entry being in either the up or down class		
print(np.unique(pred_p[:,1]>0.5, return_counts=True))	# Printing a count of the classifications with parametrized threshold of 50%



