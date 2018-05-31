import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.linear_model as skl_lm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import metrics as metrics
import statsmodels.api as sm
import statsmodels.formula.api as smf


######################################
### Lab 4.6.2: Logistic Regression ###
######################################

#Loading the dataset
mdata = pd.read_csv('Data/Smarket.csv', usecols=range(1,10), index_col=0, parse_dates=True)

# Pulling the data from 2004 and earlier to train the model on
x_train = mdata[:'2004'][['Lag1','Lag2']]
y_train = np.ravel(mdata[:'2004'][['Direction']])

# Pulling the data from 2005 and later to test the fitted model on
x_test = mdata['2005':][['Lag1','Lag2']]
y_test = np.ravel(mdata['2005':][['Direction']])

# Creating the model, training it with the x and y training data
model = skl_lm.LogisticRegression(solver = 'newton-cg')
result = model.fit(x_train, y_train);
# Using the fitted model to predict the responses of the x and y test data
y_predicted = result.predict(x_test)

print(classification_report(y_test, y_predicted, digits=3))
#print("Fitted Model Coefficients:", result.coef_)
#print("Model Accuracy:", metrics.accuracy_score(y_test,y_predicted))



