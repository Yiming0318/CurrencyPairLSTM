# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:41:56 2020

@author: yiming
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from matplotlib import pyplot
#prepare data
df = pd.read_csv('/Users/yiming/Downloads/USDCHF_FULL_DATA.csv')
df.Date = pd.to_datetime(df.Date)
df = df.set_index('Date')
df.shape
X = df[['S&P','STD','Vol_global','Vol_local',\
        'Price_MA','Primary risk',\
        'FD','Energy','Correlation','STD_short','Price_LB',\
        'Price_UB','Price_LB_Price_UB_Distance','Distance_MA']]
y = df['USDCHF=X']   

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

#######ANOVA F-TEST FEATURE SELECTION###########
#https://machinelearningmastery.com/feature-selection-for-regression-data/
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
 
# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.figure(figsize = (30,8))
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.xticks( [0,1,2,3,4,5,6,7,8,9,10,11,12,13], ('S&P500','STD','Vol_global','Vol_local',\
                                                   'Price_MA','Primary risk',\
                                                   'FD','Energy','Correlation','STD_short','Price_LB',\
                                                   'Price_UB','Price_LB_Price_UB_Distance','Distance_MA') )
pyplot.title('ANOVA F-TEST FEATURE SELECTION(based on f-regression)')
pyplot.show()


##############Mutual Information Feature Selectio##########
# example of mutual information feature selection for numerical input data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot
 
# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.figure(figsize = (30,8))
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.xticks( [0,1,2,3,4,5,6,7,8,9,10,11,12,13], ('S&P500','STD','Vol_global','Vol_local',\
                                                   'Price_MA','Primary risk',\
                                                   'FD','Energy','Correlation','STD_short','Price_LB',\
                                                   'Price_UB','Price_LB_Price_UB_Distance','Distance_MA') )
pyplot.title('Mutual Information Feature Selection')
pyplot.show()


###############Shap based on randomforest######
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
# The target variable is 'quality'.
# Build the model with the random forest regression algorithm:
model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, y_train)

import shap
shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")


