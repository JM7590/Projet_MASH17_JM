# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:37:51 2017

@author: camillette
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error

X=pd.read_csv('C:/Users/Juliette Mansard/Desktop/Projet_info/ENGIE/input_training.csv', sep=';')
Y=pd.read_csv('C:/Users/Juliette Mansard/Desktop/Projet_info/ENGIE/output_training.csv', sep=';')
#%%
Y.describe()
X.head(4)
X.info()
#%%
#supprime les variable inutiles et remplace les strings
X=X.drop(['ID','Vane_position2_min','Vane_position2_max','Vane_position2_std'],axis=1) 
X=pd.concat([X, pd.get_dummies(X['MAC_CODE'], prefix='WT')], axis=1)
X=X.drop(['MAC_CODE'],axis=1)
#%%
#supprime les lignes ou il manque des données
data=pd.concat([X,Y],axis=1)
data=data.dropna()
X=data.iloc[:,0:91]
Y=data.iloc[:,92:93]
del data
#%%
#divise l'échantillon en train et test
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
del X,Y
#%%
#REGRESSION LINEAIRE
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
print(mean_absolute_error(Y_test, Y_pred))
del Y_pred
#%%
#RIDGE REGRESSION
#recherche du paramètre alpha
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
import numpy as np
parameter_grid = {
    'alpha': np.linspace(0.1,1,10)}
grid_search = GridSearchCV(Ridge(), parameter_grid,scoring="mean_absolute_error", cv=3)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
#%%
model = Ridge(alpha=0.1)
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
print(mean_absolute_error(Y_test, Y_pred))
del Y_pred
#%%
#Sélection des données sur une éolienne
del X_train,X_test,Y_train,Y_test

X=pd.read_csv('C:/Users/Juliette Mansard/Desktop/Projet_info/ENGIE/input_training.csv', sep=';')
Y=pd.read_csv('C:/Users/Juliette Mansard/Desktop/Projet_info/ENGIE/output_training.csv', sep=';')

X=X.drop(['ID','Vane_position2_min','Vane_position2_max','Vane_position2_std'],axis=1) 

data=pd.concat([X,Y],axis=1)
del X,Y
data=data.loc[data['MAC_CODE']=="WT1",:]
data=data.dropna()

X=data.iloc[:,0:88]
Y=data.iloc[:,89:90]
del data

X=pd.concat([X, pd.get_dummies(X['MAC_CODE'], prefix='WT')], axis=1)
X=X.drop(['MAC_CODE'],axis=1)
data=pd.concat([X,Y],axis=1)

#%%
#divise l'échantillon en train et test sur éolienne 1
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
del X,Y
#%%
#REGRESSION LINEAIRE sur éolienne 1
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
print(mean_absolute_error(Y_test, Y_pred))
del Y_pred
#%%
#RIDGE REGRESSION sur éolienne 1
#recherche du paramètre alpha
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
import numpy as np
parameter_grid = {
    'alpha': np.linspace(0.1,1,10)}
grid_search = GridSearchCV(Ridge(), parameter_grid,scoring="mean_absolute_error", cv=3)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
#%%
model = Ridge(alpha=1)
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
print(mean_absolute_error(Y_test, Y_pred))
del Y_pred
#%%
#Analyse en composantes principales
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
import numpy as np
pca.components_.np.where(max(pca.components_[1,:]))