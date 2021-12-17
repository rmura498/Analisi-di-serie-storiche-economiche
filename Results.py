#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:59:27 2020

@author: raffaelemura
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import tree
from sklearn.neural_network import MLPRegressor

#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------   


class Results():
  
  def __init__():
    pass
 
  def myresults(self,name):
    
    
    mae_train=mean_absolute_error(self.ytr, self.ptr)
    mae_test=mean_absolute_error(self.yte, self.pte)
    
    mse_train=mean_squared_error(self.ytr, self.ptr)
    mse_test=mean_squared_error(self.yte, self.pte)
    
    r2_train=r2_score(self.ytr, self.ptr)
    r2_test=r2_score(self.yte, self.pte)
    
    self.results=pd.DataFrame(columns=["MAE Train","MSE Train", "R2 Train","MAE Test","MSE Test", "R2 Test"])
    self.results.loc[len(self.results)]=[mae_train,mse_train,r2_train,mae_test,mse_test,r2_test]
    
    print(f'{name}: Mae train:  {mae_train} - Mse train:   {mse_train} - R2 train:   {r2_train}')
    print(f'{name}: Mae test:  {mae_test} - Mse test:   {mse_test} - R2 test:   {r2_test}')
    
  def kindofplot(self,name):
  
    fig,axes=plt.subplots(3,1, figsize=(10,15))
 
    plt.style.use(['dark_background'])
    plt.suptitle(name, fontsize=20)
    plt.subplots_adjust(left=0.4, top=0.9, right=0.9, bottom=0.1, hspace=0.9)
    self.results[["MAE Train","MAE Test"]].plot(kind="bar",ax=axes[0],logy=True,  colormap='bwr')
    axes[0].set_title("MAE")
    self.results[["MSE Train","MSE Test"]].plot(kind="bar",ax=axes[1], logy=True, colormap='bwr')
    axes[1].set_title("MSE")
    self.results[["R2 Train","R2 Test"]].plot(kind="bar",ax=axes[2], logy=True, colormap='bwr')
    axes[2].set_title("R2")
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------  
#---------------------------------------------------------------------------------------------------------------------


class MyLinearRegression(Results):
  
  def __init__(self,**kwargs):
        
    self.model=linear_model.LinearRegression(**kwargs)



  def df(self, Xname, yname):
      
    X=pd.read_csv(Xname, header=0)
    y=pd.read_csv(yname, header=0)
    self.features=X.to_numpy()
    self.target=y.to_numpy()


  def splitting(self, **kwargs):
    
    self.xtr,self.xte,self.ytr,self.yte=train_test_split(self.features,self.target,**kwargs)
      
  def fit(self):
    
    self.model.fit(self.xtr,self.ytr)
  
  def predict(self):
    
    self.ptr=self.model.predict(self.xtr)
    self.pte=self.model.predict(self.xte)
  

    
    
#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------   
class MyDecisionTree(Results):
  
  def __init__(self,**kwargs):
        
    self.model=tree.DecisionTreeRegressor(**kwargs)



  def df(self, Xname, yname):
      
    X=pd.read_csv(Xname, header=0)
    y=pd.read_csv(yname, header=0)
    self.features=X.to_numpy()
    self.target=y.to_numpy()


  def splitting(self, **kwargs):
    
    self.xtr,self.xte,self.ytr,self.yte=train_test_split(self.features,self.target,**kwargs)
      
  def fit(self):
    
    self.model.fit(self.xtr,self.ytr)
  
  def predict(self):
    
    self.ptr=self.model.predict(self.xtr)
    self.pte=self.model.predict(self.xte)
    
#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------  
class MyMLPRegressor(Results):
  
  def __init__(self,**kwargs):
        
    self.model=MLPRegressor(**kwargs)



  def df(self, Xname, yname):
      
    X=pd.read_csv(Xname, header=0)
    y=pd.read_csv(yname, header=0)
    self.features=X.to_numpy()
    self.target=y.to_numpy()


  def splitting(self, **kwargs):
    
    self.xtr,self.xte,self.ytr,self.yte=train_test_split(self.features,self.target,**kwargs)
      
  def fit(self):
    
    self.model.fit(self.xtr,self.ytr)
  
  def predict(self):
    
    self.ptr=self.model.predict(self.xtr)
    self.pte=self.model.predict(self.xte)
    
#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------     
  
  
  
  
  
  
  
  
  
  
if __name__ == "__main__":     
  modelLR=MyLinearRegression(fit_intercept='True', normalize='False')
  modelLR.df('NASDAQ_X.csv','NASDAQ_y.csv')
  modelLR.splitting(test_size=0.3)
  modelLR.fit()
  modelLR.predict()
  modelLR.myresults('LinearRegression')
  modelLR.kindofplot('LinearRegression')
  
  modelDT=MyDecisionTree(splitter="best",max_depth=18, min_samples_split=5, max_leaf_nodes=100,random_state=1)
  modelDT.df('NASDAQ_X.csv','NASDAQ_y.csv')
  modelDT.splitting(test_size=0.5)
  modelDT.fit()
  modelDT.predict()
  modelDT.myresults('DecisionTree')
  modelDT.kindofplot('DecisionTree')
  
  modelMLP=MyMLPRegressor(hidden_layer_sizes=(200,200),solver="adam",activation="relu", random_state=2)
  modelMLP.df('NASDAQ_X.csv','NASDAQ_y.csv')
  modelMLP.splitting(test_size=0.6)
  modelMLP.fit()
  modelMLP.predict()
  modelMLP.myresults('Neural Net')
  modelMLP.kindofplot('Neural Net')
  
   
