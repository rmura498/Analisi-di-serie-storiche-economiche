#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:36:08 2020

@author: raffaelemura
"""

import pandas as pd
from sklearn import preprocessing as sklp


#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------


def main():
  
  fn="30Y^IXIC.csv"
  
  df=Csv_to_df(fn)
  df=useful_features(df)
  df=target(df)
  df=clean_df(df)
  df=more_features(df)
  
  X,y=create_Xy(df)
  
  X=X.to_numpy()
  y=y.to_numpy()
  
 

  
  X=scaling(X)
  
  y=scaling(y)
  
  df_to_csv(df=pd.DataFrame(X), fn="NASDAQ_X.csv")
  df_to_csv(df=pd.DataFrame(y), fn="NASDAQ_y.csv")

  
  
#--------------------------------------------------------------------------------------------------------------------------   
#--------------------------------------------------------------------------------------------------------------------------  

def Csv_to_df(fn):
  
  df=pd.read_csv(fn, header=0)
  
  
  return df

#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------  
 
def useful_features(df):
  
  
  df["Return"]=df.Close/df.Close.shift(1)
  df["Close_open"]=df.Close/df.Open
  df["Close_high"]=df.Close/df.High
  df["Close_low"]=df.Close/df.Low
  df=df.iloc[1:]
  return df


def target(df):
  
  df["y"]=df.Return.shift(-1)
  
  return df

#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------

def clean_df(df):
  
  df=missing(df)
  
  return df

def missing(df):
  
  c=df.isnull().sum()
  
  if sum(c)==0:
    return df
  else:
    df=df.fillna(method="ffill", axis=0).fillna("0")
    
  return df
  
#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------

def more_features(df):
  
  df=trend(df)
  df=momentum(df)
  df=volatility(df)
  df=volume(df)
  
  return df

#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------

def trend(df):
  
  df=macd(df)
  df=sma(df)
  
  return df

def sma(df):
  
  df["sma50d"]=df.Close.rolling(50).mean()
  df["sma200d"]=df.Close.rolling(200).mean()
  df["sma50_200"]=df.sma50d-df.sma200d
  
  return df

def macd(df):
  
  ema12d=df.Close.ewm(com=(12-1)/2).mean()
  ema26d=df.Close.ewm(com=(26-1)/2).mean()
  
  df["macd_line"]=ema12d-ema26d
  df["macd9d"]=df.macd_line.ewm(com=(9-1)/2).mean()
  df["macd_diff"]=df.macd_line - df.macd9d
  
  return df

#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------

def momentum(df):
  
  df=SO(df)
  df=CCI(df)
 
  return df

def SO(df):
  
  periods=14
  
  df["SO"]=(( df.Close - df.Close.rolling(periods).min()) 
            
            /
            
            (df.Close.rolling(periods).max() - df.Close.rolling(periods).min()))*100
  
  
  return df

def CCI(df):
  
  tp=(df.High+df.Low+df.Close)/3
  
  mdev= abs(tp-tp.rolling(20).mean())/20
 
  df["CCI"]=(tp-tp.rolling(20).mean()) / (0.15*mdev)
  
  return df

#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------

def volatility(df):
  
  df["5d_vol"]=df.Return.rolling(5).std()
  df["21d_vol"]=df.Return.rolling(21).std()
  df["60d_vol"]=df.Return.rolling(60).std()
  
  df=BB(df)
  df=ATR(df)
  
  return df

def BB(df):
  
  df["BB"]=((df.Close-df.Close.rolling(21).mean())
            
            /
            
            2*df.Close.rolling(21).std())
  
  return df

def ATR(df):
  
  h_l=df.High-df.Low
  h_prevclose=df.High-df.Close.shift(-1)
  l_prevclose=df.Low-df.Close.shift(-1)
  tr=h_l.to_frame("hig_low")
  tr["high_prevclose"]=h_prevclose
  tr["low_prevclose"]=l_prevclose
  tr["tr"]=tr.max(axis=1)
  df["ATR"]=tr["tr"].rolling(14).mean()
  
  return df
  
#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------  

def volume(df):
  
  df["v_rolling"]=df.Volume/df.Volume.shift(21)
  df=OBV(df)
  df=CHKO(df)
  
  return df

def OBV(df):
  
  df["odds"]=df.Close.shift(1)-df.Close
  df["OBV"]=df["Volume"]
  df.OBV=df.apply(lambda row: row.Volume*-1 if row.odds<0 else row.Volume, axis=1)
  df.OBV=df.OBV.cumsum()
  
  return df

def CHKO(df):
  
  flow_mult=((df.Close-df.Low)/ (df.High-df.Low))/(df.High-df.Low)
  flow_vol=df.Volume*flow_mult
  adl=flow_vol.cumsum()
  df["CHKO"]=adl.ewm(com=(3-1)/2).mean()-adl.ewm(com=(10-1)/2).mean()
  
  return df


#--------------------------------------------------------------------------------------------------------------------------  
#-------------------------------------------------------------------------------------------------------------------------- 

def create_Xy(df):
  
  X=df.loc[200:len(df)-1, ["Close_open","Close_high","Close_low","sma50_200","macd_diff","SO","CCI","5d_vol","21d_vol","60d_vol","BB","ATR","OBV","CHKO"]]
  y=df.loc[200:len(df)-1, ["y"]]

  return X,y

#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------


def scaling(df):
  
  model=sklp.StandardScaler().fit(df)
  X=model.transform(df)
  return X 

#--------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------

def df_to_csv(df,fn):
  
  df.to_csv(fn, index=False)
  
if __name__ == "__main__":
  main()  
    













