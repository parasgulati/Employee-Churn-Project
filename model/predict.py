import pandas as pd
import csv
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.special import comb 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 

LogisticRegression = joblib.load('linearRegressionModelEmployeeChurn.pkl')
RandomForestRegressor=joblib.load('randomForestModelEmployeeChurn.pkl')

def predictLogisticModel(predictQuery):
  temp=list(predictQuery.values())
  temp=numpy.array(temp)
  temp=temp.reshape(1, -1)
  yp=LogisticRegression.predict(temp) 
  return yp

def predictRandomForestModel(predictQuery):
  temp=list(predictQuery.values())
  temp=numpy.array(temp)
  temp=temp.reshape(1, -1)
  yp=RandomForestRegressor.predict(temp)
  return yp