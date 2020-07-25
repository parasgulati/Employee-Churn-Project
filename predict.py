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
import pickle 

with open('model_logisticRegression','rb') as f:
  log_model=pickle.load(f)
with open('model_randomForest','rb') as f:
  reg_model=pickle.load(f)


def predictLogisticModel(predictQuery):
  temp=list(predictQuery.values())
  temp=numpy.array(temp)
  temp=temp.reshape(1, -1)
  yp=log_model.predict(temp) 
  return yp

def predictRandomForestModel(predictQuery):
  temp=list(predictQuery.values())
  temp=numpy.array(temp)
  temp=temp.reshape(1, -1)
  yp=reg_model.predict(temp)
  return yp