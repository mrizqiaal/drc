# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

class GNB:
   def __init__(self, X, Y):
      self.__X = X
      self.__Y = Y
      self.__m = 2
      self.__n = 5
      self.__gnbModel = GaussianNB()
      self.__kfold = KFold(n_splits = self.__m, shuffle = True)
      
   def getModel(self):
      return self.__gnbModel
   
   def getAccuracy(self):
      scoreTemp = []
      for i in range(self.__n):
         temp = []
         for train, test in self.__kfold.split(self.__X):
            self.__gnbModel.fit(self.__X.iloc[train], self.__Y.iloc[train])
            temp.append(self.__gnbModel.score(self.__X.iloc[test], self.__Y.iloc[test]))
         scoreTemp.append(np.mean(temp))
      
      return np.mean(scoreTemp)