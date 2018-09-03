# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import KFold

class kNN:
   def __init__(self, k, X, Y):
      self.__k = k
      self.__X = X
      self.__Y = Y
      self.__m = 2
      self.__n = 5
      self.__knnModel = neighbors.KNeighborsClassifier(n_neighbors = k, metric = "euclidean")
      self.__kfold = KFold(n_splits = self.__m, shuffle = True)
      
   def getModel(self):
      return self.__knnModel
   
   def getAccuracy(self):
      scoreTemp = []
      for i in range(self.__n):
         temp = []
         for train, test in self.__kfold.split(self.__X):
            self.__knnModel.fit(self.__X.iloc[train], self.__Y.iloc[train])
            temp.append(self.__knnModel.score(self.__X.iloc[test], self.__Y.iloc[test]))
         scoreTemp.append(np.mean(temp))
      
      return np.mean(scoreTemp)