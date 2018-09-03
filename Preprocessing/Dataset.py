# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.utils import shuffle

class Dataset:
   def __init__(self, path):
      self.path = path
      self.df = shuffle(pd.read_csv(self.path, sep = ';'))
      self.df.drop(self.df.columns[[0, 3]], axis = 1, inplace = True)
      self.df.reset_index(drop = True, inplace = True)
      
   def getDF(self):
      return self.df
      
   def getXY(self):    
      X = self.df[self.df.columns[2:8]]
      Y = self.df[self.df.columns[-1]]
      
      return [X, Y]