# -*- coding: utf-8 -*-

import pandas as pd

class DataFrame:
   def __init__(self, variables, columns, path):
      self.variables = variables
      self.columns = columns
      self.path = path
      
   def getDataFrame(self):
      return pd.DataFrame(self.variables, columns = self.columns)
   
   def saveDataFrame(self, df):
      df.to_csv(self.path, sep = ';')