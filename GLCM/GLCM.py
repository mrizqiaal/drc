# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv

class GLCM:
   L = 256
   
   def __init__(self, img):
      self.img = img
      if (len(img.shape) == 3):
         self.__row, self.__col, self.Layer = img.shape
      else:
         self.__row, self.__col = img.shape
         
      #print(self.img)
      
   def getAngledGLCM(self, alpha):
      glcm = np.zeros((self.L,self.L), np.uint8)
      #print(self.__row, self.__col)
      xy = [0,0]
      rowStart = 0
      colStart = 0
      rowEnd = self.__row
      colEnd = self.__col - 1
      
      if (alpha == 0):
         xy = [0,1]
      elif (alpha == 45):
         xy = [-1, 1]
         rowStart = 1
         colStart = 0
      elif (alpha == 90):
         xy = [-1,0]
         rowStart = 1
         colEnd = self.__col
      else:
         xy = [-1,-1]
         rowStart = 1
         colStart = 1
         colEnd = self.__col
         
      for x in range(rowStart,rowEnd):
         for y in range(colStart,colEnd):
            pixVal = self.img[x,y]
            pixValNeighbor = self.img[x + xy[0],y + xy[1]]
            glcm[pixVal,pixValNeighbor] += 1
            
      return glcm / glcm.sum()
   
   def getGLCM(self):
      glcm = np.zeros((self.L,self.L), np.float64)
      alpha = 0
      
      for i in range(4):
         glcm += self.getAngledGLCM(alpha)
         alpha += 45
         
      return glcm / 4
   
   def getSumGLCM(self, glcm):
      sumX = []
      sumY = []
      
      for i in range(self.L):
         sumY.append(glcm[:,i].sum())
         sumX.append(glcm[i,:].sum())
         
      return [sumX, sumY]
   
   def getMean(self, glcm, sumX, sumY):
      meanX = 0.0
      meanY = 0.0
      
      for i in range(self.L):
         meanX += i * sumX[i]
         meanY += i * sumY[i]
         
      return [meanX, meanY]
   
   def getVarianceXY(self, glcm, sumX, sumY, meanX, meanY):
      varX = 0.0
      varY = 0.0
      
      for i in range(self.L):
         varX += np.power((i - meanX), 2)  * sumX[i]
         varY += np.power((i - meanY), 2)  * sumY[i]
         
      return [varX, varY]
   
   def getStandardDeviation(self, varX, varY):
      return [np.sqrt(varX), np.sqrt(varY)]
   
   def getASM(self, glcm):
      return np.power(glcm.flatten(), 2).sum()
   
   def getContrast(self, glcm):
      con = 0.0
      
      for x in range(self.L):
         for y in range(self.L):
            con += np.power((x - y), 2) * glcm[x,y]
            
      return con
   
   def getCorrelation(self, glcm, meanX, meanY, sdX, sdY):
      cor = 0.0
      
      for x in range(self.L):
         for y in range(self.L):
            cor += (x * y) * glcm[x,y]
            
      return (cor - (meanX * meanY)) / (sdX * sdY)
   
   def getVariance(self, glcm, meanX, meanY):
      var = 0.0
      
      for x in range(self.L):
         for y in range(self.L):
            var += (x - meanX) * (y - meanY) * glcm[x,y]
            
      return var
   
   def getIDM(self, glcm):
      idm = 0.0
      
      for x in range(self.L):
         for y in range(self.L):
            idm += glcm[x,y] / (1 + np.power((x - y), 2))
            
      return idm
   
   def getEntropy(self, glcm):
      E = 0.0
      
      for x in range(self.L):
         for y in range(self.L):
            if (glcm[x,y] <= 0.0):
               E += glcm[x,y] * np.log2(np.e)
            else:
               E += glcm[x,y] * np.log2(glcm[x,y])
               
      return -E