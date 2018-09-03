# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
from os import listdir
from os.path import isfile, join
from GLCM import GLCM
from Preprocessing import DataFrame

class ImageFolder:
   def __init__(self, path, colorspace):
      self.path = path
      self.colorspace = colorspace
      self.getImageNames()
      self.getDetails()
      
   def getImageNames(self):
      self.imgNames = [file for file in listdir(self.path) if isfile(join(self.path, file))]
      
   def getDetails(self):
      self.imgs = []
      self.dogRaces = []
      self.dogClasses = []
      for img in self.imgNames:
         if (self.colorspace == "Grayscale"):
            self.imgs.append(cv.cvtColor(cv.imread(self.path + img), cv.COLOR_BGR2GRAY))
         elif (self.colorspace == "HSV"):
            self.imgs.append(cv.cvtColor(cv.imread(self.path + img), cv.COLOR_BGR2HSV))
         else:
            self.imgs.append(cv.imread(self.path + img))
         stringSplit = img.split('.')
         self.dogRaces.append(stringSplit[0])
         self.dogClasses.append(stringSplit[2])
         
   def getImages(self):
      return self.imgs
   
   def getDogRaces(self):
      return self.dogRaces
   
   def getDogClasses(self):
      return self.dogClasses
   
   def featureExtract(self, path):
      mean = []
      asm = []
      contrast = []
      correlation = []
      variance = []
      idm = []
      entropy = []
      for img in self.imgs:
         g = GLCM.GLCM(img)
         glcm = g.getGLCM()
         sumX, sumY = g.getSumGLCM(glcm)
         meanX, meanY = g.getMean(glcm, sumX, sumY)
         varX, varY = g.getVarianceXY(glcm, sumX, sumY, meanX, meanY)
         sdX, sdY = g.getStandardDeviation(varX, varY)
         mean.append([meanX, meanY])
         asm.append(g.getASM(glcm))
         contrast.append(g.getContrast(glcm))
         correlation.append(g.getCorrelation(glcm, meanX, meanY, sdX, sdY))
         variance.append(g.getVariance(glcm, meanX, meanY))
         idm.append(g.getIDM(glcm))
         entropy.append(g.getEntropy(glcm))
         
      variables = {"img": self.imgNames, "race": self.dogRaces, "mean": mean, "asm": asm, "contrast": contrast, "correlation": correlation, "variance": variance, "idm": idm, "entropy": entropy, "class": self.dogClasses}
      columns = ["img", "race", "mean", "asm", "contrast", "correlation", "variance", "idm",  "entropy", "class"]
      dfObj = DataFrame.DataFrame(variables, columns, path)
      dfObj.saveDataFrame(dfObj.getDataFrame())