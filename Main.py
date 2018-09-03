# -*- coding: utf-8 -*-

from ui import mainUI as mu
from PyQt5.QtWidgets import QWidget, QPushButton, QFileDialog, QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
#from GLCM import GLCM
#from Preprocessing import DataFrame, Dataset, ImageFolder
#from Learning import kNN, GNB
#import cv2 as cv
import sys
import os
sys.path.append(os.getcwd() + '\\')
print(os.getcwd())

class Main(QWidget):
   __folderUrl = ""
   __csvUrl = ""
   __csvName = "Dataset.csv"
   __imgUrl = ""
   
   def __init__(self):
      super().__init__()
      self.ui = mu.Ui_Form()
      self.ui.setupUi(self)
      
      self.ui.selectFolder_radio.toggled.connect(self.radioFolderSelected)
      self.ui.selectCSV_radio.toggled.connect(self.radioCSVSelected)
      self.ui.selectFolder_edit.setDisabled(True)
      self.ui.selectCSV_edit.setDisabled(True)
      self.ui.selectFolder_browser.setDisabled(True)
      self.ui.selectCSV_browser.setDisabled(True)
      self.ui.selectImage_button.setDisabled(True)
      self.ui.selectFolder_browser.clicked.connect(self.folderBrowserClicked)
      self.ui.selectCSV_browser.clicked.connect(self.csvBrowserClicked)
      self.ui.startTrain_button.clicked.connect(self.startTrainClicked)
      self.ui.colorspace_combo.currentTextChanged.connect(self.colorspaceChanged)
      self.ui.selectImage_button.clicked.connect(self.selectImageClicked)
      
      self.show()
      
   def radioFolderSelected(self, enable):
      if enable:
         self.ui.selectFolder_edit.setDisabled(False)
         self.ui.selectFolder_browser.setDisabled(False)
         self.ui.selectCSV_edit.setDisabled(True)
         self.ui.selectCSV_browser.setDisabled(True)
         
   def radioCSVSelected(self, enable):
      if enable:
         self.ui.selectFolder_edit.setDisabled(True)
         self.ui.selectFolder_browser.setDisabled(True)
         self.ui.selectCSV_edit.setDisabled(False)
         self.ui.selectCSV_browser.setDisabled(False)
         
   def colorspaceChanged(self):
      self.ui.colorspace_label.setText(str(self.ui.colorspace_combo.currentText()))
      
   def selectImageClicked(self):
      url = str(QFileDialog.getOpenFileName(self, "Select File")[0])
      if url != "":
         self.__imgUrl = url
         self.setInputImage(self.__imgUrl)
      print(self.__imgUrl)
      
   def setInputImage(self, path):
      self.__inputImg = cv.imread(path)
      self.__inputPixmap = QPixmap(path)
      self.ui.inputImage_label.setPixmap(self.__inputPixmap)
      self.__inputImgName = path.split('/')[-1]
      self.classified()
      
   def classified(self):
      if (self.ui.colorspace_combo.currentText() == "Grayscale"):
         self.__inputImg = cv.cvtColor(self.__inputImg, cv.COLOR_BGR2GRAY)
      elif (self.ui.colorspace_combo.currentText() == "HSV"):
         self.__inputImg = cv.cvtColor(self.__inputImg, cv.COLOR_BGR2HSV)
      else:
         print()
      self.getGLCM()
      ori = str(self.learningModel.predict(self.__iX)[0])
      self.ui.originalClass_label.setText(ori)
      self.ui.predictedClass_label.setText(str(self.__inputImgName.split('.')[2]))
      print("\n", self.__inputImgName.split('.'))
         
   def getGLCM(self):
      g = GLCM.GLCM(self.__inputImg)
      self.__glcm = g.getGLCM()
      self.__sumX, self.__sumY = g.getSumGLCM(self.__glcm)
      self.__meanX, self.__meanY = g.getMean(self.__glcm, self.__sumX, self.__sumY)
      self.__varX, self.__varY = g.getVarianceXY(self.__glcm, self.__sumX, self.__sumY, self.__meanX, self.__meanY)
      self.__sdX, self.__sdY = g.getStandardDeviation(self.__varX, self.__varY)
      self.__mean = [self.__meanX, self.__meanY]
      self.__asm = g.getASM(self.__glcm)
      self.__contrast = g.getContrast(self.__glcm)
      self.__correlation = g.getCorrelation(self.__glcm, self.__meanX, self.__meanY, self.__sdX, self.__sdY)
      self.__variance = g.getVariance(self.__glcm, self.__meanX, self.__meanY)
      self.__idm = g.getIDM(self.__glcm)
      self.__entropy = g.getEntropy(self.__glcm)
      self.__variables = {"img": self.__inputImgName, "race": self.__inputImgName.split('.')[0], "mean": self.__mean, "asm": self.__asm, "contrast": self.__contrast, "correlation": self.__correlation, "variance": self.__variance, "idm": self.__idm, "entropy": self.__entropy, "class": self.__inputImgName.split('.')[2]}
      self.__columns = ["img", "race", "mean", "asm", "contrast", "correlation", "variance", "idm",  "entropy", "class"]
      dfObj = DataFrame.DataFrame(self.__variables, self.__columns, os.getcwd() + '\\' + "xx.csv")
      dfInputImg = dfObj.getDataFrame()
      dfObj.saveDataFrame(dfInputImg)
      DS = Dataset.Dataset(os.getcwd() + '\\' + "xx.csv")
      self.__iX, self.__iY = DS.getXY()
      #print(self.__iX, self.__iY)
         
   def folderBrowserClicked(self):
      url = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
      if url != "":
         self.__folderUrl = url + '/'
         self.ui.selectFolder_edit.setText(self.__folderUrl)
      print(self.__folderUrl)
      
   def csvBrowserClicked(self):
      url = str(QFileDialog.getOpenFileName(self, "Select File")[0])
      if url != "":
         self.__csvUrl = url
         self.ui.selectCSV_edit.setText(self.__csvUrl)
      print(self.__csvUrl)
      
   def startTrainClicked(self):
      self.learningSelection = str(self.ui.learning_combo.currentText())
      self.colorspace = str(self.ui.colorspace_combo.currentText())
      
      if (self.ui.selectFolder_radio.isChecked()):
         self.__folderUrl = str(self.ui.selectFolder_edit.text())
         IF = ImageFolder.ImageFolder(self.__folderUrl, self.colorspace)
         IF.featureExtract(os.getcwd() + '\\' + self.__csvName)
         DS = Dataset.Dataset(os.getcwd() + '\\' + self.__csvName)
         self.df = DS.getDF()
         self.X, self.Y = DS.getXY()
      else:
         self.__csvUrl = str(self.ui.selectCSV_edit.text())
         DS = Dataset.Dataset(self.__csvUrl)
         self.df = DS.getDF()
         self.X, self.Y = DS.getXY()
      print(self.X, self.Y)
      
      if (self.learningSelection == "k-NN"):
         k = int(self.ui.selectK_edit.text())
         self.learning = kNN.kNN(k, self.X, self.Y)
         self.learningModel = self.learning.getModel()
         self.accuracy = self.learning.getAccuracy()
      else:
         self.learning = GNB.GNB(self.X, self.Y)
         self.learningModel = self.learning.getModel()
         self.accuracy = self.learning.getAccuracy()
      print(self.accuracy)
      self.ui.accuracy_label.setText("%.2f" % (self.accuracy * 100))
      self.ui.selectImage_button.setDisabled(False)
      
mainWindow = Main()
