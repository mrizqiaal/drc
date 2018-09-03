# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
from GLCM import GLCM
from Preprocessing import Dataset
from Learning import kNN, GNB

path = "D:/~Temporary~/Python/Cats & Dogs/Oxford Dataset/KlasifikasiRasAnjing/Dataset.csv"
DS = Dataset.Dataset(path)
df = DS.getDF()
print(df.head())
X, Y = DS.getXY()
print(X)
k = 7
gnb = GNB.GNB(X, Y)
accuracy = gnb.getAccuracy()
print(accuracy)