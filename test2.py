# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv

L = 256


def getAngledGLCM(img, alpha):
    row, col = img.shape
    #row, col, layer = img.shape
    glcm = np.zeros((L,L), np.uint8)
    xy = [0,0]
    rowStart = 0
    colStart = 0
    rowEnd = row
    colEnd = col - 1
   
    if (alpha == 0):
        xy = [0,1]
    elif (alpha == 45):
        xy = [-1,1]
        rowStart = 1
        colStart = 0
    elif (alpha == 90):
        xy = [-1,0]
        rowStart = 1
        colEnd = col
    else:
        xy = [-1,-1]
        rowStart = 1
        colStart = 1
        colEnd = col
        
    for x in range(rowStart,rowEnd):
        for y in range(colStart,colEnd):
            pixVal = img[x,y]
            pixValNeighbor = img[x + xy[0],y + xy[1]]
            glcm[pixVal,pixValNeighbor] += 1
    
    return glcm / glcm.sum()


def getGLCM(img):
    glcm = np.zeros((L,L), np.float64)
    alpha = 0
    
    for i in range(4):
        glcm += getAngledGLCM(img, alpha)
        alpha += 45
        
    return glcm / 4


def getSumGLCM(glcm):
    sumX = []
    sumY = []
    
    for i in range(L):
        sumY.append(glcm[:,i].sum())
        sumX.append(glcm[i,:].sum())
        
    return [sumX, sumY]


def getMean(glcm, sumX, sumY):
    #sumX, sumY = getSumGLCM(glcm)
    meanX = 0.0
    meanY = 0.0
    
    for i in range(L):
        meanX += i * sumX[i]
        meanY += i * sumY[i]
        
    return [meanX, meanY]


def getVarianceXY(glcm, sumX, sumY, meanX, meanY):
    #meanX, meanY = getMean(glcm)
    #sumX, sumY = getSumGLCM(glcm)
    varX = 0.0
    varY = 0.0
    
    for i in range(L):
        varX += ((i - meanX) ** 2) * sumX[i]
        varY += ((i - meanY) ** 2) * sumY[i]
        
    return [varX, varY]


def getStandardDeviation(glcm, varX, varY):
    #varX, varY = getVarianceXY(glcm)
    
    return [np.sqrt(varX), np.sqrt(varY)]


def getASM(glcm):
    return np.power(glcm.flatten(), 2).sum()


def getContrast(glcm):
    con = 0.0
    
    for x in range(L):
        for y in range(L):
            con += ((x - y) ** 2) * glcm[x,y]
            
    return con


def getCorrelation(glcm, meanX, meanY, sdX, sdY):
    cor = 0.0
    #meanX, meanY = getMean(glcm)
    #sdX, sdY = getStandardDeviation(glcm)
    
    for x in range(L):
        for y in range(L):
            cor += (x * y) * glcm[x,y]
            
    return (cor - (meanX * meanY)) / (sdX * sdY)


def getVariance(glcm, meanX, meanY):
    #meanX, meanY = getMean(glcm)
    var = 0.0
    
    for x in range(L):
        for y in range(L):
            var += (x - meanX) * (y - meanY) * glcm[x,y]
            
    return var


def getIDM(glcm):
    idm = 0.0
    
    for x in range(L):
        for y in range(L):
            idm += glcm[x,y] / (1 + ((x - y) ** 2))
            
    return idm


def getEntropy(glcm):
    E = 0.0
    
    for x in range(L):
        for y in range(L):
            if (glcm[x,y] <= 0.0):
                E += glcm[x,y] * np.log2(np.e)
            else:
                E += glcm[x,y] * np.log2(glcm[x,y])
                
    return -E
 
   
img = cv.cvtColor(cv.imread("D:/~Temporary~/Python/Cats & Dogs/Oxford Dataset/7/Basset Hound.2.3.jpg"), cv.COLOR_BGR2GRAY)
print(getGLCM(img))