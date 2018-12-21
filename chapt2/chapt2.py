# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:43:58 2018

@author: HITCSC-AI
"""

import os
import random
import numpy as np
import time
import math


class perceptron():
    
    def __init__(self):
        self.data = []
        self.label = []
        self.yita = 0.1
        
        self.readData()
        self.w0 = np.array([random.random() for it in self.data[0]])
        self.b0 = random.random()

        self.k = 0
        self.kk = 0
        
        self.alpha0 = np.array([0 for it in self.data])
        self.beta0 = 0
        
        
    def readData(self):
        f = open("data/data.txt")
        r = f.readlines()
        for it in r:
            temp = it.split("\t")
            if it != None:
                self.data.append(np.array([float(x) for x in temp[:-1] ]))
                self.label.append(int(temp[-1]))
    
    def trainOri(self,yita = 0.1):
        self.w = self.w0
        self.b = self.b0
        misDivision = True
        self.yita = yita
        self.k = 0
        self.kk = 0
        while misDivision:
            for it in range(len(self.data)):
                self.kk +=1
                if self.label[it] * (np.dot(self.w, self.data[it]) + self.b) <= 0:
                    self.w += self.yita * self.label[it] * self.data[it]
                    self.b += self.yita * self.label[it]
                    self.k += 1
                    break
                if it == len(self.data) - 1:
                    misDivision = False
    
    def trainDualOld(self, yita = 1):
        self.alpha = self.alpha0
        self.beta = self.beta0
        gram = []
        for it in self.data:
            temp = []
            for ot in self.data:
                temp.append(np.dot(it,ot))
            gram.append(temp)        
        misDivision = True
        self.yita = yita
        self.k = 0
        self.kk = 0
        while misDivision:
            for it in range(len(self.data)):
                temp = 0
                self.kk +=1
                if self.label[it] * (sum([self.alpha[i] * self.label[i] * gram[i][it] for i in range(len(self.data))]) + self.beta) <= 0:
                    self.alpha[it] += self.yita
                    self.beta += self.yita * self.label[it]
                    self.k += 1
                    break
                if it == len(self.data) - 1:
                    misDivision = False
    
    def trainDual(self, yita = 1):
        self.alpha = self.alpha0
        self.beta = self.beta0
        gram = []
        for it in self.data:
            temp = []
            for ot in self.data:
                temp.append(np.dot(it,ot))
            gram.append(temp)
        gramA = np.array(gram)
        misDivision = True
        self.yita = yita
        self.k = 0
        self.kk = 0
        while misDivision:
            ay = np.array([self.alpha[x] * self.label[x] for x in range(len(self.alpha))])
            for it in range(len(self.data)):
                temp = 0
                self.kk +=1
                if self.label[it] * (np.dot(ay, gramA[it]) + self.beta) <= 0:
                    self.alpha[it] += self.yita
                    self.beta += self.yita * self.label[it]
                    self.k += 1
                    break
                if it == len(self.data) - 1:
                    misDivision = False    
        
    def iterationNumber(self):
        R = 0
        gamma = self.label[0] * (np.dot(self.w, self.data[0]) + self.b)
        for it in range(len(self.data)):
            RT = math.sqrt(sum([x * x for x in self.data[it]]) + 1)
            if R <= RT:
                R = RT
            gammaT = self.label[it] * (np.dot(self.w, self.data[it]) + self.b)
            if gamma >= gammaT:
                gamma = gammaT
        gamma /= math.sqrt(sum([x * x for x in self.w]) + self.b * self.b)
        kMax = R *R / (gamma * gamma)
        return kMax
        

if __name__ == "__main__":
    t = time.time()
    Perceptron = perceptron()
    t0 = time.time()
    Perceptron.trainOri(0.3)
    t1 = time.time()
    kMax = Perceptron.iterationNumber()
    print("Load data cosr",t0 - t,"s\nPerceptronOri cost ", t1 - t0,"s\niterate for",Perceptron.kk, "times, update for",Perceptron.k, "times") 
#    print(Perceptron.w, Perceptron.b, t1 - t0, t0 - t)
#    print("Iteration for ",Perceptron.k,'times.\nIdeal iteration number is', kMax)
#    print(int(kMax/Perceptron.k))
    t2 = time.time()
    Perceptron.trainDual()
    t3 = time.time()
    print("PerceptronDual cost ",t3 - t2,"s\niterate for",Perceptron.kk, "times, update for",Perceptron.k, "times") 
#    print(Perceptron.alpha, Perceptron.beta, t3 - t2) 