# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:46:32 2018

@author: HITCSC-AI
"""

import os
import random


def dataGeneration(length = 1000):
    f = open("data/data.txt","w")
    for i in range(length):
        x = random.random() * 100 - 50
        y = random.random() * 100 - 50
        if x + y >= 0:
            s = str(x) + "\t " + str(y) + "\t" + "1"
            f.write(s)
        else:
            s = str(x) + "\t " + str(y) + "\t" + "-1"
            f.write(s)
        if i < length - 1 :
            f.write("\n")
    f.close()
    
if __name__ == "__main__":
    dataGeneration()