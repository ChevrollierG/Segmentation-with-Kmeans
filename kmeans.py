# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 11:36:02 2022

@author: guill
"""

import cv2
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from copy import deepcopy


WEIGHT=[0.1,0.1,0.5,0.5,0.5,0.1,0.1]
NUM_PARAM=len(WEIGHT)
K=3


def LoadImage(image):
    url="C:/Users/guill/OneDrive/Bureau/Devoirs/Perso/Projets Info/Segmentation/pictures/"+image
    data=cv2.imread(url)
    return cv2.cvtColor(data,cv2.COLOR_BGR2RGB)

def ToGray(data):
    matrice=[]
    for i in range(len(data)):
        matrice.append([])
        for j in range(len(data[i])):
            matrice[i].append(int((data[i][j][0]+data[i][j][1]+data[i][j][2])/3))
    return matrice

def BigData(data):
    matrice=[]
    matrice.append(np.zeros(len(data[0])+2))
    for i in range(len(data)):
        matrice.append([0])
        for j in range(len(data[i])):
            matrice[i+1].append(data[i][j])
        matrice[i+1].append(0)
    matrice.append(np.zeros(len(data[0])+2))
    return matrice

def Edges(data,list_data):
    sobel_x=[[1,2,1],
             [0,0,0],
             [-1,-2,-1]]
    sobel_y=[[1,0,-1],
             [2,0,-2],
             [1,0,-1]]
    count=0
    data_gray=ToGray(data)
    big_data=BigData(data_gray)
    for i in range(1,len(big_data)-1):
        for j in range(1,len(big_data[i])-1):
            list_data[count].append(big_data[i-1][j-1]*sobel_x[0][0]+big_data[i-1][j]*sobel_x[0][1]+big_data[i-1][j+1]*sobel_x[0][2]+big_data[i][j-1]*sobel_x[1][0]+big_data[i-1][j+1]*sobel_x[1][2]+big_data[i+1][j-1]*sobel_x[2][0]+big_data[i+1][j]*sobel_x[2][1]+big_data[i+1][j+1]*sobel_x[2][2])
            list_data[count].append(big_data[i-1][j-1]*sobel_y[0][0]+big_data[i-1][j]*sobel_y[0][1]+big_data[i-1][j+1]*sobel_y[0][2]+big_data[i][j-1]*sobel_y[1][0]+big_data[i-1][j+1]*sobel_y[1][2]+big_data[i+1][j-1]*sobel_y[2][0]+big_data[i+1][j]*sobel_y[2][1]+big_data[i+1][j+1]*sobel_y[2][2])
            count+=1
      
def ListPixel(data):
    list_pixel=[]
    for i in range(len(data)):
        for j in range(len(data[i])):
            list_pixel.append([i,j,data[i][j][0],data[i][j][1],data[i][j][2]])
    return list_pixel

def AfficheImage(data):
    plt.axis('off')
    plt.imshow(data)
    plt.show()

def Distance(a,b):
    resul=0
    for i in range(NUM_PARAM):
        resul+=((int(a[i])-int(b[i]))*WEIGHT[i])**2
    resul=math.sqrt(resul)
    return resul

def Init(k,data,size_x,size_y):
    classes=[]
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(0,0,0)]
    for i in range(k):
        test=False
        while not test:
            test=True
            x=random.randint(0,size_x-1)
            y=random.randint(0,size_y-1)
            for item in classes:
                if abs(x-item[0])<=size_x/8 or abs(y-item[1])<=size_y/8:
                    test=False
        for i in data:
            if i[0]==x and i[1]==y:
                color=colors.pop(0)
                i.append(color)
                classes.append(i)
    return classes
    
def Traitement(data,classes):
    for item in data:
        if len(item)!=NUM_PARAM+1:
            classes=sorted(classes,key=lambda x: Distance(x,item))
            item.append(classes[0][len(classes[0])-1])
        
def Segmentation(data):
    matrice=[]
    for i in data:
        if i[0]==len(matrice):
            matrice.append([])
        matrice[len(matrice)-1].append(i[len(i)-1])
    return matrice

def UpdateCenter(data, classes):
    new_classes=[]
    for i in classes:
        colored=list(filter(lambda n: n[NUM_PARAM]==i[NUM_PARAM], data))
        x=sum(pixel[0] for pixel in colored)//len(colored)
        y=sum(pixel[1] for pixel in colored)//len(colored)
        center=list(filter(lambda n: n[0]==x and n[1]==y, data))
        center[0][NUM_PARAM]=i[NUM_PARAM]
        new_classes.append(center[0])
    return new_classes

def AfficheClasses(data, seg_data, classes):
    for c in classes:
        matrice=[]
        for i in range(len(seg_data)):
            matrice.append([])
            for j in range(len(seg_data[i])):
                if seg_data[i][j]==c[7]:
                    matrice[i].append(data[i][j])
                else:
                    matrice[i].append((255,255,255))
        AfficheImage(matrice)

if __name__ =='__main__':
    matrice=LoadImage("paysage6.jpg")
    AfficheImage(matrice)
    data=ListPixel(matrice)
    Edges(matrice, data)
    classes=Init(K, data, len(matrice), len(matrice[0]))
    test=True
    while test:
        copy_data=deepcopy(data)
        Traitement(copy_data, classes)
        copy_classes=classes
        classes=UpdateCenter(copy_data, classes)
        test=False
        for i in range(len(classes)):
            if abs(classes[i][0]-copy_classes[i][0])>=0.05*len(matrice) or abs(classes[i][1]-copy_classes[i][1])>=0.05*len(matrice[0]):
                test=True
                break
    new_matrice=Segmentation(copy_data)
    AfficheImage(new_matrice)
    AfficheClasses(matrice, new_matrice, copy_classes)