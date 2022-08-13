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


WEIGHT=[0.1,0.1,0.4,0.4,0.4,0.1,0.1]
NUM_PARAM=len(WEIGHT)
K=3


def LoadImage(image):
    """
        Load the image with OpenCV
        input: image: link to the image
        output: 2 dimensions array containing RVB tuples
    """
    url="C:/Users/guill/OneDrive/Bureau/Devoirs/Perso/Projets Info/Segmentation/pictures/"+image
    data=cv2.imread(url)
    return cv2.cvtColor(data,cv2.COLOR_BGR2RGB)

def ToGray(data):
    """
        Turns an image to gray by averaging RVB chanels for each pixels
        input: data: 2 dimensions list containing RVB tuples
        output: 2 dimensions array containing integers
    """
    matrice=[]
    for i in range(len(data)):
        matrice.append([])
        for j in range(len(data[i])):
            matrice[i].append(int((data[i][j][0]+data[i][j][1]+data[i][j][2])/3))
    return matrice

def BigData(data):
    """
        Increase the dimensions of the array by 2
        input: data: matrix
        output: bigger matrix
    """
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
    """
        Calculate edges for each pixels with the sobel operator and add it to list_data
        input: data: matrix
               list_data: a list of tuple representing each pixels by their parameters (dimensions, color, edges)
        output: no output
    """
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
    """
        Make a list of pixels with their parameters (dimensions, color)
        input: data: matrix
    """
    list_pixel=[]
    for i in range(len(data)):
        for j in range(len(data[i])):
            list_pixel.append([i,j,data[i][j][0],data[i][j][1],data[i][j][2]])
    return list_pixel

def AfficheImage(data):
    """
        Plot the image with matplotlib
        input: data: matrix
    """
    plt.axis('off')
    plt.imshow(data)
    plt.show()

def Distance(a,b):
    """
        Calculate the 'distance' or the error between two pixels
        input: a: first pixel
               b: second pixel
        output: error
    """
    resul=0
    for i in range(NUM_PARAM):
        resul+=((int(a[i])-int(b[i]))*WEIGHT[i])**2
    resul=math.sqrt(resul)
    return resul

def Init(k,data,size_x,size_y):
    """
        Choose k random centers (classes) to start the algorithm
        input: k: parameter k
               data: list of tuples representing pixels
               size_x: horizontal length of the matrix
               size_y: vertical length of the matrtix
        output: a list of tuples representing the k pixels
    """
    classes=[]
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0)]
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
    """
        Loop over every items to calculate the distance with the classes and affect the color of the nearest class
        input: data: list of tuple representing the pixels
               classes: list of tuple representing the k pixel classes
    """
    for item in data:
        if len(item)!=NUM_PARAM+1:
            classes=sorted(classes,key=lambda x: Distance(x,item))
            item.append(classes[0][len(classes[0])-1])
        
def Segmentation(data):
    """
        Build a matrix of RVB tuple with the list of pixels
        input: data: list of tuple representing the pixels
        output: matrice: matrix colored by the classes
    """
    matrice=[]
    for i in data:
        if i[0]==len(matrice):
            matrice.append([])
        matrice[len(matrice)-1].append(i[len(i)-1])
    return matrice

def UpdateCenter(data, classes):
    """
        Average dimension x and y of all cluster to update the center of each classes
        input: data: list of tuples representing the pixels
               classes: list of tuples representing the pixel classes
        output: list of tuples representing the pixel classes updated
    """
    new_classes=[]
    for i in classes:
        colored=list(filter(lambda n: n[NUM_PARAM]==i[NUM_PARAM], data))
        x=sum(pixel[0] for pixel in colored)//len(colored)
        y=sum(pixel[1] for pixel in colored)//len(colored)
        center=list(filter(lambda n: n[0]==x and n[1]==y, data))
        center[0][NUM_PARAM]=i[NUM_PARAM]
        new_classes.append(center[0])
    return new_classes

def KmeansIteration(data, list_data, classes):
    """
        Loop until the center of each classes barely changes
        input: data: matrix
               list_data: list of tuples representing the pixels
               classes: list of tuples representing the pixel classes
        output: copy_data: the last iteration of the list of tuples representing pixels
                copy_classes: the last iteration of the list of tuples representing pixel classes
    """
    test=True
    while test:
        copy_data=deepcopy(list_data)
        Traitement(copy_data, classes)
        copy_classes=classes
        classes=UpdateCenter(copy_data, classes)
        test=False
        for i in range(len(classes)):
            if abs(classes[i][0]-copy_classes[i][0])>=0.05*len(data) or abs(classes[i][1]-copy_classes[i][1])>=0.05*len(data[0]):
                test=True
                break
    return copy_data, copy_classes

def AfficheClasses(data, seg_data, classes):
    """
        Plot each classes on a different image 
        input: data: matrix
               seg_data: segmented matrix
               classes: list of tuples representing pixels
    """
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
    matrice=LoadImage("landscape4.jpg")
    AfficheImage(matrice)
    data=ListPixel(matrice)
    Edges(matrice, data)
    classes=Init(K, data, len(matrice), len(matrice[0]))
    copy_data, copy_classes=KmeansIteration(matrice, data, classes)
    new_matrice=Segmentation(copy_data)
    AfficheImage(new_matrice)
    AfficheClasses(matrice, new_matrice, copy_classes)