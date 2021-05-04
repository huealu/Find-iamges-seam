"""
Full name: Hue Dinh

COSC 6334-001 - Design And Analysis Algorithm
Mini Project
"""

import numpy as np
import pandas as pd
import os
import cv2
import time
import datetime
#from google.colab.patches import cv2_imshow   # Using for running in Google Colab


def Load_Image(path):
    """
    Purpose: Load all images from input path directory into program
    Output: The images list of all images loaded from path directory
    """
    images = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file),cv2.IMREAD_UNCHANGED)
        #print(img)
        if img is not None:
            #cv2_imshow(img)
            images.append(img)
    return images


def Delta_X_Y(img, up, down, left, right, point_x, point_y):
    """
    Purpose: Computing delta of (x, y) pixel
    Output: energy of each pixel calculating from four neighbor pixels
    """
    delta_x = 0
    delta_y = 0
    for i in range(3):
        delta_x = delta_x + pow((img[down,point_y,i] - img[up,point_y,i]),2)
        delta_y = delta_y + pow((img[point_x,right,i] - img[point_x,left,i]),2)
    #delta_x = pow((img[down,point_y,0] - img[up,point_y,0]),2) + pow((img[down,point_y,1] - img[up,point_y,1]),2) + pow((img[down,point_y,2] - img[up,point_y,2]),2)
    #delta_y = pow((img[point_x,right,0] - img[point_x,left,0]),2) + pow((img[point_x,right,1] - img[point_x,left,1]),2) + pow((img[point_x,right,2] - img[point_x,left,2]),2)
    energy = delta_x + delta_y
    return energy


def Compute_Pixel_Energy(img):
    """
    Purpose" Computing the cost of each pixel and save to disruptive measurement table
    Output: The disruptive measurement table.
    """
    rows, columns, depth = img.shape
    new_img = np.full((rows, columns), None)
    #print(new_img.shape)

    for i in range(rows):
        for j in range(columns):
            #print(i, j)
            if i == 0:
                if j == 0:
                    eng = Delta_X_Y(img, rows-1, i+1, columns-1, j+1, i, j)
                elif j == columns-1:
                    eng = Delta_X_Y(img, rows-1, i+1, i-1, 0, i, j)
                else:
                    eng = Delta_X_Y(img, rows-1, i+1, j-1, j+1, i, j)
            elif i == rows-1:
                if j == 0:
                    eng = Delta_X_Y(img, i-1, 0, columns-1, j+1, i,j)
                elif j == columns-1:
                    eng = Delta_X_Y(img, i-1, 0, j-1, 0, i, j)
                else:
                    eng = Delta_X_Y(img, i-1, 0, j-1, j+1, i, j)
            elif (j == 0) or (j == columns-1):
                if (j==0):
                    eng = Delta_X_Y(img, i-1, i+1, columns-1, j+1, i, j)
                else:
                    eng = Delta_X_Y(img, i-1, i+1, j-1, 0, i, j)
            else:
                eng = Delta_X_Y(img, i-1, i+1, j-1, j+1, i, j)
            new_img[i, j] = eng

    return new_img



def Find_Seam(input_img):
    """
    Purpose: Calculating the differences between neighbor pixels and finding the seam of images
    Output: The seam and disruption table
    """
    row, col, dep = input_img.shape
    img_energy = Compute_Pixel_Energy(input_img)
  
    # Initialize disruption table and seam array contain output
    disruption = np.full((row, col), None)
    #seam = np.full((row,1), None)
  
    # Initialize the first row of disruption table
    for i in range(col):
        disruption[0, i] = img_energy[0, i]

    # Calculation disruption table
    for i in range(1, row):
        for j in range(col):
            if (j==0):
                min = np.minimum(img_energy[i-1,j], img_energy[i-1,j+1])
                disruption[i,j] = min + img_energy[i, j]
            elif (j==col-1):
                min = np.minimum(img_energy[i-1,j-1], img_energy[i-1,j])
                disruption[i,j] = min + img_energy[i,j]
            else:
                min = np.minimum(img_energy[i-1, j-1], img_energy[i-1, j])
                min = np.minimum(min, img_energy[i-1, j+1])
                disruption[i,j] = min + img_energy[i,j]
    return disruption, img_energy



def Seam(input_img, dis):
    """
    Purpose: Determining the seam of image
    Output: the seam of input image
    """
    row, col, dep = input_img.shape
    seam = np.full((row,1), None)

    # Find the seam in the last row
    index = 0
    for i in range(1,col):
        if dis[row-1,i] < dis[row-1, index]:
            index = i
    seam[row-1,0] = index
    #print(seam[row-1,0])

    # Find the seam of a row in respect to the lower row
    for i in range(row-1,1,-1):
        #print('i: ', i)
        if seam[i,0] == 0:
            if dis[i-1,0] < dis[i-1,1]:
                seam[i-1,0] = 0
            else:
                seam[i-1,0] = 1
        elif seam[i,0] == col-1:
            if dis[i-1, col-2] < dis[i-1, col-1]:
                seam[i-1,0] = col-2
                #print('seam[i,0]= col-2: ', seam[i,0])
            else:
                seam[i-1,0] = col-1
                #print('seam[i,0]= col-1: ', seam[i,0])
        else:
            index = seam[i,0]
            print('index: ', index)
            if (dis[i-1, index-1] < dis[i-1, index]):
                min = dis[i-1, index-1]
                value = index-1
            else:
                min = dis[i-1, index]
                value = index
            if (min > dis[i-1, index+1]):
                min = dis[i-1, index+1]
                value = index+1
            seam[i-1,0] = value
            #print('value: ', value, seam[i-1,0])

    return seam


def Remove_Pixel(img, seam):
    """
    Purpose: Removing the pixels in seam
    Output: Resized image
    """
    row, col, dep = img.shape
    new_img = np.zeros((row, col-1, dep))

    for i in range(row):
        idx = 0
        for j in range(col-1):
            if (seam[i,0] == j):
                idx = idx +1
            for k in range(dep):
                new_img[i,j,k] = img[i,idx,k]
            idx = idx+1
    return new_img


def main():
    INPUT = './Input_image'   # Set input directory for program
    Output = './Output/'
    images_list = Load_Image(INPUT)
    
    
    for img in images_list:
        # Show the input image
        #cv2.imshow('Input image', img)
        print('Input size: ', img.shape)
        
        # For save output file only
        dt = datetime.datetime.now()
        str_dt = dt.strftime('%d%m%Y%H%M%S')
        
        # Compute the cost of each pixel and disruption table.
        start1 = time.time()
        disruption, engergy = Find_Seam(img)
        print(time.time() - start1)
        
        # Computing seam 
        start2 = time.time()
        seam_list = Seam(img, disruption)
        print(time.time()-start2)
        
        # Remove the pixels in seam
        start3 = time.time()
        output = Remove_Pixel(img, seam_list)
        print(time.time()-start3)
        #cv2.imshow('Resized image',output)
        cv2.imwrite(Output+str_dt+'.jpg',output)
        print('The old size of image is: {}, the new size is: {}.'.format(img.shape, output.shape))


if __name__ == "__main__":
    main()