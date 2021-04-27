# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:54:14 2021

@author: divyam
"""
"""  #### To Run the code #####
        1. To change the video comment line 22 and uncomment line 21
        2. The output would give both ransac and least square point plots and curve fits
        3. Everything should run properly, if not just run once more.(because of random sampling)
        4. To run TLS uncomment line 155 , 160, 165 and 171. Comment 163 164, 166, 172 
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import cv2 as cv
import imutils
import random
import math

#cap = cv.VideoCapture('Ball_travel_10fps.mp4')
cap = cv.VideoCapture('Ball_travel_2_updated.mp4')
if cap.isOpened() == False:
    print("Error opening the image")


coordinates = []
data = []

while cap.isOpened():
    ret, frame = cap.read()
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    except:
        break
    gray = imutils.resize(gray,width = 500)

    coordinate_list_all = []
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i][j] != 255:
                coordinate_list_all.append((j, 353 - i))
                data.append((j,353-i))
    
    coordinates.append(coordinate_list_all[30])
    if ret == True:
        cv.imshow('Frame',gray)
        
        if cv.waitKey(1) == 27:
            break
    else:
        break

plt.scatter(*zip(*coordinates))
plt.show()


def curve_fit(coordinates):
    x = []
    sum_x = 0
    sum_y = 0
    sum_xy = 0
    sum_x_sq = 0
    sum_x_cube = 0
    sum_x_4 = 0
    sum_x_sq_y = 0
    n = len(coordinates)
    for coordinate in coordinates:
        x.append(coordinate[0])
        sum_x = sum_x + coordinate[0]
        sum_y = sum_y + coordinate[1]
        sum_xy = sum_xy + coordinate[0]*coordinate[1]
        sum_x_sq = sum_x_sq + coordinate[0]*coordinate[0]
        sum_x_cube = sum_x_cube + coordinate[0]*coordinate[0]*coordinate[0]
        sum_x_4 = sum_x_4 + coordinate[0]*coordinate[0]*coordinate[0]*coordinate[0]
        sum_x_sq_y = sum_x_sq_y + coordinate[0]*coordinate[0]*coordinate[1]
    A = np.array([[n, sum_x, sum_x_sq],[sum_x, sum_x_sq, sum_x_cube],[sum_x_sq, sum_x_cube, sum_x_4]])
    B = np.array([sum_y, sum_xy, sum_x_sq_y])
    inv_A = inv(A)
    X = inv_A.dot(B)
    return [X,A,B]
        

########## RANSAC ###########
def ransac():
    iterations = math.inf
    iterations_done = 0
    y_values = []
    
    print(np.std(y_values))
    max_inlier = 0
    model = None
    threshold = 10
    outlier_prob = 0.6
    desired_prob = 0.95
    data_size = len(coordinates)
    
    
    
    while iterations > iterations_done:
        
        
        new_list = coordinates
        random.shuffle(new_list)
        sample = new_list[:3]
        sample_no = 3 
        for i in sample:
            y_values.append(sample[1])
            
        #threshold =  np.std(y_values)
        ret_model = curve_fit(sample)
        curr_model = ret_model[0]
        inlier_no = 0
        
        for i in range(len(coordinates)):
            Y = coordinates[i][1]
            y = curr_model[0] + curr_model[1]*coordinates[i][0] + curr_model[2]*coordinates[i][0] ** 2
            err = np.abs(Y-y)
            if err < threshold:
                inlier_no = inlier_no + 1
        if inlier_no > max_inlier:
            max_inlier - inlier_no
            model = curr_model
        outlier_prob = 1 - inlier_no/data_size
        iterations = int(np.log(1 - desired_prob)/np.log(1 - (1 - outlier_prob)**sample_no))
        iterations_done = iterations_done + 1
    return model

def tls(coordinates):
    x0 = np.ones(len(coordinates))
    n = len(coordinates)
    x0 = x0.reshape(n,1)
    x1 = np.array(list(zip(*coordinates))[0])
    B = np.array(list(zip(*coordinates))[1])
    x2 = np.square(x1)
    A = np.column_stack((x0, x1, x2, B))
    u, s, v = np.linalg.svd(A)
    sig = min(s)
    I = np.identity(A.shape[1])
    X = inv((A.T).dot(A) - sig**2*I).dot(A.T).dot(B)
#    ret = curve_fit(coordinates)
#    A_ret = ret[1]
#    B = ret[2]
#    A = np.column_stack((A_ret,B))
#    u, s, v = np.linalg.svd(A)
#    sig = min(s)
#    I = np.identity(A.shape[1])
#    X = inv((A.T).dot(A) - (sig**2)*I).dot(A.T).dot(B)
    
    return X 
    

def main():
    
    model = ransac()
    #model = tls(coordinates)
    ret = curve_fit(coordinates)
    model1 = ret[0]
    x = np.array(range(500))
    print(model)
    y = model[0] + model[1]*x + model[2]*x**2
    #y = (model[0] + model[1]*x + model[2]*x**2)
    y1 = model1[0] + model1[1]*x + model1[2]*x**2
    plt.scatter(*zip(*coordinates))
    plt.plot(x,y,'r--', label="RANSAC")
    #plt.plot(x,y,'g--', label="TLS")
    plt.plot(x,y1,'g--', label="Least_Sqaure")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc="upper right")
    #plt.title('RANSAC')
    #plt.title('TLS')
    plt.title('RANSAC vs Least Squares')
    plt.show()

cap.release()
cv.destroyAllWindows()
if __name__ == "__main__":
    main()
    

        