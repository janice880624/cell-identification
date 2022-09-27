# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:26:14 2022

@author: user
"""
import cv2
import math
import numpy
from scipy.signal import savgol_filter
path = "exp20/img(843).png"

point = []

def main(image_path):

  ok = True
  while ok:
    path2= "raw/cell (843).png"
    image = cv2.imread(image_path)  
    image1= cv2.imread(path2)

    # 灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化
    ret, thr = cv2.threshold(gray, 40, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)):
      
      if len(contours[i])==4 and contours[i][0][0].tolist() != [0, 0]:
        # print(len(contours[i]))
        # print(contours[i].shape)
        approx = cv2.approxPolyDP(contours[i], 20, True)
        cv2.drawContours(image,[approx], 0, (0,0,0), 1)

        con_list = contours[i].tolist()

        # print(con_list[0][0]) # 左上
        # print(con_list[1][0]) # 左下
        # print(con_list[2][0]) # 右下
        # print(con_list[3][0]) # 右上

        border = 0

        # left, right
        x_l, x_r = con_list[0][0][0]-border, con_list[2][0][0]+border 
        
        # up, down
        y_u, y_d = con_list[0][0][1]-border, con_list[1][0][1]+border

        crop_img = image1[y_u:y_d, x_l:x_r]
        blur_img = cv2.GaussianBlur(crop_img, (0, 0), 1500)
        usm = cv2.addWeighted(crop_img, 1.5, blur_img, -0.5, 0)
        cv2.imshow('aa{}'.format(i),usm)
        gray1 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray1, 150, 255, cv2.THRESH_BINARY)
        
        # dilation = cv2.dilate(th1 , kernel2, iterations = 2)
        # cv2.imshow('dilation{}'.format(i), dilation)

        ret11,thresh = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imshow('thresh{}'.format(i), thresh)
        aa='contours'+str(i)
        aa, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        

        
        
            
        
        # (x,y),radius=cv2.minEnclosingCircle(contours2[0])
        # center=(int(x),int(y))
        # print(center)
        # radius=int(radius)
        # print(radius)
        # img_contour= cv2.circle(crop_img,center,radius,(255,255,255),3)
        
        
        # ellipse=cv2.fitEllipse(contours2[0])
        # img_contour=cv2.ellipse(crop_img,ellipse,(0,255,0),3)
        
        
        # img_contour = cv2.drawContours(crop_img, contours2, -1, (0, 255, 0), 2)
        # cv2.imshow('img_contour{}'.format(i), img_contour)

        M = cv2.moments(aa[0])
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        print('center_x = {}, center_y = {}'.format(center_x, center_y))
        
        ppp=[]
        for jj in range(len(aa[0])):
            xs1=aa[0][jj][0][0]
            ys1=aa[0][jj][0][1]
            
            
            distance=math.pow(abs(center_x-xs1),2) + math.pow(abs(center_y-ys1),2)
            distance=math.sqrt(distance)
            ppp.append(distance)
        filter_value = savgol_filter(numpy.array(ppp), 5, 3, mode= 'nearest')
        for hh in range(len(aa[0])):
            xs2=aa[0][hh][0][0]
            ys2=aa[0][hh][0][1]
            
            
            xs3=(xs2*filter_value[hh])/ppp[hh]
            ys3=(ys2*filter_value[hh])/ppp[hh]
            xs3=round(xs3)
            ys3=round(ys3)
            aa[0][hh][0][0]=xs3
            aa[0][hh][0][1]=ys3
       
        cv2.drawContours(crop_img, aa, -1, (0, 255, 0), 2)
        cv2.circle(crop_img, (center_x,center_y), 7, 128, -1)#繪製中心點
        cv2.imshow('ff'.format(i), crop_img)
        point.append((center_x, x_l, center_y, y_u))

      else:
        continue

    for j in range(len(point)):
      font = cv2.FONT_HERSHEY_SIMPLEX
      text = '(' + str(point[j][0] + point[j][1]) + ',' + str(point[j][2] + point[j][3]) + ')'
      cv2.putText(image1, text, (point[j][0] + point[j][1]+5, point[j][2] + point[j][3]-5), font, 0.5, (255 ,100 ,0), 1, cv2.LINE_AA)

    cv2.imshow('final_image'.format(i), image1)
    cv2.waitKey(0)

    if cv2.waitKey(0) == 27:
        ok = False
            
  cv2.destroyAllWindows()

main(path)