#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
image = cv2.imread('ycell (240).png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
ok = True
while ok:
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thr = cv2.threshold(gray, 50, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
    cnt = contours[1]
    approx = cv2.approxPolyDP(cnt,20,True)
    cv2.drawContours(image,[approx],0,(0,255,0),1)
    cv2.imshow('im', image)
    c1 = approx[0][0][0]-5
    c2 = approx[1][0][0]+5
    c3 = approx[1][0][1]-5
    c4 = approx[3][0][1]+5
    print('approx:',approx)
    image1 = image[c3:c4, c1:c2]
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    
#     # blurred = cv2.blur(gray1,(3,3))
#     # 
#     # ret1,thr1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_OTSU)
    ret, th1 = cv2.threshold(gray1,150 , 255, cv2.THRESH_BINARY)
    
    erosion = cv2.erode(th1 , kernel, iterations=2)
    dilation = cv2.dilate(th1 , kernel2, iterations = 2)
    
    cv2.imshow('dilation ',dilation )
    canny = cv2.Canny(dilation, 50, 250)
    contours1, hierarchy1 = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours1))
    circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT,1,100,
                               param1=100,param2=30,minRadius=100,maxRadius=200)
    # for i in range(len(cnt)):
    #  # 椭圆拟合
    #     ellipse = cv2.fitEllipse(cnt[i])
    # # # 绘制椭圆
    #     cv2.ellipse(image1,ellipse,(0,0,255),2)
#     # x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
#     # y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
#     # absX = cv2.convertScaleAbs(x)# 轉回uint8
#     # absY = cv2.convertScaleAbs(y)
#     # dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#     # dst1 = cv2.bilateralFilter(dst1,5 ,50 ,20)
    cv2.imshow('img', image1)
    # cv2.imshow('erosion',erosion)
#     # cv2.imshow("gray", gray1)
    cv2.imshow('bi', th1)
#     # cv2.imshow("otsu2", dst1)
#     # cv2.imshow("blurred", blurred)
    cv2.imshow("canny", canny)
    
#     # print('approx:',approx)
#     # print('approx1:',c1,c2,c3,c4)
#     # cv2.circle(image,(approx[0][0][0],approx[0][0][1]),2,(0,255,0),2)
    
#     # gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    
#     # blurred = cv2.blur(gray,(5,5))
    
#     # # erosion1 = cv2.erode(blurred , kernel, iterations=2)
#     # x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
#     # y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
#     # absX = cv2.convertScaleAbs(x)# 轉回uint8
#     # absY = cv2.convertScaleAbs(y)
#     # dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#     # dst1 = cv2.bilateralFilter(dst1,5 ,50 ,20)
#     # ret1,thr1 = cv2.threshold(image, 5, 255, cv2.THRESH_OTSU)
#     # erosion = cv2.erode(thr1, kernel, iterations=2)

#     # closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=6)
    
#     # inv = 255-thr1

#     # cv2.imshow("otsu2", thr)
#     # cv2.imshow("gray", gray)
#     # cv2.imshow('thr1',thr1)
#     # cv2.imshow('dst1', dst1)
#     # cv2.imshow('erosion',erosion)
#     # cv2.imshow('closing ',closing )

#     # cv2.imwrite('erosion', erosion)
#     # cv2.imwrite('2.png', dst2)
    cv2.waitKey(0)
    if cv2.waitKey(0) == 27:
        ok = False
        
cv2.destroyAllWindows()
    # Esc key to stop