import cv2
import numpy as np
import matplotlib.pyplot as plt

# read enhanced image
img = cv2.imread('face2/cell (2).png', 0)
 
# morphological operations
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(img, kernel, iterations = 1)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
 
# Adaptive thresholding on mean and gaussian filter
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
            cv2.THRESH_BINARY, 11, 2)
# Otsu's thresholding
ret4, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
# Initialize the list
Cell_count, x_count, y_count = [], [], []
 
# read original image, to display the circle and center detection 
display = cv2.imread("face2/cell (1).png")
 
# hough transform with modified circular parameters
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 20,
                           param1 = 50, param2 = 28, minRadius = 1, maxRadius = 20)
 
# circle detection and labeling using hough transformation
if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
 
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
 
                cv2.circle(display, (x, y), r, (0, 255, 0), 2)
                cv2.rectangle(display, (x - 2, y - 2),
                              (x + 2, y + 2), (0, 128, 255), -1)
                Cell_count.append(r)
                x_count.append(x)
                y_count.append(y)
        # show the output image
        cv2.imshow("gray", display)
        cv2.waitKey(0)
 
# display the count of white blood cells
print(len(Cell_count))
# Total number of radius
print(Cell_count)
# X co-ordinate of circle
print(x_count)    
# Y co-ordinate of circle
print(y_count)  