import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

def main():

  ok = True

  while ok:

    # read original image
    image = cv2.imread("face2/cell (843).png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray.png', gray)
    # histogram(gray)

    blurM = cv2.medianBlur(gray, 5)
    cv2.imshow('blurM.png', blurM)
    
    histoNorm = cv2.equalizeHist(gray)
    cv2.imshow('histoNorm.png', histoNorm)

    thresh, img2 = cv2.threshold(histoNorm, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('thresholding', img2)

    img1_not =cv2.bitwise_not(img2)
    cv2.imshow('img1_not', img1_not)

    edge = cv2.Canny(img1_not, 50, 200)
    cv2.imshow('edge.png', edge)

    cv2.waitKey(0)
    if cv2.waitKey(0) == 27:
      ok = False

  cv2.destroyAllWindows()

main()