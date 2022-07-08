# 0707 test
# by janice

import cv2
import numpy as np


path = "exp20/img(300).png"
# path = "aaa.png"


def main(image_path):
  ok = True
  while ok:
    img = cv2.imread(image_path)  
    cv2.imshow('img',img)

    print(img.shape)

    lower_red = np.array([0, 0, 180]) 
    upper_red = np.array([130, 160, 255]) 
    mask = cv2.inRange(img, lower_red, upper_red) 
    cv2.imshow('mask',mask)

    output = cv2.bitwise_and(img, img, mask = mask )  # 套用影像遮罩
    cv2.imshow('output', output)

    cv2.waitKey(0)

    if cv2.waitKey(0) == 27:
        ok = False
            
  cv2.destroyAllWindows()

main(path)
              