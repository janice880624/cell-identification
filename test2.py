import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('raw/cell (7).png')

ok = True
while ok:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    blurred = cv2.blur(gray,(5,5))

    

    ret,thresh1 = cv2.threshold(blurred, 127, 150, cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(blurred, 127, 150, cv2.THRESH_BINARY_INV)

    cv2.imshow("img", img)
    cv2.imshow("thresh1", thresh1)
    cv2.imshow("thresh2", thresh2)

    cv2.waitKey(0)
    if cv2.waitKey(0) == 27:
        ok = False
        
cv2.destroyAllWindows()
