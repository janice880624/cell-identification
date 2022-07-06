import cv2
import numpy as np

image = cv2.imread('raw/cell (7).png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
ok = True
while ok:
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.blur(gray,(5,5))

    erosion1 = cv2.erode(gray , kernel, iterations=2)
    erosion2 = cv2.erode(blurred , kernel, iterations=2)

    cv2.imshow("image", image)
    cv2.imshow("gray", gray)
    cv2.imshow("blurred", blurred)
    cv2.imshow("erosion1", erosion1)
    cv2.imshow("erosion2", erosion2)
    
    cv2.waitKey(0)
    if cv2.waitKey(0) == 27:
        ok = False
        
cv2.destroyAllWindows()
    # Esc key to stop