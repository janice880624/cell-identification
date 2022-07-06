import cv2
image = cv2.imread('raw/cell (7).png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
ok = True
while ok:
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.blur(gray,(5,5))
    
    erosion1 = cv2.erode(blurred , kernel, iterations=2)
    x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
    y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
    absX = cv2.convertScaleAbs(x)# 轉回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst1 = cv2.bilateralFilter(dst,5 ,50 ,20)

    ret1,thr1 = cv2.threshold(dst1,10, 255, cv2.THRESH_OTSU)
    erosion = cv2.erode(thr1, kernel, iterations=2)

    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    inv = 255-thr1

    cv2.imshow("gray", gray)

    # cv2.imshow("otsu2", thr1)
    # cv2.imshow("sobel1", dst1)
    cv2.imshow('erosion',erosion)
    # cv2.imshow('closing', closing)

    # cv2.imwrite('erosion.png', erosion)
    # cv2.imwrite('2.png', dst2)
    cv2.waitKey(0)
    if cv2.waitKey(0) == 27:
        ok = False
        
cv2.destroyAllWindows()
    # Esc key to stop