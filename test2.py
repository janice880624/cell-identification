import cv2

def main():
    img = cv2.imread('face2/cell (7).png')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    ok = True
    while ok:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        blurred = cv2.blur(gray,(5,5))

        x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
        y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
        absX = cv2.convertScaleAbs(x)# 轉回uint8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        dst1 = cv2.bilateralFilter(dst,5 ,50 ,20)

        ret, th1 = cv2.threshold(dst1, 10, 255, cv2.THRESH_BINARY)
        ret, th2 = cv2.threshold(dst1, 10, 255, cv2.THRESH_OTSU)
        

        erosion1 = cv2.erode(th1, kernel, iterations=2)
        erosion2 = cv2.erode(th2, kernel, iterations=2)

        th3 = cv2.adaptiveThreshold(erosion1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)

        cv2.imshow("img1", img)
        cv2.imshow("dst1", dst1)

        cv2.imshow("th1", th1)
        cv2.imshow("th2", th2)

        cv2.imshow("erosion1", erosion1)
        cv2.imshow("erosion2", erosion2)

        cv2.imshow("th3", th3)

        cv2.waitKey(0)
        if cv2.waitKey(0) == 27:
            ok = False
            
    cv2.destroyAllWindows()

main()
