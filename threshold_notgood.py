import cv2

img = cv2.imread('raw/cell (7).png')
ok = True
while ok:
# 先將圖片轉為灰階
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

  # 將圖片做模糊化，可以降噪
  blur_img = cv2.medianBlur(img,5) 

  # 一般圖二值化(未模糊降噪)
  ret, th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

  # 一般圖自適應平均二值化(未模糊降噪)
  th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
          
  # 一般圖自適應高斯二值化(未模糊降噪)
  th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
          
  # 一般圖二值化(有模糊降噪)
  ret, th4 = cv2.threshold(blur_img,127,255,cv2.THRESH_BINARY)

  # 一般圖算術平均法的自適應二值化(有模糊降噪)
  th5 = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
          
  # 一般圖高斯加權均值法自適應二值化(有模糊降噪)      
  th6 = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

  cv2.imshow("img", img)
  cv2.imshow("th1", th1)
  cv2.imshow("th2", th2)
  cv2.imshow("th3", th3)
  cv2.imshow("th4", th4)
  cv2.imshow("th5", th5)
  cv2.imshow("th6", th6)

  cv2.waitKey(0)
  if cv2.waitKey(0) == 27:
      ok = False
        
cv2.destroyAllWindows()
