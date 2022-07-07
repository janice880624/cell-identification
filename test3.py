import cv2
import numpy as np

num = 50

def main(img, str):
  # 显示灰度图
  img1 = cv2.imread(img,0)
  # cv2.imshow("img1",img1)

  # 自適應分割二值化分割
  img2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 4)
  # cv2.imshow('img2', img2)

  # 圖像反色
  img3 = cv2.bitwise_not(img2)
  # cv2.imshow("img3", img3)

  # 图像扩展
  img4 = cv2.copyMakeBorder(img3, 1, 1, 1, 1, cv2.BORDER_REFLECT)
  # cv2.imshow("img4", img4)

  contours, hierarchy = cv2.findContours(img4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  # 刪除小面積
  img5 = img4
  for i in range(len(contours)):
      area = cv2.contourArea(contours[i])
      if (area < 60) | (area > 10000):
          cv2.drawContours(img5, [contours[i]], 0, 0, -1)
  # cv2.imshow("img5", img5)

  name = "out/output{}.png".format(str)
  cv2.imwrite(name, img5)

  # cv2.waitKey(0)

for i in range(num):
  path = "face2/cell ({}).png".format(str(i+1))
  print(path)
  main(path, str(i+1))  