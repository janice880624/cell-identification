# created by janice
# start at 2022/08/18

import cv2
import numpy as np

kernel = np.ones((3, 3),int)
kernel2 = np.ones((6, 6),int)

# num = 76
num = 400

yolo_path = "exp20/img({}).png".format(num)
face_path = "face2/cell ({}).png".format(num)

point = []
cv_contours = []

def main(yolo_path, face_path):

  ok = True
  while ok:
    image1 = cv2.imread(yolo_path)  
    image2 = cv2.imread(face_path)  

    # 灰階
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # 二值化
    ret, thr = cv2.threshold(gray, 50, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
      if len(contours[i])==4 and contours[i][0][0].tolist() != [0, 0]:
        # print(len(contours[i]))
        # print(contours[i].shape)
        approx = cv2.approxPolyDP(contours[i], 20, True)
        cv2.drawContours(image1, [approx], 0, (0,255,0), 1)

        con_list = contours[i].tolist()

        border = 5

        # left, right
        x_l, x_r = con_list[0][0][0]-border, con_list[2][0][0]+border 
        
        # up, down
        y_u, y_d = con_list[0][0][1]-border, con_list[1][0][1]+border

        crop_img = image2[y_u:y_d, x_l:x_r]
        cv2.imshow('crop_img{}'.format(i), crop_img)


        gray1 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray{}'.format(i), gray1)

        histoNorm = cv2.equalizeHist(gray1)
        cv2.imshow('histoNorm{}'.format(i), histoNorm)

        ret, th1 = cv2.threshold(histoNorm, 190, 235, cv2.THRESH_BINARY)
        cv2.imshow('th1{}'.format(i), th1)

        thresh = cv2.threshold(th1, 175, 235, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cv2.imshow('thresh{}'.format(i), thresh)

        cell_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('cell_close{}'.format(i), cell_close)

        cell_erode = cv2.erode(cell_close, kernel2, iterations=1)
        cv2.imshow('cell_erode{}'.format(i), cell_erode)

        contours2, hierarchy = cv2.findContours(cell_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = cv2.drawContours(crop_img, contours2, -1, (0, 255, 0), 2)
        cv2.imshow('img_contour{}'.format(i), img_contour)

        M = cv2.moments(contours2[0])
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        print('center_x = {}, center_y = {}'.format(center_x, center_y))
        img_center = cv2.circle(img_contour, (center_x,center_y), 7, 128, -1)#繪製中心點

        cv2.imshow('img_center{}'.format(i), img_center)
        point.append((center_x, x_l, center_y, y_u))

      else:
        continue

    for j in range(len(point)):
      font = cv2.FONT_HERSHEY_SIMPLEX
      text = '(' + str(point[j][0] + point[j][1]) + ',' + str(point[j][2] + point[j][3]) + ')'
      cv2.putText(image2, text, (point[j][0] + point[j][1]+5, point[j][2] + point[j][3]-5), font, 0.5, (255 ,0 ,0), 1, cv2.LINE_AA)

    cv2.imshow('final_image'.format(i), image2)
    cv2.waitKey(0)

    if cv2.waitKey(0) == 27:
        ok = False
            
  cv2.destroyAllWindows()

main(yolo_path, face_path)