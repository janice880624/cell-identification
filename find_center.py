import cv2
import numpy as np

path = "exp20/img(500).png"
# path = "aaa.png"

def main(image_path):

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

  ok = True
  while ok:
    image = cv2.imread(image_path)  
    # cv2.imshow('image', image)

    # 灰階
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    # 二值化
    ret, thr = cv2.threshold(gray, 50, 255, cv2.THRESH_OTSU)
    # cv2.imshow('thr', thr)

    contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours[1])

    for i in range(len(contours)):
      if len(contours[i])==4 and contours[i][0][0].tolist() != [0, 0]:
        print(len(contours[i]))
        print(contours[i].shape)
        approx = cv2.approxPolyDP(contours[i], 20, True)
        cv2.drawContours(image,[approx], 0, (0,255,0), 1)
        # cv2.imshow('ori_image{}'.format(i), image)

        con_list = contours[i].tolist()

        print(con_list[0][0]) # 左上
        print(con_list[1][0]) # 左下
        print(con_list[2][0]) # 右下
        print(con_list[3][0]) # 右上

        border = 10
        # left, right
        x_l, x_r = con_list[0][0][0]-border, con_list[2][0][0]+border 
        
        # up, down
        y_u, y_d = con_list[0][0][1]-border, con_list[1][0][1]+border

        crop_img = image[y_u:y_d, x_l:x_r]
        # cv2.imshow('crop_img{}'.format(i), image)

        gray1 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray1,150 , 255, cv2.THRESH_BINARY)
        # cv2.imshow('th1{}'.format(i), th1)

        erosion = cv2.erode(th1 , kernel, iterations=2)
        dilation = cv2.dilate(th1 , kernel2, iterations = 2)
        cv2.imshow('dilation{}'.format(i), dilation)

        canny = cv2.Canny(dilation, 50, 250)
        # cv2.imshow('canny{}'.format(i), canny)

        contours1, hierarchy1 = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours1))

        # color_img = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('color_img{}'.format(i), color_img)
        
        thresh = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cv2.imshow('thresh{}'.format(i), thresh)

        contours2, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_contour = cv2.drawContours(crop_img, contours2, -1, (0, 255, 0), 2)
        cv2.imshow('img_contour{}'.format(i), img_contour)


      else:
        continue

    cv2.imshow('final_image'.format(i), image)
    cv2.waitKey(0)

    if cv2.waitKey(0) == 27:
        ok = False
            
  cv2.destroyAllWindows()

main(path)