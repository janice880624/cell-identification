import cv2  
import numpy as np  
image_path = r"aaa.png"  # 图片路径  
img = cv2.imread(image_path)  
#转化成HSV颜色空间  
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
#筛选蓝色，得到二值图  
mask = cv2.inRange(hsv, (100,43,46), (124,255,255))  
#绘制框的图像  
imgResult = cv2.copyTo(img,mask)  
img_be = cv2.copyTo(img,mask)  
#设置核  
kernel = np.ones((6,6),np.uint8)  
#开运算  
# op =  cv2.MORPH_OPEN 进行开运算，指的是先进行腐蚀操作，再进行膨胀操作  
# op = cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作  
# 开运算：表示的是先进行腐蚀，再进行膨胀操作  
# 闭运算：表示先进行膨胀操作，再进行腐蚀操作  
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  
#寻找轮廓  
contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
#矩形列表  
box_ji=[]  
#根据轮廓绘制矩形  
for i in range(len(contours)):  
    area = cv2.contourArea(contours[i])  
    x, y, w, h = cv2.boundingRect(contours[i])  
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
    rect = cv2.minAreaRect(contours[i]) #提取矩形坐标  
    box = cv2.boxPoints(rect)  
    box = np.int0(box)  
    angle =abs(abs(rect[2])-45)  
    length = max(rect[1])  
    width = min(rect[1])  
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)  
    box_ji.append(box)  
#收集长度，根据长度进行匹配合适连接在一起的两个矩形  
length_ji=[]  
for b in box_ji:  
    length_ji.append(b[2][1]-b[1][1])  
#如果长度差在0.3之间，则匹配成功  
ans_ji=[]  
for l in range(len(length_ji)-1):  
    for j in range(l+1,len(length_ji)):  
        if -0.3<=length_ji[l]-length_ji[j]<=0.3:  
            ans_ji.append((l,j))  
#选择作为框的4个点，选择的顺序和矩形的边的位置和矩形的x，y坐标有关，为的是框选出最大的区域  
kuan=[]  
for a in ans_ji:  
    if box_ji[a[0]][0][0]<box_ji[a[1]][2][0]:  
        kuan.append([list(box_ji[a[0]][0]),list(box_ji[a[0]][3]),list(box_ji[a[1]][2]),list(box_ji[a[1]][1])])  
    else:  
        kuan.append([list(box_ji[a[0]][1]), list(box_ji[a[0]][2]), list(box_ji[a[1]][3]), list(box_ji[a[1]][0])])  
#绘制矩形  
for k in kuan:  
    kuang=cv2.rectangle(imgResult, k[1], k[3],  (0, 0, 255), thickness=1, lineType=4)  
  
  
cv2.imshow("begin", img_be)  
cv2.imshow("Mask", mask)  
cv2.imshow("kuang", img)  
# cv2.imshow("final",kuang)  
cv2.waitKey(0)  
cv2.destroyAllWindows()