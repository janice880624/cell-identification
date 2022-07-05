# 形態學轉換
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 以灰階讀取
img = cv2.imread('img/cluster2_0000.png', cv2.IMREAD_GRAYSCALE)

# 做黑白二值化後反轉
_, mask = cv2.threshold(img, 0, 10, cv2.THRESH_BINARY_INV)

kernal = np.ones((5,5), np.uint8)

# 膨脹 (變胖)
# 如果內核下的至少一個像素為“1”，則像素元素為“1”
dilation = cv2.dilate(mask, kernal, iterations=2)

# 腐蝕（變瘦）
# 只有當內核下的所有像素都是1時，原始圖像中的像素（1或0）才會被視為1，否則它將被侵蝕（變為零
erosion = cv2.erode(mask, kernal, iterations=1)

# 開運算
# 腐蝕再膨脹
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

# 閉運算
# 先膨脹再腐蝕，可用於關閉前景對象內的小孔或對像上的小黑點
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)


titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing']
images = [img, mask, dilation, erosion, opening, closing]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()