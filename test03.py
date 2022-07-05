import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('LBUTTONDBLCLK')
        print(x, ',', y)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(x) + ',' + str(y)
        cv2.putText(img, text, (x, y), font, 0.5, (255 ,255 ,0), 2, cv2.LINE_AA)

        cv2.imshow('img/image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        # print('RBUTTONDBLCLK')
        
        # img 內部照片是以 BGR 形式存在
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(blue) + ',' + str(green) + ',' + str(red)
        cv2.putText(img, text, (x, y), font, 0.5, (0 ,255 ,255), 2, cv2.LINE_AA)

        print(str(blue) + ',' + str(green) + ',' + str(red))

        cv2.imshow('img/image', img)

img = cv2.imread('img/cluster2_0000.png')
cv2.imshow('img/image', img)

cv2.setMouseCallback('img/image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows