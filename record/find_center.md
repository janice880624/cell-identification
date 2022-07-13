# cell identifier

## find_center test01

> 台拉力細胞辨識

### step.1

> Convert an image from one color space to another

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### step.2

> If the pixel value pixel is greater than the threshold , it is specified as white, otherwise it is black

```python
ret, thr = cv2.threshold(gray, 50, 255, cv2.THRESH_OTSU)
```

### step.3

> Find the target contour and circle it

```python
contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

### step.4 

> Choose the range we want

```python
len(contours[i])==4 and contours[i][0][0].tolist() != [0, 0]
```

```python
# left, right
x_l, x_r = con_list[0][0][0]-border, con_list[2][0][0]+border 

# up, down
y_u, y_d = con_list[0][0][1]-border, con_list[1][0][1]+border

crop_img = image[y_u:y_d, x_l:x_r]

gray1 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
```

### step.5

> Find cell boundaries

```python
contours2, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_contour = cv2.drawContours(crop_img, contours2, -1, (0, 255, 0), 2)
```

### step.6 

> Find the center of gravity

```python
M = cv2.moments(contours2[0])
center_x = int(M["m10"] / M["m00"])
center_y = int(M["m01"] / M["m00"])
print('center_x = {}, center_y = {}'.format(center_x, center_y))
```