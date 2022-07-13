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

