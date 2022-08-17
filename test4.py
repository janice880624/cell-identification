import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

def histogram(f):
  if f.ndim != 3:
    hist = cv2.calcHist([f], [0], None, [256], [0, 256])
    plt.plot(hist)
  else:
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
      hist = cv2.calcHist([f], [i], None, [256], [0, 256])
      plt.plot(hist, color = col)
  plt.xlim([0, 256])
  plt.xlabel('Intensity')
  plt.ylabel('#intensities')
  plt.show()

def pixelVal(pix, r1, s1, r2, s2):
  if (0 <= pix and pix <= r1):
    return (s1 / r1) * pix
  elif (r1 < pix and pix <= r2):
    return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
  else:
    return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

def main():

  ok = True

  while ok:

    # read original image
    image = cv2.imread("face2/cell (1).png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray.png', gray)
    # histogram(gray)

    blurM = cv2.medianBlur(gray, 5)
    cv2.imshow('blurM.png', blurM)
    
    histoNorm = cv2.equalizeHist(gray)
    cv2.imshow('histoNorm.png', histoNorm)

    thresh, img2 = cv2.threshold(histoNorm, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('thresholding', img2)

    # clahe = cv2.createCLAHE(clipLimit = 0.23, tileGridSize=(8, 8))
    # claheNorm = clahe.apply(histoNorm)
    # cv2.imshow('claheNorm.png', claheNorm)

    # r1 = 130
    # s1 = 0
    # r2 = 210
    # s2 = 255
    
    # Vectorize the function to apply it to each value in the Numpy array.
    # pixelVal_vec = np.vectorize(pixelVal)
    
    # Apply contrast stretching.
    # contrast_stretched = pixelVal_vec(claheNorm, r1, s1, r2, s2)
    # contrast_stretched_blurM = pixelVal_vec(blurM, r1, s1, r2, s2)
    
    # cv2.imshow('contrast_stretch.png', contrast_stretched)
    # cv2.imshow('contrast_stretch_blurM.png',
    #             contrast_stretched_blurM)
    
    # edge detection using canny edge detector
    # edge = cv2.Canny(gray, 30, 200)
    # cv2.imshow('edge.png', edge)
    
    # edgeG = cv2.Canny(blurG, 100, 200)
    # cv2.imshow('edgeG.png', edgeG)
    
    # edgeM = cv2.Canny(blurM, 100, 200)
    # cv2.imshow('edgeM.png', edgeM)

    cv2.waitKey(0)
    if cv2.waitKey(0) == 27:
      ok = False

  cv2.destroyAllWindows()

main()