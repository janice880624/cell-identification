import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

def show_img(img):
  plt.figure(figsize=(15,15)) 
  image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(image_rgb)
  plt.show()

def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
  brightness = 0
  contrast = -100

  B = brightness / 255.0
  c = contrast / 255.0 
  k = math.tan((45 + 44 * c) / 180 * math.pi)

  img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

  img = np.clip(img, 0, 255).astype(np.uint8)

  print("減少對比度 (白黑都接近灰，分不清楚): ")
  show_img(img)

  brightness = 0
  contrast = +200
  
  B = brightness / 255.0
  c = contrast / 255.0 
  k = math.tan((45 + 44 * c) / 180 * math.pi)

  img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    
  # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
  img = np.clip(img, 0, 255).astype(np.uint8)
  
  print("增加對比度 (白的更白，黑的更黑): ")
  show_img(img)

file_name = "img/cluster2_0000.png"
origin_img = cv2.imread(file_name)
print("origin picture:")
show_img(origin_img)

result_img = modify_contrast_and_brightness2(origin_img)