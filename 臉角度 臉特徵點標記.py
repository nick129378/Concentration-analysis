# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:36:35 2019

@author: nick
"""
from scipy import misc
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from matplotlib import pyplot as plt
import visualization_utils as vis_utils
import math

from models.detector import face_detector




number = 0
def resize_image(im, max_size=768):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
        return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
    return im

# Test images are obtained on https://www.pexels.com/
#im = cv2.imread("images/224.jpg")[..., ::-1]
im = cv2.imread("images/123.jpeg")[..., ::-1]
plt.imshow(im)
im = resize_image(im) # Resize image to prevent GPU OOM.
h, w, _ = im.shape


fd = face_detector.FaceAlignmentDetector(
    lmd_weights_path="./models/detector/FAN/2DFAN-4_keras.h5"# 2DFAN-4_keras.h5, 2DFAN-1_keras.h5
)
"""
bboxes = fd.detect_face(im, with_landmarks=False)
assert len(bboxes) > 0, "No face detected."



# Display detected face
x0, y0, x1, y1, score = bboxes[0] # show the first detected face
x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
print(y0,y1,x0,x1)
plt.imshow(im[x0:x1, y0:y1, :])
"""

bboxes, landmarks = fd.detect_face(im, with_landmarks=True)
# Display landmarks
#plt.figure(figsize=(15,8))
num_faces = len(bboxes)
print(num_faces)
#print(landmarks)
angle=[] #每個臉的角度
for i in range(num_faces):
  try:
     landmark=landmarks[i]          #  X向下 Y向右   # landmarks[i] 一個代表一個臉座標
     print(landmark)
     x3,y3 = landmark[2]            #crop下來判斷是誰 計算角度 紀錄兩個陣列 名子 角度  角度做判斷看誰誤差大
     x15,y15 = landmark[14]         #根據I查出名子
     x31,y31 = landmark[30]
     xmid = (x3 + x15) / 2
     ymid = (y3 + y15) / 2 
     y90 = ymid - y3
     yang  = ymid - y31 
     angle.append(math.floor(90 * yang / y90)) 

     print(y3 ,y15 ,y31 , ymid , y90 , yang  )   #負角度代表向右偏 正角度代表向左偏
  except:
      pass

print(angle)

image = im
image_size = 182
for i in range(num_faces):
  try:
      x0 ,y0 ,x1 , y1, score = bboxes[i]
      x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
      #print(y0,y1,x0,x1)
      #plt.subplot(1, num_faces, i+1)
       
      image=fd.draw_landmarks(image, landmarks[i], color=(0,255,0))
      #plt.imshow(fd.draw_landmarks(im, landmarks[i], color=(0,255,0)))
      #plt.imshow(image)
      """
      crop=image[x0-10:x1+10,y0-10:y1+10,]
      scaled = misc.imresize(crop, (image_size, image_size), interp='bilinear')
      #crop=im[x0-10:x1+10,y0-10:y1+10,]
      #plt.imshow(crop)
      
      name = str(angle[i])
      output_filename_n = 'images/images/'+str(name)+'.png'
      misc.imsave(output_filename_n, scaled)
      
      name = str(angle[i])
      bb_color = vis_utils.STANDARD_COLORS[i] # 給予不同的顏色
      bb_fontcolor = 'black'
      
      
      vis_utils.draw_bounding_box_on_image_array(image,x0-10,y0-10,x1+10,y1+10,
                                                 color=bb_color,
                                                 thickness=2,
                                                 display_str_list=[name],
                                                 fontname='calibrib.ttf',         # <-- 替換不同的字型
                                                 fontsize=20,                     # <-- 根據圖像大小設定字型大小
                                                 fontcolor=bb_fontcolor,
                                                 use_normalized_coordinates=False)
      
        
        """
        
  except:
     pass
        
misc.imsave('images/images/1.png', image)
















