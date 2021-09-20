# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 03:52:57 2020

@author: nick
"""

from scipy import misc
import tensorflow as tf
import detect_face
import facenet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from tqdm import tqdm
import pickle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import visualization_utils as vis_utils
import warnings
warnings.filterwarnings("ignore")

from models.detector import face_detector

import datetime

#%pylab inline

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
gpu_memory_fraction=1.0
image_size=182
margin=44
nrof_successfully_aligned = 0
input_image_size = 160


# 專案的根目錄路徑
ROOT_DIR = os.getcwd()

# 訓練/驗證用的資料目錄
DATA_PATH = os.path.join(ROOT_DIR, "data")

# 模型的資料目錄
MODEL_PATH = os.path.join(ROOT_DIR, "model")

# FaceNet的模型
FACENET_MODEL_PATH = os.path.join(MODEL_PATH, "facenet","20170512-110547","20170512-110547.pb")

# Classifier的模型
SVM_MODEL_PATH = os.path.join(MODEL_PATH, "svm", "lfw_svm_classifier.pkl")

# 訓練/驗證用的圖像資料目錄
IMG_IN_PATH = os.path.join(DATA_PATH, "lfw")

# 訓練/驗證用的圖像資料目錄
IMG_OUT_PATH = os.path.join(DATA_PATH, "lfw_crops")

# MTCNN的模型
MTCNN_MODEL_PATH = os.path.join(MODEL_PATH, "mtcnn")

def is_same_person(face_emb, face_label, threshold=1.1):
    emb_distances = []
    emb_features = emb_dict[face_label]
    for i in range(len(emb_features)):
        emb_distances.append(facenet.distance(face_emb, emb_features[i]))
    
    # 取得平均值
    if np.mean(emb_distances) > threshold: # threshold <1.1 代表兩個人臉非常相似 
        return False
    else:
        return True
# 反序列化相關可重覆使用的資料
# "人臉embedding"的資料
with open(os.path.join(DATA_PATH,'lfw_emb_features.pkl'), 'rb') as emb_features_file:
    emb_features =pickle.load(emb_features_file)

# "人臉embedding"所對應的標籤(label)的資料
with open(os.path.join(DATA_PATH,'lfw_emb_labels.pkl'), 'rb') as emb_lables_file:
    emb_labels =pickle.load(emb_lables_file)

# "標籤(label)對應到人臉名稱的字典的資料
with open(os.path.join(DATA_PATH,'lfw_emb_labels_dict.pkl'), 'rb') as emb_lables_dict_file:
    emb_labels_dict =pickle.load(emb_lables_dict_file)
emb_dict = {} # key 是label, value是embedding list
for feature,label in zip(emb_features, emb_labels):
    # 檢查key有沒有存在
    if label in emb_dict:
        emb_dict[label].append(feature)
    else:
        emb_dict[label] = [feature]
        
        
face_distance_threshold = 1.1

# 計算一個人臉的embedding是不是歸屬某一個人
# 根據Google Facenet的論文, 透過計算兩個人臉embedding的歐氏距離
# 0: 代表為同一個人臉 , threshold <1.1 代表兩個人臉非常相似 
# 創建Tensorflow Graph物件

tf_g = tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

# 創建Tensorflow Session物件
tf_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# 把這個Session設成預設的session
tf_sess.as_default()
# 載入MTCNN模型 (偵測人臉位置)
#pnet, rnet, onet = detect_face.create_mtcnn(tf_sess, MTCNN_MODEL_PATH)

# 載入Facenet模型
print('Loading feature extraction model')
modeldir =  FACENET_MODEL_PATH #'/..Path to Pre-trained model../20170512-110547/20170512-110547.pb'
facenet.load_model(modeldir)

# 取得模型的輸入與輸出的佔位符
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

# 打印"人臉特徵向量"的向量大小
print("Face embedding size: ", embedding_size)
# 載入SVM分類器模型
classifier_filename = SVM_MODEL_PATH

with open(classifier_filename, 'rb') as svm_model_file:
    (face_svc_classifier, face_identity_names) = pickle.load(svm_model_file)
    HumanNames = face_identity_names    #訓練時的人臉的身份
    
    print('load classifier file-> %s' % classifier_filename)
    print(face_svc_classifier)
    
    
    
number = 0
def resize_image(im, max_size=768):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
        return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
    return im

fd = face_detector.FaceAlignmentDetector(
    lmd_weights_path="./models/detector/FAN/2DFAN-4_keras.h5"# 2DFAN-4_keras.h5, 2DFAN-1_keras.h5
)    
print('Start Recognition!')



#main


cap = cv2.VideoCapture(0)

#cap.open()

while(cap.isOpened()):
    ret, image = cap.read()
    im = image
    #im = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
   #im = cv2.imread("images/224.jpg")[..., ::-1]
    #plt.imshow(im)
   #im = resize_image(im) # Resize image to prevent GPU OOM.
    h, w, _ = im.shape




    bboxes, landmarks = fd.detect_face(im, with_landmarks=True)
    # Display landmarks
    #plt.figure(figsize=(15,8))
    num_faces = len(bboxes)
    print(num_faces)
    #print(landmarks)
    angle=[] #每個臉的角度
    con=[]
    for i in range(num_faces):
     try:
     
     
     
        landmark=landmarks[i]          #  X向下 Y向右   # landmarks[i] 一個代表一個臉座標
        #print(landmark)
        x3,y3 = landmark[2]            #crop下來判斷是誰 計算角度 紀錄兩個陣列 名子 角度  角度做判斷看誰誤差大
        x15,y15 = landmark[14]         #根據I查出名子
        x31,y31 = landmark[30]
        xmid = (x3 + x15) / 2
        ymid = (y3 + y15) / 2 
        y90 = ymid - y3
        yang  = ymid - y31 
        angle.append(math.floor(90 * yang / y90)) 
        #angles= math.floor(90 * yang / y90)
        
        x1,y1 = landmark[0]
        x17,y17 = landmark[16]
        x20,y20 = landmark[19]
        x25,y25 = landmark[24]
        
        if x20 > x1 or x25 > x17:
         con.append(1)
        else :
         con.append(0)   
        
        

        #print(y3 ,y15 ,y31 , ymid , y90 , yang  )   #負角度代表向右偏 正角度代表向左偏
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
       
         #image=fd.draw_landmarks(image, landmarks[i], color=(0,255,0))
         #plt.imshow(fd.draw_landmarks(im, landmarks[i], color=(0,255,0)))
         #plt.imshow(image)
      
         cropped=image[x0-10:x1+10,y0-10:y1+10,]
      
         cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB) # 把BGR轉換成RGB
  
    
        
        
        
      
            
         # 根據邊界框的座標來進行人臉的裁剪
      
      
      
         #misc.imsave('1.png', cropped)
         cropped = facenet.flip(cropped, False)
         #misc.imsave('2.png', cropped)
         scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
         scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                               interpolation=cv2.INTER_CUBIC)
         #misc.imsave('3.png', scaled)
        
         scaled = facenet.prewhiten(scaled)
         scaled_reshape = scaled.reshape(-1,input_image_size,input_image_size,3)      
         feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
        
         # 進行臉部特徵擷取
         emb_array = np.zeros((1, embedding_size))
         emb_array[0, :] = tf_sess.run(embeddings, feed_dict=feed_dict)
        
         # 步驟 #3.進行人臉識別分類
         face_id_idx = face_svc_classifier.predict(emb_array)   
            
         if is_same_person(emb_array, int(face_id_idx), 1.1):            
            face_id_name = HumanNames[int(face_id_idx)] # 取出人臉的名字
            bb_color = vis_utils.STANDARD_COLORS[i] # 給予不同的顏色
            bb_fontcolor = 'black'
         else:
            face_id_name = 'Unknown'
            bb_color = 'BlueViolet' # 給予紅色
            bb_fontcolor = 'white'
       
      
       
         #ang = str(angle[i])
         if angle[i] > 0:
             ang ='左偏'+ str(angle[i])
         elif angle[i] < 0:
             ang ='右偏'+ str(angle[i])
         else :
             ang=angle[i]
             
         #co = str(con[i])
         if con[i] == 0:
             co ='抬頭'
         else :   
             co ='低頭'
         bb_color = vis_utils.STANDARD_COLORS[i] # 給予不同的顏色
         bb_fontcolor = 'black'
        
      
        
        
         with open("紀錄.txt","a+") as f:
            
          concentrate = face_id_name +':'+ ang +':'+ co +'\n'
          #concentrate = ': '+ str(ang) +'\n'
          #concentrate = str.decode('utf8') 
          f.write(concentrate) #这句话自带文件关闭功能，不需要再写f.close()
         #concentrate = concentrate.decode('utf8') 
        # 進行視覺化的展現
         vis_utils.draw_bounding_box_on_image_array(image,x0-10,y0-10,x1+10,y1+10,
                                                 color=bb_color,
                                                 thickness=2,
                                                 display_str_list=[concentrate],
                                                 fontname='simsun.ttc',         # <-- 替換不同的字型
                                                 fontsize=20,                     # <-- 根據圖像大小設定字型大小
                                                 fontcolor=bb_fontcolor,
                                                 use_normalized_coordinates=False)
        
      except:
         pass 
  
    cv2.imshow("Face Detection", image)
  
    if cv2.waitKey(1) &  0xFF == ord('q'):
       cap.release()
       break
 
cap.release()
cv2.destroyAllWindows()

#misc.imsave('drawn.png', image)

"""
# 設定展示的大小
plt.figure(figsize=(20,10))

# 展示偵測出來的結果
plt.imshow(draw[:,:,::-1]) # 轉換成RGB來給matplotlib展示
plt.show()

"""

#rgb_image=cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)






