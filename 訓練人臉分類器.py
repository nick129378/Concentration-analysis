# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 04:51:10 2019

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

# 使用Tensorflow的Facenet模型
with tf.Graph().as_default():
    with tf.Session() as sess:
        datadir = IMG_OUT_PATH # 經過偵測、對齊 & 裁剪後的人臉圖像目錄
        # 取得人臉類別(ImageClass)的列表與圖像路徑
        dataset = facenet.get_dataset(datadir)        
        # 原始: 取得每個人臉圖像的路徑與標籤
        paths, labels,labels_dict = facenet.get_image_paths_and_labels(dataset)        
        print('Origin: Number of image: %d' % len(labels))
        print('Origin: Number of images: %d' % len(paths))
        print('Origin: Number of classes: %d' % len(labels_dict))
        
        
        # 由於lfw的人臉圖像集中有很多的人臉類別只有1張的圖像, 對於訓練來說樣本太少
        # 因此我們只挑選圖像樣本張數大於5張的人臉類別
        
        # 過濾: 取得每個人臉圖像的路徑與標籤 (>=5)
        """
        paths, labels, labels_dict = facenet.get_image_paths_and_labels(dataset, enable_filter=True, filter_size=5)        
        print('Filtered: Number of classes: %d' % len(labels_dict))
        print('Filtered: Number of images: %d' % len(paths))
        """
        # 載入Facenet模型
        print('Loading feature extraction model')
        modeldir =  FACENET_MODEL_PATH #'/..Path to Pre-trained model../20170512-110547/20170512-110547.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        # 打印"人臉特徵向量"的向量大小
        print("Face embedding size: ", embedding_size)
        
        # 計算人臉特徵向量 (128 bytes)
        print('Calculating features for images')
        batch_size = 1000 # 批次量
        image_size = 160  # 要做為Facenet的圖像輸入的大小
        
        nrof_images = len(paths) # 總共要處理的人臉圖像
        # 計算總共要跑的批次數
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        # 構建一個變數來保存"人臉特徵向量"
        emb_array = np.zeros((nrof_images, embedding_size)) # <-- Face Embedding
        
        for i in tqdm(range(nrof_batches_per_epoch)):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            
            # 序列化相關可重覆使用的資料

# 保存"人臉embedding"的資料
emb_features_file = open(os.path.join(DATA_PATH,'lfw_emb_features.pkl'), 'wb')
pickle.dump(emb_array, emb_features_file)
emb_features_file.close()

# 保存"人臉embedding"所對應的標籤(label)的資料
emb_lables_file = open(os.path.join(DATA_PATH,'lfw_emb_labels.pkl'), 'wb')
pickle.dump(labels, emb_lables_file)
emb_lables_file.close()
# 保存"標籤(label)對應到人臉名稱的字典的資料
emb_lables_dict_file = open(os.path.join(DATA_PATH,'lfw_emb_labels_dict.pkl'), 'wb')
pickle.dump(labels_dict, emb_lables_dict_file)
emb_lables_dict_file.close()
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
    
print("人臉embedding featues: {}, shape: {}, type: {}".format(len(emb_features), emb_features.shape, type(emb_features)))
print("人臉embedding labels: {}, type: {}".format(len(emb_labels), type(emb_labels)))
print("人臉embedding labels dict: {}, type: {}", len(emb_labels_dict), type(emb_labels_dict))

# 準備相關變數
X_train = []; y_train = []
X_test = []; y_test = []

# 保存己經有處理過的人臉label
processed = set()

# 分割訓練資料集與驗證資料集
for (emb_feature, emb_label) in zip(emb_features, emb_labels):
    if emb_label in processed:
        X_train.append(emb_feature)
        y_train.append(emb_label)
    else:
        X_test.append(emb_feature)
        y_test.append(emb_label)
        processed.add(emb_label)

# 結果
print('X_train: {}, y_train: {}'.format(len(X_train), len(y_train)))
print('X_test: {}, y_test: {}'.format(len(X_test), len(y_test)))


# 訓練分類器
print('Training classifier')
linearsvc_classifier = LinearSVC(C=1, multi_class='ovr')

# 進行訓練
linearsvc_classifier.fit(X_train, y_train)

# 使用驗證資料集來檢查準確率
score = linearsvc_classifier.score(X_test, y_test)

# 打印分類器的準確率
print("Validation result: ", score)
# 序列化"人臉辨識模型"到檔案
classifier_filename = SVM_MODEL_PATH

# 產生一個人臉的人名列表，以便辨識後來使用
#class_names = [cls.name.replace('_', ' ') for cls in dataset]

class_names = []
for key in sorted(emb_labels_dict.keys()):
    class_names.append(emb_labels_dict[key].replace('_', ' '))

# 保存人臉分類器到檔案系統
with open(classifier_filename, 'wb') as outfile:
    pickle.dump((linearsvc_classifier, class_names), outfile)
    
print('Saved classifier model to file "%s"' % classifier_filename)

# 反序列化相關可重覆使用的資料
# "人臉embedding"的資料