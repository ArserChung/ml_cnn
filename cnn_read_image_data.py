import glob as gb
import cv2 as cv
from sklearn.model_selection import train_test_split
import numpy as np
import os

images = []
label = []

files_cat = gb.glob("PetImages\Cat\*")
files_dog = gb.glob("PetImages\Dog\*")

print(len(files_dog),len(files_cat))
#                12500 12500

print(len(files_cat))  #<class 'list'>

size = (40,40)

for path_cat in files_cat:
    print(path_cat+"資料讀取中")
    try:
        img = cv.imread(path_cat)
        if img is not None:
            img = cv.resize(img,size)
            images.append(img)
            label.append(0)
    except:
        print(path_cat+"載入失敗")
    else:
        print(path_cat+"資料存取成功")

cat_len = len(images)

for path_dog in files_dog:
    print(path_dog+"資料讀取中")
    try:
        img = cv.imread(path_dog)
        if img is not None:
            img = cv.resize(img,size)
            images.append(img)
            label.append(1)
    except:
        print(path_dog+"載入失敗")
    else:
        print(path_dog+"資料存取成功")

dog_len = len(images) - cat_len

print("成功讀取"+str(cat_len)+"張貓貓照片")
print("成功讀取"+str(dog_len)+"張狗狗照片")
print("總共有"+str(len(label))+"張照片")

train_feature,test_feature,train_label,test_label = \
    train_test_split(images,label,test_size=0.2,random_state=42)
# train_test_split()可以幫助資料打散，並且分割測試、訓練組
#test_size = 0.2 mean all_data*20% =test_data
# random_state mean 隨機切分的固定次數(洗牌的概念，用相同厚度每次洗牌)
#把0~100是貓 101~200是狗的排隨機打散(42per)
train_feature = np.array(train_feature)
test_feature = np.array(test_feature)
train_label = np.array(train_label)
test_label = np.array(test_label)

print("train_feature shpae="+str(train_feature.shape))
print("test feature shpae="+str(test_feature.shape))
print("train label shpae="+str(train_label.shape))
print("test_label shpae="+str(test_label.shape))


#np.save(fileName.npy,num arrary,[allow_pickle=True],
#                                   [fix_import=True])
#fix_import 能不能兼容不同版本numpy
#allow_pickle能否使用python_pickle
np.save('train_feature.npy',train_feature)
np.save('train_label.npy',train_label)
np.save('test_feature.npy',test_feature)
np.save('test_label.npy',test_label)

