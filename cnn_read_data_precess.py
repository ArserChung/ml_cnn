import numpy as np
from cnn_basic import create_cnn
from keras.models import Sequential
#loading npy
train_feature = np.load('train_feature.npy')
train_label = np.load('train_label.npy')
test_feature = np.load('test_feature.npy')
test_label = np.load('test_label.npy')

# print(train_feature.shape) #(19956, 40, 40, 3)

# print(test_feature.shape) #(4990, 40, 40, 3)
#diminsion=4 (num,dim_row,dim_column,RBG)

#tras element 2 float32
train_feature =train_feature.astype('float32')
test_feature =test_feature.astype('float32')
#normalize
train_feature_normal = train_feature/255
test_feature_normal = test_feature/255


#one-hog encdding for label
train_label_onehot = np.eye(2)[train_label]
test_label_onehot = np.eye(2)[test_label]

model = Sequential()

model = create_cnn()