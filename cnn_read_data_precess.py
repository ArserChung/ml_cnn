import numpy as np
from cnn_basic import create_cnn

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

model = create_cnn()
try:
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print("model compile.....")
except:
    print("compile fail,check in cnn_baisc.py")
    pass
else :
    print("sucess compile model")

model.fit(
    x=train_feature_normal,
    y=train_label_onehot,
    validation_split=0.2,
    epochs=10,
    batch_size=200,
    verbose=2
)
try:
    model.save('cnn_recognize_pet.h5')
except:
    print("save the model fail")
else:
    _,score = model.evaluate(test_feature_normal,test_label_onehot)
    print("accuracy="+str(score))
    print("model saving sucess")
    del model