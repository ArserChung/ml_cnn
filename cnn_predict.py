from keras.models import load_model
import numpy as np
import cv2 as cv

model = load_model('cnn_recognize_pet.h5')

test_feature = np.load('test_feature.npy')
test_label = np.load('test_label.npy')

test_feature = test_feature.astype('float32')
test_feature = test_feature/255.0

pet = {
        0 : 'cat',
        1 : 'dog'
    }

print(test_feature[0].shape)
def show_image(image,label):

    cv.imshow(str(pet[label]),image)
    cv.waitKey(0)
    cv.destroyAllWindows()

prediction = model.predict(test_feature)



for i in range(len(test_label)):
    show_image(test_feature[i],test_label[i])
    arg = np.argmax(prediction[i])
    pre = pet[arg]
    print("model predict:",pre)

