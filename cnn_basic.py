from keras.models import Sequential
from keras.layers import *

def create_cnn():

    model = Sequential()

    #必須先建立輸入層，無法用Dense or Conv2D去建立輸入層
    model.add(
        Input(
            shape=(40,40,3)
        )
    )

    #加入卷積層
    model.add(
        Conv2D(
            filters=10,#濾鏡個數，(40,40,3)變成10個不同濾鏡
            kernel_size=(5,5), #設定濾鏡的大小
            # input_shape = (28,28,1), 與mlp相同原因，不支援參
            #必須另外add輸入層Input()
            padding='same', #設定卷積運算圖片的大小，same為與原始圖片相同，也就是28*28
            activation='relu'
        )
    )
    #加入池化層
    model.add(
        MaxPooling2D(
            pool_size=(2,2) #變成(20,20)*10張進入下個卷積層
        )
    )

    #10%進入拋棄層，以防overfitting
    model.add(
        Dropout(0.1)
    )

    #第二層卷積層
    model.add(
        Conv2D(
            filters=20, #將20*20做卷積處理(濾鏡)，共20張不同濾鏡
            kernel_size=(5,5),
            padding='same', #表示:20張不同濾鏡大小和上一層一樣的照片(20*20)
            activation='relu' #小於0的特徵歸於0
        )
    )
    #第二層池化層。
    model.add(
        MaxPooling2D(
            pool_size=(2,2) #將10*10不同濾鏡下的照片做池化處理，變成(10*10)20張
        )
    )
    #drog 20%
    model.add(
        Dropout(0.2)
    )

    #建立平坦層、隱藏層、輸出層(與MLP相同)

    #create 平坦層
    #第二層池化層出來的20張10*10卷積運算圖片，轉換成
    # vetor 10*10*20=2000
    #的一維向量 ，2000個float數字，這就是神經元輸入數目
    model.add(
        Flatten()
    )

    #create hiding layer
    model.add(
        Dense(
            units = 512, #512隱藏層
            activation='relu'
        )
    )
    #create output layer
    model.add(
        Dense(
            units=2, #輸出只會有貓與狗，兩個答案
            activation='softmax'
        )
    )
    return model
