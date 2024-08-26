from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Input,Dense

model = Sequential()

#必須先建立輸入層，無法用Dense or Conv2D去建立輸入層
model.add(
    Input(
        shape=(28,28,1)
    )
)

#加入卷積層
model.add(
    Conv2D(
        filters=10,#濾鏡個數
        kernel_size=(3,3), #設定濾鏡的大小
        # input_shape = (28,28,1), 與mlp相同原因，不支援參
        #必須另外add輸入層
        padding='same', #設定卷積運算圖片的大小，same為與原始圖片相同，也就是28*28
        activation='relu'
    )
)
#加入池化層
model.add(
    MaxPooling2D(
        pool_size=(2,2) #變成(14,14)*10張進入下個卷積層
    )
)
#第二層卷積層
model.add(
    Conv2D(
        filters=20, #將14*14做卷積處理(濾鏡)，共20張
        kernel_size=(3,3),
        padding='same', #表示:20張不同濾鏡大小和上一層一樣的照片(14*14)
        activation='relu' #小於0的特徵歸於0
    )
)
#第二層池化層。
model.add(
    MaxPooling2D(
        pool_size=(2,2) #將14*14不同濾鏡下的照片做池化處理，變成(7*7)20張
    )
)

#建立平坦層、隱藏層、輸出層(與MLP相同)
