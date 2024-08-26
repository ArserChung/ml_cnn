# CNN(Convolution Nerual Network)
## INTRO ME
if u have any opnion or problem . pls mail or mess me
every sorce are open to learing or create new pro . 
## Basis strutrue
![unnamed](https://github.com/user-attachments/assets/b7f03c0e-f078-4502-8320-91d457321f00)

## 卷積的基本原理
卷積就是將原始圖片與設定的濾鏡(feature detector)進行卷積運算，你也可以將卷積運算看成是原始圖片濾鏡特效的處理，每一個濾鏡(卷積核)都會以亂數處理的方式產生不同的卷積運算，因此可以得到不同的濾鏡特效效果，增加圖片的數量

### 卷積操作
卷積層通過一個小的濾波器（濾鏡）（或卷積核）在輸入數據上滑動，進行點積運算，生成特徵圖（Feature Map）。卷積操作的步驟如下：
1. **選擇濾波器**：濾波器是一個小矩陣，通常比輸入數據要小（例如 3x3 或 5x5 的矩陣）。
2. **滑動濾波器**：從輸入數據的左上角開始，濾波器在數據上逐步滑動。
3. **點積運算**：濾波器與當前覆蓋的輸入數據區域進行逐元素的點積運算，然後將結果求和。
4. **生成特徵圖**：每次點積運算的結果會填入特徵圖中的一個位置，最終生成特徵圖。


### 卷積核（濾波器）
卷積核是一組參數，通常是一個小矩陣，它通過學習來檢測輸入中的特定特徵（如邊緣、紋理等）。每個卷積層可以包含多個卷積核，每個卷積核會生成一個特徵圖。

### 步幅 (Stride)
步幅決定了濾波器在輸入數據上滑動的步長。較大的步幅會導致較小的特徵圖，因為濾波器覆蓋的數據區域會減少。

### 填充 (Padding)
填充是在輸入數據的邊緣添加額外的像素（通常是零），以保證濾波器可以完整覆蓋邊緣數據。常見的填充方式有：
- 無填充 (Valid)：不進行填充，導致輸出特徵圖變小。
- 全填充 (Same)：填充以保證輸出特徵圖的大小與輸入數據相同。

### Code example
```python
from keras.models import Sequential
from keras.layers import Conv2D

# 创建一个 Sequential 模型
model = Sequential()

# 添加一个卷积层
model.add(
  Conv2D(
    filters=10,
    kernel_size=(3, 3),
    padding = 'same',
    activation='relu',
    input_shape=(28, 28, 1)
  )
)
```
- filters:設定濾鏡(卷積核)個數，每一個filter都會產生不同的濾淨特效效果
- kernel_size:設定濾鏡(卷積核)大小(matrix)，一般為5*5或3*3*3的大小 (濾鏡(卷積核)，通常都是一個陣列的運算)
- input_shape:設定原始圖片的大小，(28,28,1)表示每一張圖片的大小為28*28
- activation:設定激勵函式，relu函式會將小於0的資訊設定為0


## 池化層(Pooling Layer)
池化層（Pooling Layer）是卷積神經網絡（CNN）中的一種重要層，它用於減少特徵圖（feature map）的空間尺寸，同時保持重要的信息。這種操作能夠幫助減少計算量並防止過擬合。常見的池化層操作有：
1. **平均池化（Average Pooling）**：在每個池化窗口的區域內計算平均值。與最大池化不同，平均池化保留了區域內所有值的信息，但可能會丟失一些細節。
2. **最大池化(Max Polling)**: 只挑出矩陣中最大的值，相當於只挑出圖片局部最明顯的特徵，這樣就可以所減卷積層產生的卷積運算圖片數量。

### Code
pool_size方法設定縮減的比率。
```python
model.add(
  MaxPooling2D(
    pool_size = (2,2)
  )
```
以上面來說，原來10張(filters=10)28*28(input_size=(28,28,1))的卷積運算圖片，經過pool_size=(2,2)的池化處理，就會得到10張14*14的卷積圖片，如果使用pool_size=(4,4)的池化處理則會得到10張7*7的卷積運算圖片
![unnamed (2)](https://github.com/user-attachments/assets/e49d0ee8-56c7-4426-99ad-16f86195d220)

