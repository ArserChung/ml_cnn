import glob as gb
import cv2 as cv

list_cat = []
list_dog = []

files_cat = gb.glob("PetImages\Cat\*")
files_dog = gb.glob("PetImages\Dog\*")

# print(len(files_dog),len(files_cat))
#                1250 12500

# print(len(files_cat))  <class 'list'>
cat_img = []
dog_img = []
img_size = (40,40)
for cat_path in files_cat:
    print(cat_path,"正在讀取中.....")
    try :
        img = cv.imread(cat_path)
        img = cv.resize(img,dsize=img_size)
        cat_img.append(img)
    except :
        print("載入照片失敗")
        pass
    else :
        print("載入成功，已存取")

print(len(cat_img))


