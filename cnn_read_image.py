import glob as gb
import cv2 as cv

list_cat = []
list_dog = []

files_cat = gb.glob("PetImages\Cat\*")
files_dog = gb.glob("PetImages\Dog\*")

print(len(files_cat))



