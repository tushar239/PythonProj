# https://www.geeksforgeeks.org/getting-started-scikit-image-image-processing-python/

# Python3 program to process
# images using scikit-image
import os

# importing io from skimage
import skimage
from skimage import io

# way to load car image from file
#file = os.path.join(skimage.data_dir, 'cc.jpg')

cars = io.imread('Ganeshji.webp')
print(type(cars)) # <class 'numpy.ndarray'>
print(cars)
'''
[[[22 22 22]
  [22 22 22]
  [22 22 22]
  ...
  [22 22 24]
  [27 26 29]
  [41 40 43]]

 [[22 22 22]
  [22 22 22]
  [22 22 22]
  ...
  [15 15 17]
  [21 21 23]
  [35 35 37]]

 [[22 22 22]
  [22 22 22]
  [22 22 22]
  ...
  [19 18 21]
  [24 24 26]
  [38 38 40]]

 ...

 [[21 21 21]
  [ 7  7  7]
  [ 2  2  2]
  ...
  [21 22 28]
  [23 24 31]
  [30 31 37]]

 [[27 27 27]
  [13 13 13]
  [ 8  8  8]
  ...
  [23 24 31]
  [25 26 33]
  [32 33 40]]

 [[30 30 30]
  [16 16 16]
  [12 12 12]
  ...
  [25 26 33]
  [28 29 35]
  [35 36 42]]]

'''

# way to show the input image
io.imshow(cars)
io.show()