# https://www.geeksforgeeks.org/getting-started-scikit-image-image-processing-python/

'''
scikit-image is an image processing Python package that works with NumPy arrays
which is a collection of algorithms for image processing.

It is built on the top of NumPy, SciPy, and matplotlib.
'''

# importing data from skimage
from skimage import data # you have to install scikit-image package

camera = data.camera()

# An image with 512 rows
# and 512 columns
print(type(camera)) # <class 'numpy.ndarray'>

print(camera.shape) # (512, 512)