# https://www.geeksforgeeks.org/image-processing-in-python/

'''

cv2.resize(src, dsize,interpolation)
Here,
src          :The image to be resized.
dsize        :The desired width and height of the resized image.
interpolation:The interpolation method to be used.

- When the python image is resized, the interpolation method defines how the new pixels are computed.
There are several interpolation techniques, each of which has its own quality vs. speed trade-offs.
- It is important to note that resizing an image can reduce its quality.
This is because the new pixels are calculated by interpolating between the existing pixels, and this can introduce some blurring.
'''

# Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Ganeshji.webp')

'''
The main difference between BGR and RGB is the order in which the color channels are specified for each pixel in an image:

BGR Stands for blue-green-red, and is often associated with libraries like OpenCV
RGB Stands for red-green-blue, and is more commonly used in general-purpose digital imaging 

The program that reads and interprets the image file determines how the image's color channels are interpreted. 
For example, the OpenCV library (cv2) uses BGR by default when reading images in Python, while the PIL library uses RGB. 

RGB is an additive color model that uses red, green, and blue to create color by mixing light. 
In an RGB image, each pixel is represented by three values (red, green, blue) ranging from 0 to 255. 

'''
# Convert BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the scale factor
# Increase the size by 3 times
scale_factor_1 = 3.0
# Decrease the size by 3 times
scale_factor_2 = 1/3.0

print(image_rgb.shape) # (560, 500, 3)

# Get the original image dimensions
height, width = image_rgb.shape[:2]

# Calculate the new image dimensions
new_height = int(height * scale_factor_1)
new_width = int(width * scale_factor_1)

# Resize the image
zoomed_image = cv2.resize(src =image_rgb,
                          dsize=(new_width, new_height),
                          interpolation=cv2.INTER_CUBIC)

# Calculate the new image dimensions
new_height1 = int(height * scale_factor_2)
new_width1 = int(width * scale_factor_2)

# Scaled image
scaled_image = cv2.resize(src= image_rgb,
                          dsize =(new_width1, new_height1),
                          interpolation=cv2.INTER_AREA)

# Create subplots
# read about subplots in subplots.py
fig, axs = plt.subplots(1, 3, figsize=(10, 4))

# Plot the original image
axs[0].imshow(image_rgb)
axs[0].set_title('Original Image Shape:'+str(image_rgb.shape))

# Plot the Zoomed Image
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image Shape:'+str(zoomed_image.shape))

# Plot the Scaled Image
axs[2].imshow(scaled_image)
axs[2].set_title('Scaled Image Shape:'+str(scaled_image.shape))

# Remove ticks from the subplots
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

# Display the subplots
plt.tight_layout()
plt.show()