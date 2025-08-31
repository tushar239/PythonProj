import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# ################ Part 1 - Data Preprocessing ###########
'''
ImageDataGenerator is a utility in Keras that helps with:
    Data augmentation → generating new variations of images (rotated, zoomed, flipped, etc.) to prevent overfitting.
    Efficient data loading → feeding images to the model in batches, without loading the entire dataset into memory.

Read more about it in 'Udemy Course.docx'-> ImageDataGenerator in Code
'''

# Preprocessing the Training set
'''
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

- rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures
- width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally
- rescale is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor.
- shear_range is for randomly applying shearing transformations (https://en.wikipedia.org/wiki/Shear_mapping)
- zoom_range is for randomly zooming inside pictures
- horizontal_flip is for randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).
- fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
'''
'''
Image augmentation is required to avoid overfitting.
- If you train a model on limited images, it may just memorize them.
- Augmentation introduces variations → model learns general patterns (edges, shapes, textures) instead of memorizing exact images.
- In practice, objects appear in different orientations, lighting, backgrounds, and partial occlusions.
  Augmentation helps the model handle such unseen variations.
'''

train_datagen = ImageDataGenerator(rescale = 1./255, # normalize pixel values (0-255 → 0-1). Each pixel has a value from 0-255. Normalizing it to 0-1 by dividing original value by 255.
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set', # path to training data
                                                 # target_size = (150, 150), # made the training slower.
                                                 target_size = (64, 64), # resize images. Resizing an image means changing its dimensions (width and height), which can affect its file size and visual appearance. This can involve either increasing the size (upscaling) or decreasing the size (downscaling) of the image.
                                                 batch_size = 32, # number of images per batch
                                                 class_mode = 'binary') # for binary classification; use 'categorical' for multi-class
# Found 8000 images belonging to 3 classes.
# Preprocessing the Test set
# Unlike to training data set, we don't want to apply shearing, zooming and flipping transformations on test images.
# We do not want to increase total number of images in test data.
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Found 2000 images belonging to 3 classes.

# ################# Part 2 - Building the CNN ###############
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
# how to decide number of filters and it size?
# look at 'Udemy Course.docx -> Number of filters section'
cnn.add(tf.keras.layers.Conv2D(filters=32, # we are using a classic cnn architecture with filters=32
                               kernel_size=3, # shape of filter/feature detector/kernel will be 3x3
                               activation='relu',
                               input_shape=[64, 64, 3])) # we have resized the images to 64x64 using target_size in flow_from_directory(). Third parameter is 3 because these are color images. If they are black and white, then there should be 1.
'''
/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
'''
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) # pool_size=2 means 2x2 matrix will be used for pooling. strides=2 means matrix will move by 2 pixels. padding=valid(default) means you ignore missing cells when you are moving 2x2 matrix on the image. padding=same means pad extra cells with the same cell values which are available.

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               activation='relu')) # input_shape is required only for the first convolutional layer.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
# In ANN problem, we used smaller number of neurons(units) because that was a simple classification problem.
# Here, the problem is more complex as we are dealing with images, let's keep number of neurons to 128.
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
# we have binary output (cat or dog), so units=1 anc activation function is sigmoid
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# ############## Part 3 - Training the CNN #################
# Compiling the CNN
#'adam' optimizer performs Stochastic Gradient Descent.
#https://www.geeksforgeeks.org/deep-learning/optimizers-in-tensorflow/
# we have binary output, so loss function is binary_corssentropy
# Read 'crossentropy loss function' section in 'Udemy course.docx'
# accuracy metrics is the most relevant way to measure the performance of classification problem
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
# For each epoch, CNN will be trained with training_set and evaluated with test_set
# This is a bit different from what we used to before
# We used to train the NN with epochs and then used to evaluate just once.
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
'''
# Trying to fit training_set, evaluating test_set and predicting some images(test_set) just like how we did in ann. 
cnn.fit(x = training_set, epochs = 1)

#loss, accuracy = cnn.evaluate(test_set, verbose=0)
print(f"Test Loss: {loss}") #  0.6554234027862549
print(f"Test Accuracy: {accuracy}") # 0.6035000085830688

# Predicting the Test set results
y_pred = cnn.predict(test_set)
print(y_pred)

'''
# ################### Part 4 - Making a single prediction ##################
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_3.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
print(result) # [[1.]]
#training_set.class_indices

# sigmoid function returns a probability value between 0 and 1
# For us, labels are cat=0 and dog=1
# if result has value closer to 0, then prediction is a cat
# if result has value closer to 1, then prediction is a dog
if result[0][0] < 0.5:
  prediction = 'cat'
else:
  prediction = 'dog'

print(prediction) # dog


