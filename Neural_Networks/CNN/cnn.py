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
cnn.add(tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               activation='relu',
                               input_shape=[64, 64, 3]))
'''
/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
'''
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# ############## Part 3 - Training the CNN #################
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
'''
Epoch 1/25
/usr/local/lib/python3.10/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
7/7 ━━━━━━━━━━━━━━━━━━━━ 5s 256ms/step - accuracy: 0.3230 - loss: -2.2601 - val_accuracy: 0.5000 - val_loss: -23.8118
Epoch 2/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 4s 121ms/step - accuracy: 0.4982 - loss: -41.1298 - val_accuracy: 0.5000 - val_loss: -160.0631
Epoch 3/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 126ms/step - accuracy: 0.5162 - loss: -223.8972 - val_accuracy: 0.5000 - val_loss: -633.7938
Epoch 4/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 123ms/step - accuracy: 0.4903 - loss: -810.8215 - val_accuracy: 0.5000 - val_loss: -1911.9957
Epoch 5/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 3s 136ms/step - accuracy: 0.5091 - loss: -2393.6680 - val_accuracy: 0.5000 - val_loss: -4851.8799
Epoch 6/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 4s 223ms/step - accuracy: 0.4933 - loss: -5910.8174 - val_accuracy: 0.5000 - val_loss: -10857.8613
Epoch 7/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.5140 - loss: -12052.2305 - val_accuracy: 0.5000 - val_loss: -21890.6836
Epoch 8/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 124ms/step - accuracy: 0.5167 - loss: -24240.6094 - val_accuracy: 0.5000 - val_loss: -40895.3164
Epoch 9/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 3s 119ms/step - accuracy: 0.4353 - loss: -49519.0586 - val_accuracy: 0.5000 - val_loss: -71038.4766
Epoch 10/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 120ms/step - accuracy: 0.5378 - loss: -72490.2500 - val_accuracy: 0.5000 - val_loss: -117384.2656
Epoch 11/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 120ms/step - accuracy: 0.5195 - loss: -122115.1406 - val_accuracy: 0.5000 - val_loss: -187146.6719
Epoch 12/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 118ms/step - accuracy: 0.4779 - loss: -202863.3438 - val_accuracy: 0.5000 - val_loss: -286768.2812
Epoch 13/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 3s 210ms/step - accuracy: 0.5105 - loss: -296021.3125 - val_accuracy: 0.5000 - val_loss: -422930.8125
Epoch 14/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 117ms/step - accuracy: 0.4695 - loss: -470339.0312 - val_accuracy: 0.5000 - val_loss: -610107.8125
Epoch 15/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 131ms/step - accuracy: 0.5197 - loss: -625722.1250 - val_accuracy: 0.5000 - val_loss: -856135.1875
Epoch 16/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 117ms/step - accuracy: 0.4777 - loss: -940437.4375 - val_accuracy: 0.5000 - val_loss: -1183422.0000
Epoch 17/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 116ms/step - accuracy: 0.5118 - loss: -1186474.0000 - val_accuracy: 0.5000 - val_loss: -1598235.2500
Epoch 18/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 116ms/step - accuracy: 0.4538 - loss: -1754937.0000 - val_accuracy: 0.5000 - val_loss: -2126164.5000
Epoch 19/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 116ms/step - accuracy: 0.5038 - loss: -2120881.2500 - val_accuracy: 0.5000 - val_loss: -2771290.7500
Epoch 20/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 116ms/step - accuracy: 0.5007 - loss: -2834427.0000 - val_accuracy: 0.5000 - val_loss: -3576980.0000
Epoch 21/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 230ms/step - accuracy: 0.5079 - loss: -3545062.5000 - val_accuracy: 0.5000 - val_loss: -4533354.5000
Epoch 22/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.4555 - loss: -4910181.0000 - val_accuracy: 0.5000 - val_loss: -5697485.0000
Epoch 23/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 126ms/step - accuracy: 0.4305 - loss: -6426874.0000 - val_accuracy: 0.5000 - val_loss: -7062758.5000
Epoch 24/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 115ms/step - accuracy: 0.4586 - loss: -7566496.5000 - val_accuracy: 0.5000 - val_loss: -8630248.0000
Epoch 25/25
7/7 ━━━━━━━━━━━━━━━━━━━━ 3s 136ms/step - accuracy: 0.4535 - loss: -9405489.0000 - val_accuracy: 0.5000 - val_loss: -10498646.0000
<keras.src.callbacks.history.History at 0x7b9deeed7640>
'''

# ################### Part 4 - Making a single prediction ##################
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction) # dog


