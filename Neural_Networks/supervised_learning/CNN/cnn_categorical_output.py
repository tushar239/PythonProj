import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# ################ Part 1 - Data Preprocessing ###########

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255, # normalize pixel values (0-255 → 0-1). Each pixel has a value from 0-255. Normalizing it to 0-1 by dividing original value by 255.
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set', # path to training data
                                                 # target_size = (150, 150), # made the training slower.
                                                 target_size = (64, 64), # resize images. Resizing an image means changing its dimensions (width and height), which can affect its file size and visual appearance. This can involve either increasing the size (upscaling) or decreasing the size (downscaling) of the image.
                                                 batch_size = 32, # number of images per batch
                                                 class_mode = 'binary') # for binary classification; use 'categorical' for multi-class
# Found 8000 images belonging to 2 classes(cats and dogs).
# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Found 2000 images belonging to 2 classes(cats and dogs).

# ################# Part 2 - Building the CNN ###############
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               activation='relu',
                               input_shape=[64, 64, 3]))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
# we have categorical output (cat and dog), so units=2 and activation=softmax
'''
Binary classification (2 classes, 0/1 labels)
    When Labels: [0, 1, 0, 1, ...] (shape (batch_size,))
    Dense(1, activation="sigmoid")
    loss="binary_crossentropy"

Categorical classification (2 classes, one-hot labels)
    When Labels: [[1,0], [0,1], ...] (shape (batch_size, 2))
    Dense(2, activation="softmax")
    loss="categorical_crossentropy"

Categorical classification with integer labels
    When Labels: [0, 1, 0, 1, ...] (shape (batch_size,), NOT one-hot)
    Dense(2, activation="softmax")
    loss="sparse_categorical_crossentropy"

Rule of thumb:

    binary_crossentropy → 1 output neuron (sigmoid), labels = 0/1
    categorical_crossentropy → N output neurons (softmax), labels = one-hot
    sparse_categorical_crossentropy → N output neurons (softmax), labels = integers
    
One-Hot Encoding (for labels)
    One-hot encoding is a way of representing categorical class labels as vectors where:
    The correct class = 1
    All other classes = 0

Example

Suppose you have 3 classes:
    Class 0 = Cat
    Class 1 = Dog
    Class 2 = Horse
    
If labels are integers:
    y = [0, 1, 2, 1, 0]

One-hot encoded labels:
    y_onehot = [
      [1, 0, 0],   # Cat
      [0, 1, 0],   # Dog
      [0, 0, 1],   # Horse
      [0, 1, 0],   # Dog
      [1, 0, 0]    # Cat
    ]

    
'''
cnn.add(tf.keras.layers.Dense(units=2, activation='softmax'))

# ############## Part 3 - Training the CNN #################
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
'''
Note that with each epoch, accuracy is increasing and loss is decreasing.
If you do not augment the training_set images, you will observe high accuracy for training_set(around 98%), but low accuracy for test_set.
So, you will see overfitting. 

Epoch 1/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 201s 774ms/step - accuracy: 0.6094 - loss: 0.6515 - val_accuracy: 0.6590 - val_loss: 0.6121
Epoch 2/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 43s 173ms/step - accuracy: 0.6998 - loss: 0.5794 - val_accuracy: 0.6940 - val_loss: 0.5788
Epoch 3/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 28s 111ms/step - accuracy: 0.7308 - loss: 0.5364 - val_accuracy: 0.7615 - val_loss: 0.4959
Epoch 4/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 121ms/step - accuracy: 0.7516 - loss: 0.5125 - val_accuracy: 0.7460 - val_loss: 0.5147
Epoch 5/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 29s 116ms/step - accuracy: 0.7591 - loss: 0.4989 - val_accuracy: 0.7700 - val_loss: 0.5040
Epoch 6/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 44s 174ms/step - accuracy: 0.7632 - loss: 0.4813 - val_accuracy: 0.7730 - val_loss: 0.4779
Epoch 7/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 41s 165ms/step - accuracy: 0.7821 - loss: 0.4576 - val_accuracy: 0.7400 - val_loss: 0.5284
Epoch 8/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 25s 102ms/step - accuracy: 0.7857 - loss: 0.4484 - val_accuracy: 0.7470 - val_loss: 0.5205
Epoch 9/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 26s 102ms/step - accuracy: 0.7945 - loss: 0.4271 - val_accuracy: 0.7730 - val_loss: 0.4830
Epoch 10/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 43s 173ms/step - accuracy: 0.8098 - loss: 0.4122 - val_accuracy: 0.7710 - val_loss: 0.4912
Epoch 11/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 116ms/step - accuracy: 0.8192 - loss: 0.4024 - val_accuracy: 0.7575 - val_loss: 0.5102
Epoch 12/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 119ms/step - accuracy: 0.8183 - loss: 0.3947 - val_accuracy: 0.7975 - val_loss: 0.4601
Epoch 13/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 38s 153ms/step - accuracy: 0.8279 - loss: 0.3768 - val_accuracy: 0.7840 - val_loss: 0.4976
Epoch 14/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 39s 156ms/step - accuracy: 0.8325 - loss: 0.3681 - val_accuracy: 0.7930 - val_loss: 0.4642
Epoch 15/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 39s 155ms/step - accuracy: 0.8399 - loss: 0.3618 - val_accuracy: 0.7995 - val_loss: 0.4640
Epoch 16/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 38s 152ms/step - accuracy: 0.8455 - loss: 0.3507 - val_accuracy: 0.7800 - val_loss: 0.5136
Epoch 17/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 43s 172ms/step - accuracy: 0.8510 - loss: 0.3314 - val_accuracy: 0.8010 - val_loss: 0.5044
Epoch 18/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 41s 166ms/step - accuracy: 0.8633 - loss: 0.3128 - val_accuracy: 0.8040 - val_loss: 0.4782
Epoch 19/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 45s 180ms/step - accuracy: 0.8720 - loss: 0.3015 - val_accuracy: 0.7930 - val_loss: 0.5318
Epoch 20/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 46s 185ms/step - accuracy: 0.8786 - loss: 0.2892 - val_accuracy: 0.8075 - val_loss: 0.4671
Epoch 21/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 33s 132ms/step - accuracy: 0.8776 - loss: 0.2873 - val_accuracy: 0.8110 - val_loss: 0.4991
Epoch 22/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 122ms/step - accuracy: 0.8834 - loss: 0.2753 - val_accuracy: 0.8030 - val_loss: 0.4986
Epoch 23/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 37s 148ms/step - accuracy: 0.8827 - loss: 0.2708 - val_accuracy: 0.7980 - val_loss: 0.5276
Epoch 24/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 46s 185ms/step - accuracy: 0.8903 - loss: 0.2572 - val_accuracy: 0.8005 - val_loss: 0.5065
Epoch 25/25
250/250 ━━━━━━━━━━━━━━━━━━━━ 43s 173ms/step - accuracy: 0.8985 - loss: 0.2462 - val_accuracy: 0.7965 - val_loss: 0.5157
'''

# ################### Part 4 - Making a single prediction ##################
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
print(result) # [[0. 1.]]

print(training_set.class_indices) # {'cats': 0, 'dogs': 1}

# softmax function returns a probability value for each category in such a way that sum becomes 1
# if result[0][0] = 0.7 and result[0][1] = 0.3, it means that prediction is a cat
if result[0][0] >= 0.5:
  prediction = 'cat'
else:
  prediction = 'dog'

print(prediction) # cat


