# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:04:39 2022

@author: Charles
"""
import os
import numpy as np
import cv2

## Instantiating a small convnet

from keras import layers
from keras import models
from keras.utils import to_categorical

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

## Adding a classifier on top of the convnet

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))

model.summary()

cur_dir = os.getcwd()
## Extracting the data from the dataset
trainpath = cur_dir + '\\Digits\\Digits_Dataset_Training\\'
testpath = cur_dir + '\\Digits\\Digits_Dataset_Testing\\'

def create_data(inputpath):
    x = []
    y = []    
    for i in range(12):
        if i < 10:
            path = inputpath + str(i) + '\\'
        elif i == 10:
            path = inputpath + 'B\\'
        else:
            path = inputpath + 'E\\'
        tlist = os.listdir(path)
        for j in range(len(tlist)):
               img = cv2.imread(os.path.join(path, tlist[j]), 0)
               x.append(img)
               y.append(i) 
    return np.array(x), np.array(y)

x_train, y_train = create_data(trainpath)
x_test, y_test = create_data(testpath)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

train_images = x_train.astype('float32') / 255
test_images = x_test.astype('float32') / 255

## Training Model
train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=64)

## Testing Model
test_loss, test_acc = model.evaluate(test_images, test_labels)

## Saving Model and Weights
model.save(cur_dir + '\\Models\\DigitsModel.h5')
model.save_weights(cur_dir + '\\Models\\DigitsModelWeights.h5')