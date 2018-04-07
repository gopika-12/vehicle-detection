import csv
import cv2
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
import sklearn
import random
from pathlib import Path
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

batch_size = 32
n_epochs = 3

def generator(samples, batch_size=32):
    """Takes the a list of lines from a CSV and generates a list of images
    """
    n_samples = len(samples)
    while True: # Loop forever so the generator never exits, as expected by keras.fit_generator
        for offset in range(0, n_samples, batch_size):
            batch_filenames = samples[offset:offset+batch_size]
            images = []
            labels = []
            for filename in batch_filenames:
                filename = str(filename)
                if 'extra' in filename or 'image' in filename:
                    # It's a nonvehicle image
                    label = 0
                else:
                    label = 1
                image = cv2.imread(filename)
                if image is None:
                    print('Error! Cannot find image')
                    pdb.set_trace()
                images.append(image)
                labels.append(label)

            # After we've run through the entire batch, go through and augment the data via mirroring
            aug_images = []
            aug_labels = []
            for image, label in zip(images, labels):
                flipped_image = cv2.flip(image, 1)
                aug_images.append(flipped_image)
                aug_labels.append(label)
            images += aug_images
            labels += aug_labels
    
            X_train = np.array(images) 
            y_train = np.array(labels) 
            # yield (X_train, y_train) # inputs, targets
            # Shuffle again for some reason (this might accidentally unshuffle it and break everything)
            yield sklearn.utils.shuffle(X_train, y_train) # inputs, targets

car_imgs = list(Path('./data/vehicles').iterdir())
non_car_imgs = list(Path('./data/non-vehicles').iterdir())
n_cars = len(car_imgs)
n_non_cars = len(non_car_imgs)

all_imgs = car_imgs = non_car_imgs # List of all filenames
random.shuffle(all_imgs) # Shuffle the order
train, val = train_test_split(all_imgs, test_size=0.2) # Split the CSV into a test/val dataset

train_gen = generator(train, batch_size)
val_gen = generator(val, batch_size)

model = Sequential()
# Normalize the images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
model.add(Convolution2D(24, (5, 5), strides=2, activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=2, activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=2, activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
# Add dropout here?
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_gen, steps_per_epoch = len(train) / batch_size, \
    epochs=n_epochs, validation_data=val_gen, validation_steps = len(val) / batch_size) 


print('Saving model to file...')
model.save('model.h5')
