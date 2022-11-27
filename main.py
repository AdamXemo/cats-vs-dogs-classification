import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

NAME = 'Cats-and-dogs-cnn-64x2-{}'.format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

classes = ['Dog', 'Cat']

X = pickle.load(open('cats_dogs_classification/X.pickle', 'rb'))
y = pickle.load(open('cats_dogs_classification/y.pickle', 'rb'))

X = X/255.0

model = keras.models.load_model('cats_dogs_classification/cats_n_dogs.h5')

model = Sequential()

'''model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

model.save('cats_n_dogs.h5')'''


predictions = model.predict(X[0], use_multiprocessing=True)

print(predictions)

for i in range(len(X)):
        plt.grid(False)
        # Showing image
        plt.imshow(X[i], cmap=plt.cm.binary)
        # Writing number label
        plt.xlabel(f'Animal shown: {classes[y[i]]}')
        # Writing Neural Network predict
        #  plt.title(f'Neural Network predict: {classes[np.argmax(predictions[i])]}')
        # Showing all of that stuff
        plt.show()