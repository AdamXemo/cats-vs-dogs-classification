import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.callbacks import TensorBoard
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time


asd = 20

loh = False

NAME = 'Cats-and-dogs-cnn-64x2-{}'.format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

classes = ['Dog', 'Cat']

X = pickle.load(open('load/X.pickle', 'rb'))
y = pickle.load(open('load/y.pickle', 'rb'))

X = X/255.0

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

# Training our model for many times with different amount of layers, layer sizes and dense layers, and all information during training we save with tensorboard, then analyse, find patterns and choose the best model configuration.
'''for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)
            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:], padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3), padding='same', activation='relu'))          
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))

            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard], workers=4, use_multiprocessing=True)'''


model = keras.models.load_model('load/cats_n_dogs.h5')

print(X.shape[1:])

'''model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = X.shape[1:], padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))          
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))          
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard], workers=4, use_multiprocessing=True)
'''

model.evaluate(X_test, y_test)

predictions = model.predict([X_test])

model.save('load/cats_n_dogs.h5')
predictions_formated = []

for i in range(len(X_test)):
        if predictions[i] < 0.45:
            predictions_formated.append(0)
        elif predictions[i] > 0.55:
            predictions_formated.append(1)
        else:
            predictions_formated.append(0)

        plt.grid(False)
        # Showing image
        plt.imshow(X_test[i], cmap=plt.cm.binary)
        # Writing number label
        plt.xlabel(f'Animal shown: {classes[y_test[i]]}')
        # Writing Neural Network predict
        plt.title(f'Neural Network prediction: {classes[predictions_formated[i]]}')
        # Showing all of that stuff
        plt.show()