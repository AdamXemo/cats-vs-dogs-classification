import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ['Dog', 'Cat']
IMAGE = 'images/glib.jpg'

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model('load/cats_n_dogs.h5')

prediction = model.predict([prepare(IMAGE)])

print(prediction)

if prediction < 0.40:
    prediction = 0
else:
    prediction = 1


print('Neural network prediction:', CATEGORIES[prediction])

plt.imshow(prepare(IMAGE)[0], cmap=plt.cm.binary)
plt.show()

