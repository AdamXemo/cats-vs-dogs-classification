import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle


DATADIR = 'kagglecatsanddogs_5340/PetImages'
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 50

training_data = []

def create_training_data():
    for category in CATEGORIES:
        # Path to cats or dogs directory
        path = os.path.join(DATADIR, category)
        # Labels
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                # if we use rgb (bgr cuz cv2) we have to deal with 3d arrays. To gray color
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                # Resizing image
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # Training data
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

# We can have like 70% dogs and 30% cats and our nn wont be learning cuz it just can predict dogs and get accuracy. We need to prevent this

# We need to shuffle our data
random.shuffle(training_data)

# Train
X = []
# Labels
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

#                                          this 1 because its gray scale
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)