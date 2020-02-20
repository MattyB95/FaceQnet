import os

import cv2
import numpy as np
from keras.models import load_model

import PreprocessImage

batch_size = 1

INPUT_DIRECTORY = "/Volumes/Samsung_T5/Data Collection"
MODALITY = "/Face/"
IMAGE_FILE_EXTENSION = ".jpg"


def get_requested_images():
    requested_images = list()
    for (directory_path, directory_names, filenames) in os.walk(INPUT_DIRECTORY):
        for filename in filenames:
            if (MODALITY in directory_path) and filename.endswith(IMAGE_FILE_EXTENSION):
                requested_images.append(os.path.join(directory_path, filename))
    return requested_images


def load_test():
    X_test = []
    images_names = []
    images = get_requested_images()
    print('Read test images')
    for image in images:
        print(image)
        imread = PreprocessImage.load_image_file(image)
        img = cv2.resize(imread, (224, 224))
        X_test.append(img)
        images_names.append(image)
    return X_test, images_names


def read_and_normalize_test_data():
    test_data, images_names = load_test()
    test_data = np.array(test_data, copy=False, dtype=np.float32)
    return test_data, images_names


model = load_model('FaceQnet.h5')
test_data, images_names = read_and_normalize_test_data()
y = test_data
m = 0.7
s = 0.5
score = model.predict(y, batch_size=batch_size, verbose=1)
predictions = score
fichero_scores = open('scores_quality.txt', 'w')
i = 0
fichero_scores.write("img;score\n")
for item in predictions:
    fichero_scores.write("%s" % images_names[i])
    if float(predictions[i]) < 0:
        predictions[i] = '0'
    elif float(predictions[i]) > 1:
        predictions[i] = '1'
    fichero_scores.write(";%s\n" % predictions[i])
    i = i + 1
fichero_scores.close()
