import models
import load_lfw as load
import sys
import os
import keras
import numpy as np
import imageio
import cv2

import keras.backend as K

# Important global data
classes = ['Other', 'Halle Berry']
K.set_learning_phase(0)

# Grab training/testing data
norm_freq = load.normalize_frequencies_function(num_classes=2)
X_adversarial, y_adversarial = load.get_dataset('./dataset/adversarial')
y_adversarial = keras.utils.to_categorical(y_adversarial - 1, num_classes=2)
X_noisy, y_noisy = load.get_dataset('./dataset/noisy')
y_noisy = keras.utils.to_categorical(y_noisy - 1, num_classes=2)
X_benign, y_benign = load.get_dataset('./dataset/benign')
y_benign = keras.utils.to_categorical(y_benign - 1, num_classes=2)

modelset = [
    models.HB_ResNet50(weights='weights/HB_ResNet50.h5.final'),
    models.HB_InceptionV3(weights='weights/HB_InceptionV3.h5.final'),
    models.HB_VGG16(weights='weights/HB_VGG16.h5.final'),
    models.HB_MobileNetV2(weights='weights/HB_MobileNetV2.h5.final'),
    models.HB_DenseNet121(weights='weights/HB_DenseNet121.h5.final')
]

def resize(img, ratio=1/2):
    down = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, None, fx=1/ratio, fy=1/ratio, interpolation=cv2.INTER_CUBIC)
    return up

X_downup_benign = np.array([resize(img) for img in X_benign])
X_downup_adversarial = np.array([resize(img) for img in X_adversarial])
X_downup_noisy = np.array([resize(img) for img in X_noisy])

for m in modelset:
    print('\n{}, benign'.format(type(m).__name__))
    m.test_accuracy(X_benign, y_benign)
    print('\n{}, adversarial'.format(type(m).__name__))
    m.test_accuracy(X_adversarial, y_adversarial)
    print('\n{}, noisy'.format(type(m).__name__))
    m.test_accuracy(X_noisy, y_noisy)
    print('\n{}, downscaled-upscaled benign'.format(type(m).__name__))
    m.test_accuracy(X_downup_benign, y_benign)
    print('\n{}, downscaled-upscaled adversarial'.format(type(m).__name__))
    m.test_accuracy(X_downup_adversarial, y_adversarial)
    print('\n{}, downscaled-upscaled noisy'.format(type(m).__name__))
    m.test_accuracy(X_downup_noisy, y_noisy)
    print('-------------------------------------------------\n')

def write_image(info, directory):
    i, (x, y) = info

    if not os.path.exists('dataset-downscale/{}/{}'.format(directory, np.argmax(y) + 1)):
        os.makedirs('dataset-downscale/{}/{}'.format(directory, np.argmax(y) + 1))

    filename = 'dataset-downscale/{}/{}/{}.png'.format(directory, np.argmax(y) + 1, i)
    open('dataset-downscale/faces_' + directory + '.txt', 'a').write(os.path.abspath(filename) + '\n')
    imageio.imwrite(filename, x)
    print('Wrote', filename)

for i in enumerate(zip((X_downup_benign), y_benign)):
    write_image(i, 'benign')

for i in enumerate(zip((X_downup_adversarial), y_adversarial)):
    write_image(i, 'adversarial')

for i in enumerate(zip((X_downup_noisy), y_noisy)):
    write_image(i, 'noisy')
