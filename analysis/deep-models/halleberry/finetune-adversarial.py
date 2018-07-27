import models
import load_lfw as load
import sys
import os
import keras
import sys
import numpy as np

import keras.backend as K

# Important global data
classes = ['Other', 'Halle Berry']
K.set_learning_phase(0)

# Grab training/testing data
norm_freq = load.normalize_frequencies_function(num_classes=2)
X_train, y_train = load.get_dataset('/home/jspringer/Workspace/hb/dataset/train', preprocessing=[norm_freq, load.shuffle])
X_test, y_test = load.get_dataset('/home/jspringer/Workspace/hb/dataset/test')
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10.0,
    width_shift_range=0.25,
    height_shift_range=0.25,
    channel_shift_range=0.2,
    shear_range=5.0,
    zoom_range=0.1,
    horizontal_flip=True)
datagen.fit(X_train)

adam_params = { 'lr': 1e-5, 'decay': 1e-7 }

modelset = [
#    lambda: models.HB_ResNet50(adam_params=adam_params, weights='weights/HB_ResNet50.h5.adversarial.6'),
#    lambda: models.HB_InceptionV3(adam_params=adam_params, weights='weights/HB_InceptionV3.h5.final'),
    lambda: models.HB_VGG16(adam_params=adam_params, weights='weights/HB_VGG16.h5.final'),
    lambda: models.HB_MobileNetV2(adam_params=adam_params, weights='weights/HB_MobileNetV2.h5.adversarial.10'),
    lambda: models.HB_DenseNet121(adam_params=adam_params, weights='weights/HB_DenseNet121.h5.final')
]

num_iter = 10

# Augmented data includes benign + adversarial + noisy
X_augmented, y_augmented = np.ndarray((len(X_train)*(num_iter+1), 64, 64, 3)), np.ndarray((len(X_train)*(num_iter+1), 2))

for model in modelset:
    model = model()
    i = 0
    for X, y in datagen.flow(X_train, y_train, batch_size=len(X_train)):
        X_benign, X_adversarial, X_noisy = model.make_adversarial_and_noisy_batch(X, y, eps=8.0, num_iter=5, batch_size=32)
        X_augmented[:len(X_benign)] = X_benign
        X_augmented[(i+1)*len(X_benign):(i+2)*len(X_benign)] = X_adversarial
        y_augmented[:len(X_benign)] = y
        y_augmented[(i+1)*len(X_benign):(i+2)*len(X_benign)] = y

        print('Training {}, iteration {}'.format(type(model).__name__, i))
        model.train(X_augmented[:(i+2)*len(X_benign)], y_augmented[:(i+2)*len(X_benign)],
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    save_on_best=False,
                    epochs=30,
                    pretraining_stage=False)
        model.save('weights/' + type(model).__name__ + '.h5.adversarial.' + str(i+1))
        model.test(X_test, y_test)
        print()
        if i >= num_iter-1:
            break
        else:
            i += 1
