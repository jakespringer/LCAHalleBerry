import models
import load_lfw as load
import sys
import os
import keras
import sys

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
#    models.HB_ResNet50(adam_params=adam_params),
#    models.HB_InceptionV3(adam_params=adam_params),
#    models.HB_VGG16(adam_params=adam_params),
#    models.HB_MobileNetV2(adam_params=adam_params),
    models.HB_DenseNet121(adam_params=adam_params)
]

for model in modelset:
    print('Pre-training {}'.format(type(model).__name__))
    model.train(X_train, y_train, 
                 generator=datagen, 
                 validation_data=(X_test, y_test),
                 epochs=5,
                 shuffle=True,
                 save_on_best=False,
                 pretraining_stage=True)

    print('Training {}'.format(type(model).__name__))
    model.train(X_train, y_train,
                generator=datagen,
                validation_data=(X_test, y_test),
                shuffle=True,
                save_on_best=True,
                epochs=75,
                pretraining_stage=False)
    print()
