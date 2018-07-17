import models
import load_lfw as load
import sys
import os
import keras

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

target_model_name = sys.argv[1]
if target_model_name == 'resnet50':
    model = models.HB_ResNet50()
elif target_model_name == 'inceptionv3':
    model = models.HB_InceptionV3()
else:
    print('Invalid model')
    exit(1)

# Run model
model.train(X_train, y_train, 
            generator=datagen, 
            validation_data=(X_test, y_test),
            epochs=1,
            shuffle=True,
            save_on_best=False,
            pretraining_stage=True)

model.train(X_train, y_train,
            generator=datagen,
            validation_data=(X_test, y_test),
            shuffle=True,
            save_on_best=True)
