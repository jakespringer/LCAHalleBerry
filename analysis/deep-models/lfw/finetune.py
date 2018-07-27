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
norm_freq = load.normalize_frequencies_function(num_classes=10)
X_train, y_train = load.get_dataset('/home/jspringer/Workspace/lfw/dataset/train', preprocessing=[norm_freq, load.shuffle])
X_test, y_test = load.get_dataset('/home/jspringer/Workspace/lfw/dataset/validate')
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=5.0,
    width_shift_range=0.10,
    height_shift_range=0.10,
    channel_shift_range=0.1,
    shear_range=2.5,
    zoom_range=0.05,
    horizontal_flip=True)
datagen.fit(X_train)

target_model_name = sys.argv[1]
checkpoint = None if len(sys.argv) < 3 else sys.argv[2]
if target_model_name == 'resnet50':
    weights = 'imagenet' if checkpoint is None else 'weights/LFW_ResNet50.h5.{}'.format(checkpoint)
    model = models.LFW_ResNet50(weights=weights)
elif target_model_name == 'inceptionv3':
    weights = 'imagenet' if checkpoint is None else 'weights/LFW_InceptionV3.h5.{}'.format(checkpoint)
    model = models.LFW_InceptionV3(weights=weights)
elif target_model_name == 'vgg16':
    weights = 'imagenet' if checkpoint is None else 'weights/LFW_VGG16.h5.{}'.format(checkpoint)
    model = models.LFW_VGG16(weights=weights)
elif target_model_name == 'mobilenetv2':
    weights = 'imagenet' if checkpoint is None else 'weights/LFW_MobileNetV2.h5.{}'.format(checkpoint)
    model = models.LFW_MobileNetV2(weights=weights)
elif target_model_name == 'densenet121':
    weights = 'imagenet' if checkpoint is None else 'weights/LFW_DenseNet121.h5.{}'.format(checkpoint)
    model = models.LFW_DenseNet121(weights=weights)
else:
    print('Invalid model')
    exit(1)

# Run model
if checkpoint is None:
     model.train(X_train, y_train, 
                 generator=datagen, 
                 validation_data=(X_test, y_test),
                 epochs=5,
                 shuffle=True,
                 save_on_best=False,
                 pretraining_stage=True)

model.train(X_train, y_train,
            generator=datagen,
            validation_data=(X_test, y_test),
            shuffle=True,
            save_on_best=True,
            pretraining_stage=False)
