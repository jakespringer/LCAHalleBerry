import models
import load_lfw as load
import sys
import os
import keras
import principle_curvature as pc

import keras.backend as K

# Important global data
classes = ['Other', 'Halle Berry']
K.set_learning_phase(0)

# Grab training/testing data
norm_freq = load.normalize_frequencies_function(num_classes=2)
X_test, y_test = load.get_dataset('/home/jspringer/Workspace/hb/dataset/holdout')
X_test = X_test[y_test == 1][:1]
y_test = y_test[y_test == 1][:1]
y_test = keras.utils.to_categorical(y_test, num_classes=2)

modelset = [
    lambda: models.HB_ResNet50(weights='weights/HB_ResNet50.h5.final'),
    lambda: models.HB_InceptionV3(weights='weights/HB_InceptionV3.h5.final'),
    lambda: models.HB_VGG16(weights='weights/HB_VGG16.h5.final'),
    lambda: models.HB_MobileNetV2(weights='weights/HB_MobileNetV2.h5.final'),
    lambda: models.HB_DenseNet121(weights='weights/HB_DenseNet121.h5.final')
]

target_model_name = sys.argv[1]
if target_model_name == 'resnet50':
    model = modelset[0]() 
elif target_model_name == 'inceptionv3':
    model = modelset[1]()
elif target_model_name == 'vgg16':
    model = modelset[2]()
elif target_model_name == 'mobilenetv2':
    model = modelset[3]()
elif target_model_name == 'densenet121':
    model = modelset[4]()
else:
    print('Invalid model')
    exit(1)

graph = pc.principle_curvatures_graph(model.model, 0)
e, v = pc.principle_curvatures(model.model, 0, X_test, y_test, graph=graph)
print(e)
print(v)

#for m in modelset:
#    print('\n{}, benign'.format(type(m).__name__))
#    m.test_accuracy(X_benign, y_test)
#    print('\n{}, adversarial'.format(type(m).__name__))
#    m.test_accuracy(X_adversarial, y_test)
#    print('\n{}, noisy'.format(type(m).__name__))
#    m.test_accuracy(X_noisy, y_test)
#    print('-------------------------------------------------\n')
