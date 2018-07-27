import numpy as np
import keras
import tensorflow as tf
import sklearn
import sklearn.metrics
import adversarial

import resize_layer
import util as util

import keras.backend as K
K.set_learning_phase(0)

class HB_Model():
    def train(self, X_train, y_train, generator=None, batch_size=32, verbose=1, epochs=1000, validation_data=None, shuffle=True, current_epoch=0, save_on_best=True, pretraining_stage=False):
        if pretraining_stage:
            self.set_mode_pretrain()
        else:
            self.set_mode_train()

        X_train = self._preprocess(X_train)
        validation_data = self._preprocess(validation_data[0]), validation_data[1]
        callbacks = None if not save_on_best else [util.SaveOnBest(self.model, 
                                                        'weights/' + type(self).__name__ + '.h5', 
                                                        validation_data[0], 
                                                        validation_data[1])]
        if generator is None:
            self.model.fit(X_train,
                           y_train,
                           batch_size=batch_size,
                           verbose=verbose,
                           epochs=epochs,
                           validation_data=validation_data,
                           shuffle=shuffle,
                           callbacks=callbacks,
                           initial_epoch=current_epoch)
        else:
            self.model.fit_generator(generator.flow(X_train, y_train, batch_size=batch_size),
                                     verbose=verbose,
                                     epochs=epochs,
                                     steps_per_epoch=len(X_train) // batch_size,
                                     validation_data=validation_data,
                                     shuffle=shuffle,
                                     callbacks=callbacks,
                                     initial_epoch=current_epoch)

    def test(self, X_test, y_test, verbose=1, target_names=None):
        X_test = self._preprocess(X_test)
        predictions = self.model.predict(X_test, verbose=1)
        print(sklearn.metrics.classification_report(
            np.argmax(y_test, axis=1),
            np.argmax(predictions, axis=1),
            target_names=target_names))
        print('Accuracy:', sklearn.metrics.accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))

    def test_accuracy(self, X_test, y_test, verbose=1):
        X_test = self._preprocess(X_test)
        predictions = self.model.predict(X_test, verbose=1)
        print('Accuracy:', sklearn.metrics.accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))

    def save(self, filename):
        self.model.save_weights(filename)

    def make_adversarial_and_noisy(self, X, y, eps=0.7, min_lag=6, max_iter=100):
        X = self._preprocess(X)
        X_adv = np.ndarray(X.shape)
        X_noisy = np.ndarray(X.shape)
        adv_graph = adversarial.fgm_misclassify_graph(self.model)
        for i in range(len(X)):
            print('Beginning iteration', i)
            X_adv[i:i+1] = np.clip(
                adversarial.fgm_misclassify_images(
                    self.model,
                    X[i:i+1],
                    y[i:i+1],
                    max_iter=max_iter,
                    graph=adv_graph, eps=(self.max_color - self.min_color) * eps, min_lag=min_lag),
                self.min_color, self.max_color)

            adv_noise = X_adv[i:i+1] - X[i:i+1]
            means_stds = map(lambda img: (np.mean(img), np.std(img)), adv_noise)
            noise = np.array(list(map(lambda mean_std: np.random.normal(loc=mean_std[0], scale=mean_std[1], size=(64, 64, 3)), means_stds)))
            X_noisy[i:i+1] = np.clip(noise + (X[i:i+1]), self.min_color, self.max_color)
        X = self._deprocess(X)
        X_adv = self._deprocess(X_adv)
        X_noisy = self._deprocess(X_noisy)
        return X, X_adv, X_noisy

    def make_adversarial_and_noisy_batch(self, X, y, eps=0.7, num_iter=4, batch_size=32):
        X = self._preprocess(X)
        X_adv = np.ndarray(X.shape)
        X_noisy = np.ndarray(X.shape)
        adv_graph = adversarial.fgm_misclassify_graph(self.model)
        for i in range(len(X) // batch_size):
            print('Beginning iteration', i)
            X_adv[i*batch_size:(i+1)*batch_size] = np.clip(
                adversarial.fgm_misclassify_images(
                    self.model,
                    X[i*batch_size:(i+1)*batch_size],
                    y[i*batch_size:(i+1)*batch_size],
                    num_iter=num_iter,
                    graph=adv_graph, eps=(self.max_color - self.min_color) * eps),
                self.min_color, self.max_color)

            adv_noise = X_adv[i*batch_size:(i+1)*batch_size] - X[i*batch_size:(i+1)*batch_size]
            means_stds = map(lambda img: (np.mean(img), np.std(img)), adv_noise)
            noise = np.array(list(map(lambda mean_std: np.random.normal(loc=mean_std[0], scale=mean_std[1], size=(64, 64, 3)), means_stds)))
            X_noisy[i*batch_size:(i+1)*batch_size] = np.clip(noise + (X[i*batch_size:(i+1)*batch_size]), self.min_color, self.max_color)
        X = self._deprocess(X)
        X_adv = self._deprocess(X_adv)
        X_noisy = self._deprocess(X_noisy)
        return X, X_adv, X_noisy

    def print_summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print('Empty model.')

class HB_ResNet50(HB_Model):
    def __init__(self, input_size=(64, 64), weights='imagenet', adam_params=None):
        self.model = None
        self.model_input_height, self.model_input_width = 224, 224
        self.input_height, self.input_width = input_size
        self.input_channels = 3
        self.num_classes = 2
        self.weights_path = weights
        self.imagenet_mean = [103.939, 116.779, 123.68]
        self.min_color = 0. - 128.68
        self.max_color = 255. - 103.939
        if adam_params is not None:
            self.adam_params = adam_params
        else:
            self.adam_params = { 'lr': 1e-5, 'decay': 1e-6 }
        self.make_model()

    def make_model(self):
        input_tensor = keras.layers.Input(shape=(self.input_width, self.input_height, self.input_channels))
        input_resized = resize_layer.ResizeImages(output_dim=(self.model_input_height, self.model_input_width))(input_tensor)

        weights = None if self.weights_path != 'imagenet' else 'imagenet'
        base_model = keras.applications.ResNet50(input_tensor=input_resized, weights=weights, include_top=False)

        x = base_model.output
        x = keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = keras.layers.Flatten()(x)
        activations = keras.layers.Dense(self.num_classes, activation='softmax')(x) 

        self.model = keras.models.Model(inputs=base_model.input, outputs=activations)

        if self.weights_path is not None and self.weights_path is not 'imagenet':
             self.model.load_weights(self.weights_path)

        self.set_mode_pretrain()

    def set_mode_pretrain(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-4:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def set_mode_train(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[155:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def _preprocess(self, X):
        return X[..., ::-1] - self.imagenet_mean

    def _deprocess(self, X):
        return np.clip(np.round((X + self.imagenet_mean)[..., ::-1]), 0, 255).astype(np.uint8)

# End ResNet50

class HB_InceptionV3(HB_Model):
    def __init__(self, input_size=(64, 64), weights='imagenet', adam_params=None):
        self.model = None
        self.model_input_height, self.model_input_width = 224, 224
        self.input_height, self.input_width = input_size
        self.input_channels = 3
        self.num_classes = 2
        self.weights_path = weights
        self.min_color = -1
        self.max_color = 1
        if adam_params is not None:
            self.adam_params = adam_params
        else:
            self.adam_params = { 'lr': 1e-5, 'decay': 1e-6 }
        self.make_model()

    def make_model(self):
        input_tensor = keras.layers.Input(shape=(self.input_width, self.input_height, self.input_channels))
        input_resized = resize_layer.ResizeImages(output_dim=(self.model_input_height, self.model_input_width))(input_tensor)

        weights = None if self.weights_path != 'imagenet' else 'imagenet'
        base_model = keras.applications.InceptionV3(input_tensor=input_resized, weights=weights, include_top=False)

        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        activations = keras.layers.Dense(self.num_classes, activation='softmax')(x) 

        self.model = keras.models.Model(inputs=base_model.input, outputs=activations)

        if self.weights_path is not None and self.weights_path is not 'imagenet':
             self.model.load_weights(self.weights_path)

        self.set_mode_pretrain()

    def set_mode_pretrain(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-3:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def set_mode_train(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def _preprocess(self, X):
        return X.astype(np.float32) * (1. / 127.5) - 1

    def _deprocess(self, X):
        return np.clip(np.round((X + 1) * 127.5), 0, 255).astype(np.uint8)    

# End InceptionV3

class HB_VGG16(HB_Model):
    def __init__(self, input_size=(64, 64), weights='imagenet', adam_params=None):
        self.model = None
        self.model_input_height, self.model_input_width = 224, 224
        self.input_height, self.input_width = input_size
        self.input_channels = 3
        self.num_classes = 2
        self.weights_path = weights
        self.imagenet_mean = [103.939, 116.779, 123.68]
        self.min_color = 0. - 128.68
        self.max_color = 255. - 103.939
        if adam_params is not None:
            self.adam_params = adam_params
        else:
            self.adam_params = { 'lr': 1e-5, 'decay': 1e-6 }
        self.make_model()

    def make_model(self):
        input_tensor = keras.layers.Input(shape=(self.input_width, self.input_height, self.input_channels))
        input_resized = resize_layer.ResizeImages(output_dim=(self.model_input_height, self.model_input_width))(input_tensor)

        weights = None if self.weights_path != 'imagenet' else 'imagenet'
        base_model = keras.applications.VGG16(input_tensor=input_resized, weights=weights, include_top=False)

        x = base_model.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(4096, activation='relu')(x)
        x = keras.layers.Dense(4096, activation='relu')(x)
        activations = keras.layers.Dense(self.num_classes, activation='softmax')(x) 

        self.model = keras.models.Model(inputs=base_model.input, outputs=activations)

        if self.weights_path is not None and self.weights_path is not 'imagenet':
             self.model.load_weights(self.weights_path)

        self.set_mode_pretrain()

    def set_mode_pretrain(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-3:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def set_mode_train(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-8:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def _preprocess(self, X):
        return X[..., ::-1] - self.imagenet_mean

    def _deprocess(self, X):
        return np.clip(np.round((X + self.imagenet_mean)[..., ::-1]), 0, 255).astype(np.uint8)

# End VGG16

class HB_MobileNetV2(HB_Model):
    def __init__(self, input_size=(64, 64), weights='imagenet', adam_params=None):
        self.model = None
        self.model_input_height, self.model_input_width = 224, 224
        self.input_height, self.input_width = input_size
        self.input_channels = 3
        self.num_classes = 2
        self.weights_path = weights
        self.min_color = -1
        self.max_color = 1
        if adam_params is not None:
            self.adam_params = adam_params
        else:
            self.adam_params = { 'lr': 1e-5, 'decay': 1e-6 }
        self.make_model()

    def make_model(self):
        input_tensor = keras.layers.Input(shape=(self.input_width, self.input_height, self.input_channels))
        input_resized = resize_layer.ResizeImages(output_dim=(self.model_input_height, self.model_input_width))(input_tensor)

        weights = None if self.weights_path != 'imagenet' else 'imagenet'
        base_model = keras.applications.MobileNetV2(input_tensor=input_resized, weights=weights, include_top=False)

        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        activations = keras.layers.Dense(self.num_classes, activation='softmax', use_bias=True)(x) 

        self.model = keras.models.Model(inputs=base_model.input, outputs=activations)

        if self.weights_path is not None and self.weights_path is not 'imagenet':
             self.model.load_weights(self.weights_path)

        self.set_mode_pretrain()

    def set_mode_pretrain(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-2:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def set_mode_train(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-22:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def _preprocess(self, X):
        return X.astype(np.float32) * (1. / 127.5) - 1

    def _deprocess(self, X):
        return np.clip(np.round((X + 1) * 127.5), 0, 255).astype(np.uint8)

# End MobileNetV2

class HB_DenseNet121(HB_Model):
    def __init__(self, input_size=(64, 64), weights='imagenet', adam_params=None):
        self.model = None
        self.model_input_height, self.model_input_width = 224, 224
        self.input_height, self.input_width = input_size
        self.input_channels = 3
        self.num_classes = 2
        self.weights_path = weights
        self.min_color = -2.17
        self.max_color = 2.66
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        if adam_params is not None:
            self.adam_params = adam_params
        else:
            self.adam_params = { 'lr': 1e-5, 'decay': 1e-6 }
        self.make_model()

    def make_model(self):
        input_tensor = keras.layers.Input(shape=(self.input_width, self.input_height, self.input_channels))
        input_resized = resize_layer.ResizeImages(output_dim=(self.model_input_height, self.model_input_width))(input_tensor)

        weights = None if self.weights_path != 'imagenet' else 'imagenet'
        base_model = keras.applications.DenseNet121(input_tensor=input_resized, weights=weights, include_top=False)

        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        activations = keras.layers.Dense(self.num_classes, activation='softmax', use_bias=True)(x) 

        self.model = keras.models.Model(inputs=base_model.input, outputs=activations)

        if self.weights_path is not None and self.weights_path is not 'imagenet':
             self.model.load_weights(self.weights_path)

        self.set_mode_pretrain()

    def set_mode_pretrain(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-2:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def set_mode_train(self):
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-24:]:
            layer.trainable = True

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(**self.adam_params),
              metrics=['accuracy'])

    def _preprocess(self, X):
        return ((X / 255.) - self.imagenet_mean) / self.imagenet_std

    def _deprocess(self, X):
        return np.clip(np.round((((X * self.imagenet_std) + self.imagenet_mean) * 255.)), 0, 255).astype(np.uint8)
