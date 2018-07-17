import numpy as np
import keras
import tensorflow as tf
import sklearn
import adversarial

import resize_layer
import util as util

import keras.backend as K
K.set_learning_phase(0)

class HB_Model():
    def train(self, X_train, y_train, generator=None, batch_size=32, verbose=1, epochs=1000, validation_data=None, shuffle=True, save_on_best=True, pretraining_stage=False):
        if pretraining_stage:
            self.set_mode_pretrain()
        else:
            self.set_mode_train()

        X_train = self._preprocess(X_train)
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
                           callbacks=callbacks)
        else:
            self.model.fit_generator(generator.flow(X_train, y_train, batch_size=batch_size),
                                     verbose=verbose,
                                     epochs=epochs,
                                     steps_per_epoch=len(X_train) // batch_size,
                                     validation_data=validation_data,
                                     shuffle=shuffle,
                                     callbacks=callbacks)

    def test(self, X_test, y_test, verbose=1, target_names=None):
        X_test = self._preprocess(X_test)
        predictions = self.model.predict(X_test, verbose=1)
        print(sklearn.metrics.classification_report(
            np.argmax(y_test, axis=1),
            np.argmax(predictions, axis=1),
            target_names=target_names))
        print('Accuracy:', sklearn.metrics.accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))

    def make_adversarial_and_noisy(self, X, y):
        X = self._preprocess(X)
        X_adv = np.ndarray(X.shape)
        X_noisy = np.ndarray(X.shape)
        adv_graph = adversarial.fgm_misclassify_graph(model)
        for i in range(len(X_test)):
            print('Beginning iteration', i)
            X_adv[i:i+1] = np.clip(
                adversarial.fgm_misclassify_images(
                    adversarial_model,
                    X[i:i+1],
                    y[i:i+1],
                    graph=adv_graph, eps=2.0, min_lag=6),
                -1, 1)

            adv_noise = X_adv[i:i+1] - X[i:i+1]
            means_stds = map(lambda img: (np.mean(img), np.std(img)), adv_noise)
            noise = np.array(list(map(lambda mean_std: np.random.normal(loc=mean_std[0], scale=mean_std[1], size=(64, 64, 3)), means_stds)))
            X_noisy[i:i+1] = np.clip(noise + (X[i:i+1]), -1, 1)
        X = self._deprocess(X)
        X_adv = self._deprocess(X_adv)
        X_noisy = self._deprocess(X_noisy)
        return X, X_adv, X_noisy

    def print_summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print('Empty model.')

    def _preprocess(self, X):
        return X.astype(np.float32) * (1. / 127.5) - 1

    def _deprocess(self, X):
        return np.round((X + 1) * 127.5).astype(np.uint8)

class HB_ResNet50(HB_Model):
    def __init__(self, input_size=(64, 64), weights='imagenet', adam_params=None):
        self.model = None
        self.model_input_height, self.model_input_width = 224, 224
        self.input_height, self.input_width = input_size
        self.input_channels = 3
        self.num_classes = 2
        self.weights_path = weights
        if adam_params is not None:
            self.adam_params = adam_params
        else:
            self.adam_params = { 'lr': 1e-4, 'decay': 1e-6 }
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

    
