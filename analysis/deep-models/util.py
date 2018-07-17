import keras

class SaveOnBest(keras.callbacks.Callback):
  def __init__(self, model, weights_path, X_test, y_test, target_names=None):
    self.model = model
    self.weights_path = weights_path
    self.target_names = target_names
    self.X_test = X_test
    self.y_test = y_test

  def on_train_begin(self, logs=None):
    self.best_val_acc = 0

  def on_epoch_end(self, epoch, logs=None):
    if logs.get('val_acc') > self.best_val_acc:
      self.best_val_acc = logs.get('val_acc')
      self.model.save_weights(self.weights_path + '.' + str(epoch+1))
      print('Saved weights. Evaluating...')
      predictions = self.model.predict(self.X_test, verbose=1)
      print(metrics.classification_report(
          np.argmax(self.y_test, axis=1),
          np.argmax(predictions, axis=1),
          target_names=self.target_names))
