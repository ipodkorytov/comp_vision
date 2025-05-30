from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(features_train.shape[0], 784).astype('float32') / 255.0
    return features_train, target_train

def create_model(input_shape=(784,)):  # Изменено на плоский формат
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def train_model(model, train_data, test_data, batch_size=128, epochs=15,
               steps_per_epoch=None, validation_steps=None):
    features_train, target_train = train_data
    features_test, target_test = test_data
    features_test = features_test.reshape(features_test.shape[0], 784).astype('float32') / 255.0
    
    history = model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, 
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=1,
              shuffle=True)
    return model