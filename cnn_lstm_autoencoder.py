import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_lstm_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.TimeDistributed(layers.Dense(input_shape[-1]))(x)
    model = models.Model(inputs, x)
    model.compile(optimizer='adam', loss='mse')
    return model
