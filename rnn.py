import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


def rnn_model():
    inputs = layers.Input(shape=(None, 28))
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, activation='relu'))(inputs)
    x = layers.LSTM(512, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)

    return model


# Model
model = rnn_model()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5, verbose=2, batch_size=32)

# Evaluate
model.evaluate(x_test, y_test, verbose=2, batch_size=32)
