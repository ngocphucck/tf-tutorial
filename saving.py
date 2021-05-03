import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# Load a dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float') / 255
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.astype('float') / 255
x_test = x_test.reshape(-1, 28 * 28)


def get_model():
    input = keras.Input(shape=(28 * 28,))
    x = layers.Dense(units=64)(input)
    x = layers.ReLU()(x)
    output = layers.Dense(units=10)(x)

    model = keras.Model(inputs=[input], outputs=output)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model


class Classification(keras.Model):
    def __init__(self, n_classes=10):
        super(Classification, self).__init__()
        self.n_classes = n_classes
        self.hidden = layers.Dense(units=64)
        self.classifier = layers.Dense(units=self.n_classes)
        self.relu = layers.ReLU()
        self.flatten = layers.Flatten()

    def __call__(self, input_tensor, training=False):
        x = self.flatten(input_tensor)
        x = self.relu(self.hidden(x))
        output = self.classifier(x)

        return output


model = get_model()

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)
model.evaluate(x_test, y_test, batch_size=32, verbose=1)
model.save('completely_saved_model')
