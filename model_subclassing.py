import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def __call__(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        output_tensor = self.relu(x)

        return output_tensor


class ResBlock(layers.Layer):
    def __init__(self, channels, kernels=[3, 3, 3]):
        super(ResBlock, self).__init__()
        self.conv1 = CNNBlock(channels[0], kernel_size=kernels[0])
        self.conv2 = CNNBlock(channels[1], kernel_size=kernels[1])
        self.conv3 = CNNBlock(channels[2], kernel_size=kernels[2])
        self.pooling = layers.MaxPool2D(pool_size=2)
        self.identity_mapping = layers.Conv2D(channels[1], kernel_size=kernels[1], padding='same')

    def __call__(self, input_tensor, training=False):
        x = self.conv1(input_tensor, training=training)
        x = self.conv2(input_tensor, training=training)
        x = self.conv3(x + self.identity_mapping(x), training=training)

        return self.pooling(x)


class ResNet(keras.Model):
    def __init__(self, n_classes=10):
        super(ResNet, self).__init__()
        self.block1 = ResBlock(channels=[32, 32, 32])
        self.block2 = ResBlock(channels=[32, 32, 64])
        self.block3 = ResBlock(channels=[64, 64, 128])
        self.pool = layers.GlobalAvgPool2D()
        self.classifier = layers.Dense(n_classes)

    def __call__(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(input_tensor, training=training)
        x = self.block3(input_tensor, training=training)
        x = self.pool(x)

        return self.classifier(x)

    def model(self):
        x = keras.Input(shape=(28, 28, 1))

        return keras.Model(inputs=[x], outputs=self.__call__(x))


model = ResNet(n_classes=10)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=32, verbose=2, epochs=3)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
