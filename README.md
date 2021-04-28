# Tensorflow tutorial

## Installation
```
pip install tensorflow
```

## Tensor basics
### Initialization

- Create constant tensor
```python
x = tf.constant(4, shape=(2, 2, 3), dtype=tf.float32)
```
- Create constant tensor from a list
```python
x = tf.constant([[2, 1, 3], [1, 1, 1]], dtype=tf.float64)
```
- Create ones tensor
```python
x = tf.ones((2, 3))
```
- Create a tensor with a specific distribution
```python
x = tf.random.normal((2, 3), mean=2, stddev=0.3)
```
- Casting
```python
x = tf.cast(x, dtype=tf.int32)
```
### Mathematics operations

- Addition
```python
z = tf.add(x, y)
z = x + y
```
- Multiplication
```python
z = tf.mul(x, y)
z = x * y
```
- Matrix multiplication
```python
z = tf.matmul(x, y)
z = x @ y
```
### Indexing
```python
z = x[::-1]
```

### Reshaping

- Reshape a matrix
```python
z = tf.reshape(x, (2, 1, 2, 1))
```
- Transpose a matrix
```python
z = tf.transpose(x)
```

## Neural network
We can build model by using some method: sequential api or functional api. I summarize the general architecture, more detail 
you can find in [keras documents](https://keras.io/about/).

```python
# Build a custom dataset
datasets = ___

# Create a new model
model = keras.Sequential(___)
# Setup some parameters like loss_fn, optimizer, metrics 
model.compile(loss=___, optimizer=___, metrics=___)

# Train
model.fit(___, ___, batch_size=___, epochs=___, verbose=___)

# Evaluation
model.evaluate()

# Prediction
model.predict()
```

## Convolution neural network
Tensorflow provides us a convolution function which is same as Pytorch.
```python
tf.keras.layers.Conv2D()
```

## Regularization
We can use regularization techniques that can prevent our model from overfitting. The [Dropout](https://keras.io/api/layers/regularization_layers/dropout/) 
or [weight regulizers](https://keras.io/api/layers/regularizers/) are the most popular and you can click on this to read more the manner to use. **Tensorflow** 
also supports us to custom our regularization function:

```python
class CustomRegularization(keras.regularizers.Regularizer):
    def __init__(self, args):
        super(CustomRegularization, self).__init__()
        self.___ = ___

    def __call__(self, x):

        return ___

    def get_config(self):

        return ___
```

## RNN, GRU, LSTM
Tensorflow supports some method to build a recurrent model:
```python
layers.SimpleRNN()

layers.GRU()

layers.LSTM()
```

If you want to apply bidirectional method, you need to add the past layer 
in the following function:
```python
layers.Bidirectional(layers.___)
```

## Model subclassing
We can use subclassing style to struct your custom model clearly.
```python
class ResNet(keras.Model):
    def __init__(self):
        ___
    
    def __call__(self):
        ___

    def model(self):
        ___
```
