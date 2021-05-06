import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


# Text processing with tfds
(train_ds, test_ds), data_info = tfds.load(
    name='imdb_reviews',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

tokenizer = tfds.deprecated.text.Tokenizer()


def build_vocabulary():
    vocab = set()
    for text, _ in train_ds:
        vocab.update(tokenizer.tokenize(text.numpy().lower()))

    return vocab


vocab = build_vocabulary()
encoder = tfds.deprecated.text.TokenTextEncoder(
    vocab_list=vocab,
    oov_token='<UNKNOWN>',
    lowercase=True,
    tokenizer=tokenizer
)


def encode_sentence(text_tensor, label):

    return encoder.encode(text_tensor.numpy()), label


def encode_map(text, label):
    encode_text, label = tf.py_function(
        encode_sentence, inp=[text, label], Tout=(tf.int64, tf.int64),
    )

    encode_text.set_shape([None])
    label.set_shape([])

    return encode_text, label


# Dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16

train_ds = train_ds.map(encode_map, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(data_info.splits['train'].num_examples)
train_ds = train_ds.padded_batch(16, padded_shapes=([None], ()))
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = test_ds.map(encode_map, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.padded_batch(16, padded_shapes=([None], ()))


# Model
model = keras.Sequential(
    [
        layers.Masking(mask_value=0),
        layers.Embedding(input_dim=len(vocab) + 2, output_dim=16),
        # BATCH_SIZE x 1000 --> BATCH_SIZE X 1000 x 16
        layers.GlobalAvgPool1D(),
        # BATCH_SIZE x 16
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(train_ds, epochs=5, verbose=1)
model.evaluate(test_ds, verbose=1)
