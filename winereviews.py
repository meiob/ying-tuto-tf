# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow_hub as hub

from imblearn.over_sampling import RandomOverSampler

# print(df.head)
df = pd.read_csv("data/wine-reviews.csv", usecols = ['country', 'description', 'points', 'price', 'variety', 'winery'])

df = df.dropna(subset=['description', 'points'])

def plt_histgram():
    plt.hist(df.points, bins=20)
    plt.title("Point Histgram")

    plt.ylabel("N")
    plt.xlabel("POints")
    # plt.show()
    return()

print(plt_histgram)

df["label"] = (df.points >= 90).astype(int)
df = df[["description", "points", "label"]]

print(df.head())
print(df.tail())

train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])

def df_to_dataset(dataframe, shuffle=True, batch_size=1024):
    df = dataframe.copy()
    labels = df.pop('label')
    df = df["description"]
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_data = df_to_dataset(train)
valid_data = df_to_dataset(val)
test_data = df_to_dataset(test)

print(list(train_data)[0])

# Embedding + Model

# Transform text data into vector of numbers for model
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)
print(hub_layer(list(train_data)[0][0]))

# Model
# model = tf.keras.Sequential()
# model.add(hub_layer)
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy'])

# Model
input_signature = [tf.TensorSpec(shape=(None,), dtype=tf.string, name='description_input')]
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, input_shape=[], trainable=True)
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Evaluate the model
model.evaluate(train_data)
model.evaluate(valid_data)
