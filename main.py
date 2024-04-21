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
df = pd.read_csv("diabetes.csv")

def plting():
    
    for i in range(len(df.columns[:-1])):
        label = df.columns[i]
        plt.hist(df[df['Outcome']==1][label], color='blue', label="Diabetes", alpha=0.7)
        plt.hist(df[df['Outcome']==0][label], color='red', label="No Diabetes", alpha=0.7)
        plt.title(label)
        plt.ylabel("N")
        plt.xlabel(label)
        plt.legend()
        # plt.show()
    return

print(len(df[df['Outcome']==1]), len(df[df['Outcome']==0]))

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

print(X)
print(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

print(X.shape, y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

over = RandomOverSampler()
X, y = over.fit_resample(X, y)

data = np.hstack((X, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=df.columns)

print(len(transformed_df[transformed_df['Outcome']==1]), len(transformed_df[transformed_df['Outcome']==0]))

model = tf.keras.Sequential([
    # 16 neulons per layer
    # relu: if x <== 0 --> 9, x > 0 --> x
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.evaluate(X_train, y_train)

model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_valid, y_valid))

model.evaluate(X_valid, y_valid)

model.evaluate(X_test, y_test)
