import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


#Imports mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizes Data. Normalization is used because otherwise optimization would be harder
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

#Model
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation = 'relu'),
        layers.Dense(256, activation = 'relu'),
        layers.Dense(10),

    ]

)
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
print(model.summary())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))


#training model
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]

)
#Evaluation of Model
model.fit(x_train,y_train, batch_size=32, epochs =5, verbose=2)
model.evaluate(x_train,y_train, batch_size=32, verbose=2)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss: {}, Test Accuracy: {}".format(test_loss,test_accuracy))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)
