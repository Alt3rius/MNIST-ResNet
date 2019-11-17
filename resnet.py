import tensorflow as tf
import pandas as pd
import numpy as np
#import IPython
#IPython.embed()



(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train/255
X_test = X_test/255

y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

X_train = np.reshape(X_train, (-1, 28, 28, 1))
X_test = np.reshape(X_test, (-1, 28, 28, 1))


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=units, kernel_size=(3, 3),  padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=units, kernel_size=(3, 3),  padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=units, kernel_size=(3, 3),  padding="same", activation="relu")
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        output = inputs + x
        return output

class ResNet(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.residual_block1 = ResidualBlock(10)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid")
        self.residual_block2 = ResidualBlock(10)
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid")
        self.residual_block3 = ResidualBlock(10)
        self.max_pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid")
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(units=10, activation="softmax")

    def call(self, inputs):
        x = self.residual_block1(inputs)
        x = self.max_pool1(x)
        x = self.residual_block2(x)
        x = self.max_pool2(x)
        x = self.residual_block3(x)
        x = self.max_pool3(x)
        x = self.gap(x)
        output = self.dense(x)
        return output


resnet = ResNet()

dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(512)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy()
accuracy_object = tf.keras.metrics.CategoricalAccuracy()

for i in range(25):
    losses = []
    accuracies = []
    for X, y in dataset_train:
        with tf.GradientTape() as tape:
            out = resnet(X)
            loss = loss_object(y, out)
        losses.append(loss)
        accuracies.append(accuracy_object(y, out))
        gradients = tape.gradient(loss, resnet.trainable_variables)
        xyz = optimizer.apply_gradients(zip(gradients, resnet.trainable_variables))
    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(accuracies)
    print("Epoch: " + str(i+1) + " Loss: " + str(mean_loss) + " Accuracy: " + str(mean_accuracy))

y_pred = resnet.predict(X_test)
accuracy = accuracy_object(y_test, y_pred)
print(accuracy.numpy())
