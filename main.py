import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_classes = 10

# Loading data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print('Shape of training images array:', X_train.shape)
print('Shape of training labels array:', y_train.shape)

# Visualizing data
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap="gray", interpolation="none")
    plt.title("Class {}".format(y_train[i]))

plt.tight_layout()
plt.show()

## Prepare data for training
# Flatten into single array instead of 2D 28x28 array
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)

# Make sure all data is in float format for arithmetic
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Normalize data to 0->1 range
X_train /= 255
X_test /= 255

# One-hot encode the labels
y_train = tf.keras.utils_to_categorical(y_train, num_classes)
y_test = tf.keras.utils_to_categorical(y_test, num_classes)