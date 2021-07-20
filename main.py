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