import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_classes = 10

# Loading data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print('Shape of training images array:', X_train.shape)
print('Shape of training labels array:', y_train.shape)