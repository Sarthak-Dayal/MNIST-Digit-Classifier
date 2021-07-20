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
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build the Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_shape=(784,)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Activation('softmax')
])

# Compile and Train Model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=4, verbose=1, validation_data=(X_test, y_test))

# Test Model
score = model.evaluate(X_test, y_test, verbose=0)
print("TEST SCORE:", score[0])
print("TEST ACCURACY:", score[1])

# Visualize Predictions
predictions = np.argmax(model.predict(X_test), axis=-1)
y_test = np.argmax(y_test, axis=-1)
correct_indices = np.nonzero(predictions == y_test)[0]
incorrect_indices = np.nonzero(predictions != y_test)[0]

plt.figure()
for _, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap="gray", interpolation="none")
    plt.title("Predicted {}, Class {}".format(predictions[correct], y_test[correct]))
plt.tight_layout()
plt.show()

plt.figure()
for _, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap="gray", interpolation="none")
    plt.title("Predicted {}, Class {}".format(predictions[incorrect], y_test[incorrect]))
plt.tight_layout()
plt.show()