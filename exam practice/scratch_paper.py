import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = f'{os.getcwd()}/tmp/mnist_train.csv'
test_path = f'{os.getcwd()}/tmp/mnist_test.csv'

def get_data(filename, has_labels=True):
    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)
        images = []
        if has_labels:
            labels = []
            for row in reader:
                labels.append(np.array(row[0]).astype(float))
                images.append(np.reshape(np.array(row[1:]).astype(float), (28, 28)))
            return np.array(images), np.array(labels)
        for row in reader:
            images.append(np.reshape(np.array(row).astype(float), (28, 28)))
        return np.array(images)

train_images, train_labels = get_data(train_path, has_labels=True)
test_images = get_data(test_path, has_labels=False)

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(
    train_images,
    train_labels,
    batch_size=32
)

model = tf.keras.models.Sequential([
    # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10
)

acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='training accuracy')
plt.plot(epochs, loss, 'b', label='training loss')
plt.title('training accuracy and loss')
plt.legend()

plt.show()