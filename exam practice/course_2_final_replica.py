import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

def get_data(filename):
    with open(filename) as training_file:
        images, labels = [], []
        csv_reader = csv.reader(training_file)
        next(csv_reader)
        for row in csv_reader:
            labels.append(np.array(row[0]).astype(float))
            images.append(np.reshape(np.array(row[1:785]).astype(float), (28, 28)))
    return np.array(images), np.array(labels)

sign_mnist_train_path = f'{getcwd()}/tmp/sign_mnist_train.csv'
sign_mnist_test_path = f'{getcwd()}/tmp/sign_mnist_test.csv'
train_images, train_labels = get_data(sign_mnist_train_path)
test_images, test_labels = get_data(sign_mnist_test_path)

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

training_generator = train_datagen.flow(
    train_images,
    train_labels,
    batch_size=32
)

validation_generator = validation_datagen.flow(
    test_images,
    test_labels,
    batch_size=32
)

history = model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=5,
    verbose=1
)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Accuracy on test set: {round(acc*100, 2)}%')

# plot the graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.title('training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()