import tensorflow as tf
import os
import zipfile

zip_ref = zipfile.ZipFile(f'{os.getcwd()}/tmp/happy-or-sad.zip', 'r')
zip_ref.extractall('tmp/h-or-s')
zip_ref.close()

def train_happy_sad_model():
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > DESIRED_ACCURACY:
                print('\nReached 99.9% accuracy so cancelling training!')
                self.model.stop_training = True

    callbacks = myCallback()

    # define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        f'{os.getcwd()}/tmp/h-or-s',
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary'
    )

    history = model.fit_generator(
        train_generator,
        epochs=15,
        steps_per_epoch=8,
        verbose=1,
        callbacks=[callbacks]
    )

    return history.history['accuracy'][-1]

train_happy_sad_model()