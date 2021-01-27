import os
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# define the model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activaiton='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',#tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']#tf.metrics.SparseCategoricalAccuracy()]
    )
    return model

# create basic model instance
model = create_model()

model.summary()

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

# train model w/ callbacks
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]
)

# create a new untrained model instance
model = create_model()

# evaluate the untrained model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Untrained model, accuracy: {acc*100}')

# load previously trained weights from checkpoint
model.load_weights(checkpoint_path)

# re-evaluate the new model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Restored model, accuracy: {acc*100}')

# include the epoch in the filename (uses 'str.format')
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5
)

# create another new model instance
model = create_model()

# save the weights using the 'checkpoint_path' format
model.save_weights(checkpoint_path.format(epoch=0))

# train the model with the new callback
model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    verbose=1,
    epochs=50,
    callbacks=[cp_callback]
)

# grab latest checkpoint weights
latest = tf.train.latest_checkpoint(checkpoint_dir)

# create yet another model instance
model = create_model()

# load the previously saved weights
model.load_weights(latest)

# evaluate
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Restored model w/ 50 epochs, accuracy: {acc*100}')

# Second method: save the weights
model.save_weights('./checkpoints/my_checkpoint')

# create new model instance
model = create_model()

# restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Restored model w/ save_weights method, accuracy: {acc*100}')


##### save the entire model #######
# create and train new model instance
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# save the entire model as a SavedModel
model.save('scratch_paper_folder/my_model')

# load the new model
new_model = tf.keras.models.load_model('../scratch_paper_folder/my_model')

# evaluate
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print(f'Restored model w/ save entire model method, accuracy: {acc*100}')
print(new_model.predict(test_images).shape)