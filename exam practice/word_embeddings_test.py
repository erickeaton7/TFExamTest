import io
import os
import re
import shutil
import string
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# download dataset
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='..',
                                  cache_subdir='')
# fuck this stupid bullshit

print('1')

dataset_dir = os.path.join(os.path.dirname(dataset), '../aclImdb')
# os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
# os.listdir(train_dir)

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed
)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed
)

print('2')

# # look at a few movie reviews and their labels
# for text_batch, label_batch in train_ds.take(1):
#     for i in range(5):
#         print(label_batch[i].numpy(), text_batch.numpy()[i])

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# embed 1000 word vocab into 5 dimensions
embedding_layer = tf.keras.layers.Embedding(1000, 5)

# create custom standardization function to strip HTML break tags '<br />'
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')

vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence_length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

print('3')

# make a text-only dataset (no labels) and call adapt to build the vocabulary
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

embedding_dim = 16
model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name='embedding'),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback]
)

print('4')

model.summary()

# %tensorboard --logdir logs

vocab = vectorize_layer.get_vocabulary()
print(vocab[:10])
# get weights matrix of layer named 'embedding'
weights = model.get_layer('embedding').get_weights()[0]
print(weights.shape)

# save the files
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(vocab):
    if num == 0:
        continue
    vec = weights[num]
    out_m.write(word + '\n')
    out_v.write('\t'.join([str(x) for x in vec]) + '\n')
out_v.close()
out_m.close()

print('5')

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download('vecs.tsv')
    files.download('meta.tsv')

print('6')