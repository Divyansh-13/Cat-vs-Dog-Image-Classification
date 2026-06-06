import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    BatchNormalization,
    Dropout
)

import matplotlib.pyplot as plt
import cv2
import os

# ==========================
# DATASET LOADING
# ==========================

train_ds = keras.utils.image_dataset_from_directory(
    directory='dataset/train',
    labels='inferred',
    label_mode='int',
    batch_size=16,
    image_size=(128,128)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='dataset/test',
    labels='inferred',
    label_mode='int',
    batch_size=16,
    image_size=(128,128)
)

# ==========================
# NORMALIZATION
# ==========================

def process(image,label):
    image = tf.cast(image/255.0, tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# ==========================
# CNN MODEL
# ==========================

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1,activation='sigmoid'))

model.summary()

# ==========================
# COMPILE
# ==========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================
# TRAIN
# ==========================

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=validation_ds
)

# ==========================
# SAVE MODEL
# ==========================

os.makedirs("models",exist_ok=True)

model.save("models/cat_dog_classifier.keras")

print("Model Saved!")

# ==========================
# PLOT
# ==========================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")

plt.show()