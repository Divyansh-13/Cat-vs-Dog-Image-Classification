import tensorflow as tf
import cv2
import numpy as np

# Load trained model
model = tf.keras.models.load_model(
    "models/cat_dog_classifier.keras"
)

# Read image
img = cv2.imread("dataset/test/cats/cat.26.jpg")

# Resize
img = cv2.resize(img, (128,128))

# Normalize
img = img / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)

print("Raw Prediction:", prediction)

if prediction[0][0] < 0.5:
    print("CAT")
else:
    print("DOG")