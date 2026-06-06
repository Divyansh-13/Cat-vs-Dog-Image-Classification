import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_paste_button import paste_image_button

# Load model
model = tf.keras.models.load_model(
    "models/cat_dog_classifier.keras"
)

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

st.title("🐱 Cat vs Dog Classifier 🐶")

st.write(
    "Upload an image or paste one directly from your clipboard."
)

# -------------------------------
# Upload Option
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Paste Option
# -------------------------------

paste_result = paste_image_button(
    "📋 Paste Image (Ctrl + V)"
)

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif paste_result.image_data is not None:
    image = paste_result.image_data.convert("RGB")

# -------------------------------
# Prediction
# -------------------------------

if image is not None:

    st.image(
        image,
        caption="Selected Image",
        width=350
    )

    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )
    img = np.array(image)

    img = cv2.resize(img, (128, 128))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    dog_prob = float(prediction[0][0])
    cat_prob = 1 - dog_prob

    if dog_prob > 0.5:

        st.success(
            f"🐶 Dog ({dog_prob*100:.2f}%)"
        )

    else:

        st.success(
            f"🐱 Cat ({cat_prob*100:.2f}%)"
        )

    st.write("### Confidence Scores")

    st.progress(cat_prob)
    st.write(f"Cat: {cat_prob*100:.2f}%")

    st.progress(dog_prob)
    st.write(f"Dog: {dog_prob*100:.2f}%")