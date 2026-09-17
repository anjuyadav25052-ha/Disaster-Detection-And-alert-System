import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Disaster Detection & Alert System",
    page_icon="🚨",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/disaster_detection_model.keras"
    )

model = load_model()

# Class names
class_names = [
    "Earthquake",
    "Fire",
    "Flood",
    "Normal"
]

# -----------------------------
# Title
# -----------------------------
st.title("🚨 Disaster Detection & Alert System")

st.write(
    "Upload an image to detect whether it represents "
    "an Earthquake, Fire, Flood, or Normal condition."
)

st.divider()

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a disaster image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    image_resized = image.resize((224, 224))

    # Convert to array
    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        prediction[predicted_index] * 100
    )

    # -----------------------------
    # Result
    # -----------------------------
    st.divider()

    st.subheader("🔍 Detection Result")

    st.write(
        f"### Prediction: **{predicted_class}**"
    )

    st.write(
        f"**Confidence: {confidence:.2f}%**"
    )

    st.progress(
        min(int(confidence), 100)
    )

    # -----------------------------
    # Alert System
    # -----------------------------
    if predicted_class != "Normal" and confidence >= 70:

        st.error(
            f"🚨 DISASTER ALERT: "
            f"{predicted_class.upper()} DETECTED!"
        )

        st.warning(
            "⚠️ Immediate attention required."
        )

    elif predicted_class == "Normal":

        st.success(
            f"✅ No disaster detected."
        )

    else:

        st.warning(
            "⚠️ Low confidence prediction. "
            "Please verify the image."
        )

    # -----------------------------
    # All Class Probabilities
    # -----------------------------
    st.divider()

    st.subheader("📊 Prediction Probabilities")

    for i, class_name in enumerate(class_names):

        probability = float(
            prediction[i] * 100
        )

        st.write(
            f"{class_name}: {probability:.2f}%"
        )

        st.progress(
            min(int(probability), 100)
        )