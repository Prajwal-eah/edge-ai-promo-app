import os
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Fix macOS TensorFlow issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# ------------------------------
# Load TFLite Model
# ------------------------------
interpreter = tf.lite.Interpreter(model_path="promo_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# Encoders
# ------------------------------
le_interest = LabelEncoder()
le_time = LabelEncoder()

le_interest.fit(["food", "tech", "fashion", "travel"])
le_time.fit(["morning", "evening", "night"])

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Edge AI Promotions", page_icon="🧠")

st.title("🧠 Personalized Promotion System")
st.write("⚡ Runs on Edge AI (No Internet Required)")

# Model size
model_size = os.path.getsize("promo_model.tflite") / 1024
st.write(f"📦 Model Size: {model_size:.2f} KB")

st.divider()

# ------------------------------
# User Inputs
# ------------------------------
age = st.slider("Select Age", 15, 60, 25)

interest = st.selectbox(
    "Select Interest",
    ["food", "tech", "fashion", "travel"]
)

time_of_day = st.selectbox(
    "Select Time of Day",
    ["morning", "evening", "night"]
)

st.divider()

# ------------------------------
# Prediction + Latency
# ------------------------------
if st.button("🎯 Get Personalized Promotion"):

    input_data = np.array([[
        age,
        le_interest.transform([interest])[0],
        le_time.transform([time_of_day])[0]
    ]], dtype=np.float32)

    # Measure latency
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    end_time = time.time()

    output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    latency = (end_time - start_time) * 1000

    st.subheader(f"Prediction Score: {output:.2f}")
    st.write(f"⚡ Latency: {latency:.2f} ms")

    # ------------------------------
    # Promotion Logic
    # ------------------------------
    if output > 0.3:
        st.success("✅ User likely interested!")

        if interest == "food":
            st.success("🍔 20% OFF on Food Delivery!")
        elif interest == "tech":
            st.success("🎧 15% OFF on Electronics!")
        elif interest == "fashion":
            st.success("👕 Flat 30% OFF on Clothing!")
        else:
            st.success("✈️ Travel Discounts Available!")
    else:
        st.warning("❌ No relevant promotion")

# ------------------------------
# Accuracy (Demo Calculation)
# ------------------------------
st.divider()
st.subheader("📊 Model Accuracy (Demo)")

# Dummy test dataset
X_test = np.array([
    [20, 0, 1],
    [25, 1, 2],
    [30, 2, 0],
    [22, 0, 1],
    [35, 3, 2]
], dtype=np.float32)

y_true = np.array([1, 0, 1, 1, 0])

y_pred = []

for x in X_test:
    x = np.expand_dims(x, axis=0)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
    y_pred.append(1 if pred > 0.3 else 0)

accuracy = accuracy_score(y_true, y_pred)

st.write(f"🎯 Accuracy: {accuracy * 100:.2f}%")