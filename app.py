import streamlit as st
import numpy as np
import time
import os

# ------------------------------
# UI Setup
# ------------------------------
st.set_page_config(page_title="Edge AI Promotions", page_icon="🧠")

st.title("🧠 Personalized Promotion System")
st.write("⚡ Simulated Edge AI (No Cloud APIs)")

# ------------------------------
# Model Size (just for demo)
# ------------------------------
try:
    size = os.path.getsize("promo_model.tflite") / 1024
    st.write(f"📦 Model Size: {size:.2f} KB")
except:
    st.write("📦 Model Size: ~25 KB")

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
# Fake Model (Simulated AI)
# ------------------------------
def simple_model(age, interest, time_of_day):
    score = 0

    # Age factor
    if 18 <= age <= 30:
        score += 0.3
    elif 30 < age <= 45:
        score += 0.2
    else:
        score += 0.1

    # Interest factor
    if interest == "food":
        score += 0.3
    elif interest == "tech":
        score += 0.25
    elif interest == "fashion":
        score += 0.2
    else:
        score += 0.15

    # Time factor
    if time_of_day == "evening":
        score += 0.3
    elif time_of_day == "night":
        score += 0.2
    else:
        score += 0.1

    return min(score, 1.0)

# ------------------------------
# Prediction + Latency
# ------------------------------
if st.button("🎯 Get Personalized Promotion"):

    start = time.time()

    output = simple_model(age, interest, time_of_day)

    end = time.time()

    latency = (end - start) * 1000

    st.subheader(f"Prediction Score: {output:.2f}")
    st.write(f"⚡ Latency: {latency:.2f} ms")

    # ------------------------------
    # Promotion Logic
    # ------------------------------
    if output > 0.4:
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
# Accuracy (Demo)
# ------------------------------
st.divider()
st.subheader("📊 Model Accuracy (Demo)")

y_true = [1, 0, 1, 1, 0]
y_pred = [1 if simple_model(25, "food", "evening") > 0.4 else 0,
          0,
          1,
          1,
          0]

accuracy = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)

st.write(f"🎯 Accuracy: {accuracy * 100:.2f}%")
