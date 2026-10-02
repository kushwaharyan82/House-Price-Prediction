import os
import gzip
import shutil
import requests
import joblib
import pandas as pd
import streamlit as st


# Model download URL
MODEL_URL = "https://github.com/kushwaharyan82/House-Price-Prediction/releases/download/v1.0/house_price_model.pkl.gz"

MODEL_GZ = "house_price_model.pkl.gz"
MODEL_FILE = "house_price_model.pkl"


# Load model
@st.cache_resource
def load_model():

    # Download compressed model if not available
    if not os.path.exists(MODEL_FILE):

        if not os.path.exists(MODEL_GZ):
            response = requests.get(MODEL_URL)
            response.raise_for_status()

            with open(MODEL_GZ, "wb") as f:
                f.write(response.content)

        # Extract .gz file
        with gzip.open(MODEL_GZ, "rb") as f_in:
            with open(MODEL_FILE, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    return joblib.load(MODEL_FILE)


model = load_model()


# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# Title
st.title("🏠 House Price Prediction")
st.write("Predict the estimated house value using a Machine Learning model.")

st.divider()


# Input section
st.subheader("🏡 Enter House Details")

col1, col2 = st.columns(2)


with col1:

    MedInc = st.number_input(
        "Median Income",
        min_value=0.0,
        value=3.0
    )

    HouseAge = st.number_input(
        "House Age",
        min_value=0.0,
        value=20.0
    )

    AveRooms = st.number_input(
        "Average Rooms",
        min_value=0.0,
        value=5.0
    )

    AveBedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.0,
        value=1.0
    )


with col2:

    Population = st.number_input(
        "Population",
        min_value=0.0,
        value=1000.0
    )

    AveOccup = st.number_input(
        "Average Occupancy",
        min_value=0.0,
        value=3.0
    )

    Latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    Longitude = st.number_input(
        "Longitude",
        value=-118.0
    )


st.divider()


# Prediction
if st.button("🔮 Predict House Price", use_container_width=True):

    input_data = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }])

    prediction = model.predict(input_data)[0]

    price = prediction * 100000

    st.success("Prediction completed successfully!")

    st.metric(
        label="🏠 Estimated House Value",
        value=f"${price:,.2f}"
    )


st.divider()


# Model information
st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)


with col1:
    st.write("**Algorithm**")
    st.write("Random Forest")


with col2:
    st.write("**R² Score**")
    st.write("0.805")


with col3:
    st.write("**RMSE**")
    st.write("0.505")