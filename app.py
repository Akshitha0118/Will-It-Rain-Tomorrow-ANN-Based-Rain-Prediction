import streamlit as st
import numpy as np
import pickle

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="Rain Prediction App",
    page_icon="🌧️",
    layout="centered"
)

# ======================
# Custom CSS
# ======================
st.markdown("""
<style>
.main {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 10px;
}
h1 {
    color: #1f77b4;
    text-align: center;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    padding: 10px 22px;
    font-size: 16px;
}
.result {
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 8px;
}
.rain {
    background-color: #d4edda;
    color: #155724;
}
.no-rain {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# ======================
# Load Model & Scaler
# ======================
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ======================
# Title
# ======================
st.markdown("<h1>🌧️ Will It Rain Tomorrow?</h1>", unsafe_allow_html=True)
st.write("ANN-based Rain Prediction using Machine Learning")

st.divider()

# ======================
# Inputs
# ======================
st.subheader("🌤️ Weather Inputs")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 150.0, 15.0)

with col2:
    pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
    cloud = st.number_input("Cloud Cover (%)", 0.0, 100.0, 60.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 0.0)

# ======================
# Prediction
# ======================
if st.button("🔍 Predict"):
    input_data = np.array([[temperature, humidity, wind_speed,
                            pressure, cloud, rainfall]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.markdown("<div class='result rain'>🌧️ Rain Expected Tomorrow</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='result no-rain'>☀️ No Rain Expected Tomorrow</div>",
                    unsafe_allow_html=True)

# ======================
# Footer
# ======================
st.divider()
st.caption("Built with ❤️ using Streamlit & Scikit-learn ANN")
