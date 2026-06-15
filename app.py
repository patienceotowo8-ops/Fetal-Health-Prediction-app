import streamlit as st
import pandas as pd
import joblib

rf_fetal_health_model = joblib.load("rf_fetal_health_model.pkl")

model =rf_fetal_health_model["model"]
scaler =rf_fetal_health_model["scaler"]
features =rf_fetal_health_model["features"]

selected_features = [
    'abnormal_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'histogram_mean',
    'histogram_median',
    'mean_value_of_short_term_variability',
    'accelerations',
    'histogram_mode',
    'prolongued_decelerations',
    'mean_value_of_long_term_variability',
    'baseline value',
    'fetal_movement'
]

st.set_page_config(
    page_title ="Fetal Health Prediction",
    layout="centered"
)

st.title("Fetal Health Prediction App")

st.write(
    "This app predicts the likelihood of fetal health stage using a Random Forest Model."
)

st.markdown("---"

st.subheader ("Enter fetal information")

abnormal_short_term_variability = st.number_input(
    "Abnormal Short Term Variability",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

percentage_of_time_with_abnormal_long_term_variability = st.number_input(
    "Percentage of Time with Abnormal Long Term Variability",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

histogram_mean = st.number_input(
    "Histogram Mean",
    min_value=50.0,
    max_value=200.0,
    value=130.0
)

histogram_median = st.number_input(
    "Histogram Median",
    min_value=50.0,
    max_value=200.0,
    value=130.0
)

mean_value_of_short_term_variability = st.number_input(
    "Mean Value of Short Term Variability",
    min_value=0.0,
    max_value=20.0,
    value=1.5
)

accelerations = st.number_input(
    "Accelerations",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    format="%.3f"
)

histogram_mode = st.number_input(
    "Histogram Mode",
    min_value=50.0,
    max_value=200.0,
    value=130.0
)

prolongued_decelerations = st.number_input(
    "Prolongued Decelerations",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    format="%.3f"
)

mean_value_of_long_term_variability = st.number_input(
    "Mean Value of Long Term Variability",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

baseline_value = st.number_input(
    "Baseline Value",
    min_value=50.0,
    max_value=200.0,
    value=130.0
)

fetal_movement = st.number_input(
    "Fetal Movement",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    format="%.3f"
)

input_data = pd.DataFrame([{
    'abnormal_short_term_variability':abnormal_short_term_variability,
'percentage_of_time_with_abnormal_long_term_variability':percentage_of_time_with_abnormal_long_term_variability,
    'histogram_mean':histogram_mean,
    'histogram_median':histogram_median,
'mean_value_of_short_term_variability':mean_value_of_short_term_variability,
    'accelerations':accelerations,
    'histogram_mode':histogram_mode,
    'prolongued_decelerations':prolongued_decelerations,
'mean_value_of_long_term_variability':mean_value_of_long_term_variability,
    'baseline value':baseline value,
    'fetal_movement':fetal_movement
}])

input_data_scaled = input_data.copy()

input_data_scaled[selected_features] = scaler.transform(input_data[selected_features])

input_data_scaled = input_data_scaled[features]

if st.button ("Predict fetal health"):
    try:
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)

        st.markdown("---")

        st.subheader("Prediction Result")

if prediction[0] == 1:
        st.success("Normal")

    elif prediction[0] == 2:
        st.warning("Suspect")

    else:
        st.error("Pathological")        

st.subheader("Prediction Probabilities")

st.write(
    f"Normal: {probabilities[0][0]*100:.2f}%"
)

st.write(
    f"Suspect: {probabilities[0][1]*100:.2f}%"
)

st.write(
    f"Pathological: {probabilities[0][2]*100:.2f}%"
)

except Exception as e:

st.error("An error occurred during prediction.")

st.write("Error Details:")

st.code(str(e))