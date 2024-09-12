import streamlit as st
import joblib
import pandas as pd

# Load the model and the scaler
model_path = r'C:\Users\Megauna\Desktop\agro\modelo\STREAMLET\log_reg_model.joblib'
scaler_path = r'C:\Users\Megauna\Desktop\agro\modelo\STREAMLET\scaler.joblib'
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Custom CSS to make the app visually appealing
st.markdown("""
    <style>
    .main {
        background-color: #f0f5f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title h1 {
        color: #2c7a7b;
        text-align: center;
        font-size: 3rem;
    }
    .stButton button {
        background-color: #2c7a7b;
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1.2rem;
    }
    .stButton button:hover {
        background-color: #285e61;
    }
    .result {
        text-align: center;
        font-size: 1.5rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit interface
st.title("Categorización de calidad para lechugas en un sistema hidropónico")
st.markdown("""
🌱🌿 **Bienvenido al clasificador de calidad de lechugas en un sistema hidropónico**. Ingresa los parámetros relevantes para obtener una clasificación de la calidad. 🌿🌱
""")

# Input fields with emoji and placeholders
temperature = st.number_input("🌡️ Temperatura (°C)", min_value=0.0, max_value=50.0, value=25.0)
stage = st.selectbox("📊 Etapa de crecimiento", options=[1, 2, 3, 4])
ph = st.number_input("🔬 pH", min_value=0.0, max_value=14.0, value=6.5)
ppm = st.number_input("💧 PPM (Partículas por millón)", min_value=0.0, max_value=2000.0, value=900.0)
conductivity = st.number_input("⚡ Conductividad (mS/cm)", min_value=0.0, max_value=10.0, value=1.5)

# Optimal ranges
optimal_ranges = {
    'temperature': (20.0, 30.0),
    'ph': (5.3, 6.3),
    'conductivity': (1.0, 1.8)
}

# Predict button
if st.button("✨ Predecir Calidad ✨"):
    input_data = pd.DataFrame([[temperature, stage, ph, ppm, conductivity]], 
                              columns=['Temperature', 'Stage', 'pH', 'PPM', 'Conductivity'])
    input_data_scaled = loaded_scaler.transform(input_data)
    prediction = loaded_model.predict(input_data_scaled)

    st.subheader("🔍 Predicción de la categoría de calidad:")

    if prediction[0] == 1:
        st.success("🌟 🥬 ¡Felicidades! 🥬 🌟 \n\n Tu lechuga está en condiciones **perfectas**. ¡Sigue cuidando esos parámetros! 🌱")
        st.balloons()
    else:
        st.error("❌ 😞 Lo sentimos, tu lechuga no está en las mejores condiciones.")
        
        # Check and print values outside the optimal range
        errors = []
        if not (optimal_ranges['temperature'][0] <= temperature <= optimal_ranges['temperature'][1]):
            errors.append(f"🌡️ Temperatura: {temperature}°C (Debe estar entre {optimal_ranges['temperature'][0]}°C y {optimal_ranges['temperature'][1]}°C)")
        if not (optimal_ranges['ph'][0] <= ph <= optimal_ranges['ph'][1]):
            errors.append(f"🔬 pH: {ph} (Debe estar entre {optimal_ranges['ph'][0]} y {optimal_ranges['ph'][1]})")
        if not (optimal_ranges['conductivity'][0] <= conductivity <= optimal_ranges['conductivity'][1]):
            errors.append(f"⚡ Conductividad: {conductivity} mS/cm (Debe estar entre {optimal_ranges['conductivity'][0]} y {optimal_ranges['conductivity'][1]} mS/cm)")

        if errors:
            st.warning("🔧 **Parámetros fuera de rango:**")
            for error in errors:
                st.write(error)
            st.info("💡 **Recomendación:** Ajusta los parámetros a los rangos óptimos para mejorar la calidad de tu lechuga.")
    
    st.markdown("🍀 Si necesitas más ayuda, ajusta los parámetros para ver cómo afecta a la calidad de la lechuga.")
