import joblib
import pandas as pd
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

# Load the model and the scaler
model_path = "log_reg_model.joblib"
scaler_path = "scaler.joblib"
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Custom CSS to make the app visually appealing
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Streamlit interface
st.title("Categorización de calidad para lechugas en un sistema hidropónico")
st.markdown(
    """
🌱🌿 **Bienvenido al clasificador de calidad de lechugas en un sistema hidropónico**. Ingresa los parámetros relevantes para obtener una clasificación de la calidad. 🌿🌱
"""
)

# Upload image functionality
image_file = st.file_uploader(
    "📷 Sube una imagen de la lechuga", type=["jpg", "jpeg", "png"]
)

if image_file is not None:
    # Open the image file
    image = Image.open(image_file)
    st.image(image, caption="Imagen de la lechuga", use_column_width=True)

    # Save the uploaded image temporarily
    image_path = "uploaded_image.png"
    image.save(image_path)

    # Perform inference on the uploaded image
    try:
        # Step 1: Inference on the original image
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com", api_key="KCWVMT3fmYDvDWiQEqwM"
        )
        result = CLIENT.infer(image_path, model_id="diseased-healthy-lettuce/1")

        # Process the result
        x = result["predictions"][0]["x"]
        y = result["predictions"][0]["y"]
        width = result["predictions"][0]["width"]
        height = result["predictions"][0]["height"]

        # Calcular los límites de la bounding box (x_min, y_min, x_max, y_max)
        x_min = x - width / 2
        y_min = y - height / 2
        x_max = x + width / 2
        y_max = y + height / 2

        # Recortar la imagen usando la bounding box
        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        # Guardar la imagen recortada
        cropped_image_path = "crop.png"
        cropped_image.save(cropped_image_path)

        # Step 2: Inference on the cropped image
        result_cropped = CLIENT.infer(
            cropped_image_path,
            model_id="lettuce-deficiency-detection-5qags/1",
        )
        if (
            "predictions" in result_cropped
            and len(result_cropped["predictions"]) > 0
            and "class" in result_cropped["predictions"][0]
        ):
            result_inference = result_cropped["predictions"][0]["class"]
        else:
            result_inference = "Healthier"

        # Show the final result in Streamlit
        st.subheader("Resultado de la inferencia final:")
        st.text(result_inference)

    except Exception as e:
        st.error(f"Ocurrió un error durante la inferencia: {e}")

# Input fields for lettuce quality prediction
temperature = st.number_input(
    "🌡️ Temperatura (°C)", min_value=0.0, max_value=50.0, value=25.0
)
stage = st.selectbox("📊 Etapa de crecimiento", options=[1, 2, 3, 4])
ph = st.number_input("🔬 pH", min_value=0.0, max_value=14.0, value=6.5)
ppm = st.number_input(
    "💧 PPM (Partículas por millón)", min_value=0.0, max_value=2000.0, value=900.0
)
conductivity = st.number_input(
    "⚡ Conductividad (mS/cm)", min_value=0.0, max_value=10.0, value=1.5
)

# Optimal ranges
optimal_ranges = {
    "temperature": (20.0, 30.0),
    "ph": (5.3, 6.3),
    "conductivity": (1.0, 1.8),
}

# Predict button for lettuce quality
if st.button("✨ Predecir Calidad ✨"):
    input_data = pd.DataFrame(
        [[temperature, stage, ph, ppm, conductivity]],
        columns=["Temperature", "Stage", "pH", "PPM", "Conductivity"],
    )
    input_data_scaled = loaded_scaler.transform(input_data)
    prediction = loaded_model.predict(input_data_scaled)

    st.subheader("🔍 Predicción de la categoría de calidad:")

    if prediction[0] == 1:
        st.success(
            "🌟 🥬 ¡Felicidades! 🥬 🌟 \n\n Tu lechuga está en condiciones **perfectas**. ¡Sigue cuidando esos parámetros! 🌱"
        )
        st.balloons()
    else:
        st.error("❌ 😞 Lo sentimos, tu lechuga no está en las mejores condiciones.")

        # Check and print values outside the optimal range
        errors = []
        if not (
            optimal_ranges["temperature"][0]
            <= temperature
            <= optimal_ranges["temperature"][1]
        ):
            errors.append(
                f"🌡️ Temperatura: {temperature}°C (Debe estar entre {optimal_ranges['temperature'][0]}°C y {optimal_ranges['temperature'][1]}°C)"
            )
        if not (optimal_ranges["ph"][0] <= ph <= optimal_ranges["ph"][1]):
            errors.append(
                f"🔬 pH: {ph} (Debe estar entre {optimal_ranges['ph'][0]} y {optimal_ranges['ph'][1]})"
            )
        if not (
            optimal_ranges["conductivity"][0]
            <= conductivity
            <= optimal_ranges["conductivity"][1]
        ):
            errors.append(
                f"⚡ Conductividad: {conductivity} mS/cm (Debe estar entre {optimal_ranges['conductivity'][0]} y {optimal_ranges['conductivity'][1]} mS/cm)"
            )

        if errors:
            st.warning("🔧 **Parámetros fuera de rango:**")
            for error in errors:
                st.write(error)
            st.info(
                "💡 **Recomendación:** Ajusta los parámetros a los rangos óptimos para mejorar la calidad de tu lechuga."
            )

    st.markdown(
        "🍀 Si necesitas más ayuda, ajusta los parámetros para ver cómo afecta a la calidad de la lechuga."
    )
