import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import pickle
from keras.models import load_model
import os

with st.sidebar:
    seleccion = option_menu(
        "Menú Principal", 
        ["Acerca de", "Clasificación de Imágenes", "Clasificación de Texto", "Regresión de Vivienda"], 
        icons=['info_circle', 'chat-right-text', 'house', 'image'],
        menu_icon="app", 
        default_index=0
    )

if seleccion == "Clasificación de Imágenes":
    st.title("Clasificación de Imágenes con Fashion-MNIST")
    st.write("Sube una imagen en escala de grises para clasificarla.")
 # Cargar modelo de Fashion-MNIST
    try:
        model_img = load_model("fashion_mnist_model.h5", compile=False)
    except:
        st.error("No se pudo cargar el modelo de clasificación de imágenes.")

    uploaded_file = st.file_uploader("Sube una imagen en escala de grises", type=["png","jpg","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L").resize((28,28))
        st.image(image, caption="Imagen subida", use_column_width=False)
        # Preprocesamiento correcto para modelo con entrada (784,)
        img_array = np.array(image.convert("L").resize((28, 28))) / 255.0
        img_array = img_array.reshape(1, 784).astype("float32")  # aplanado
        
        # Predicción
        pred = model_img.predict(img_array)
        idx = int(np.argmax(pred))
        prob = float(pred[0][idx])
        class_names = ["Camiseta","Pantalón","Suéter","Vestido","Abrigo","Sandalia","Camisa","Zapatilla","Bolso","Botín"]
        st.write(f"**Predicción:** {class_names[idx]}  ({prob*100:.1f}%)")
elif seleccion == "Clasificación de Texto":
    st.title("Clasificación de Texto con IMDB")
    st.write("Ingresa una reseña de película para analizar su sentimiento (positivo o negativo).")
    with open('tokenizer_imdb.pkl','rb') as f:
        tokenizer = pickle.load(f)
    model_text = load_model("imdb_model.h5", compile=False)

    from keras.preprocessing.sequence import pad_sequences

    review = st.text_area("Ingresa la reseña de la película", "")
    if st.button("Analizar sentimiento"):
        if review:
            seq = tokenizer.texts_to_sequences([review])
            seq = pad_sequences(seq, maxlen=200)
            pred = model_text.predict(seq)[0][0]
            label = "Positiva" if pred > 0.5 else "Negativa"
            st.write(f"**Sentimiento:** {label} (prob={pred:.2f})")
        else:
            st.error("Por favor escribe una reseña.")
elif seleccion == "Regresión de Vivienda":
    st.title("Regresión de Vivienda con Boston Housing")
    st.write("Ingresa las características de una vivienda para predecir su precio en miles de dólares.")
    try:
        model_reg = load_model("boston_housing_model.h5", compile=False)
    except:
        st.error("No se pudo cargar el modelo de regresión.")

    col1, col2 = st.columns(2)
    with col1:
        cr = st.number_input("Crimen per cápita (CRIM)", value=0.28)
        zn = st.number_input("Zonas residenciales (>25k sq.ft.) (ZN)", value=0)
        indus = st.number_input("Terrenos no-comerciales (%) (INDUS)", value=9.69)
        chas = st.number_input("Orilla Río Charles? (CHAS=1)", min_value=0, max_value=1, value=0)
        nox = st.number_input("Concentración de óxidos nítricos (ppm)", value=0.585)
        rm = st.number_input("Habitaciones promedio (RM)", value=5.926)
        age = st.number_input("Unidades habitadas por propetarios counstruidas pre-1940 (%)", value=46.2)
    with col2:
        dis = st.number_input("Distancia ponderada a centros de empleo (DIS)", value=2.3817)
        rad = st.number_input("Accesibilidad a autopistas (RAD)", value=6)
        tax = st.number_input("Tasa impuestos ($/10k) (TAX)", value=391)
        ptratio = st.number_input("Alumnos por maestro en poblado (PTRATIO)", value=19.2)
        black = st.number_input("1000(Bk-0.63)^2 (B)", value=396.9)
        lstat = st.number_input("% de población de bajo nivel (LSTAT)", value=13.59)

    if st.button("Predecir Precio"):
        features = np.array([[cr, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]])
        # Supongamos que `pred_price = model.predict(...)`
        pred_price = float(model_reg.predict(features)[0][0])  # <-- extraer valor escalar
        
        st.write(f"Precio predicho (miles $): **{pred_price:.2f}**")
elif seleccion == "Acerca de":
    st.title("Acerca de esta Aplicación")
        
    st.markdown("""
    Esta aplicación fue desarrollada como parte de la materia **Modeling Learning with Artificial Intelligence**.
    
    Incluye tres modelos de aprendizaje profundo entrenados con datasets clásicos de Keras:
    
    - 🧥 **Fashion MNIST**: Clasificación de imágenes de prendas.
    - 🎬 **IMDB**: Clasificación de sentimiento en reseñas de películas.
    - 🏠 **Boston Housing**: Predicción de precios de viviendas.
    
    El código fuente de la Streamlit App completo está disponible en GitHub:
    
    🔗 [Ver repositorio en GitHub](https://github.com/A01286211/My-First-Streamlit-App)

    El código del cual se obtuvieron los modelos entrenados se encuentra en Colab:

    🔗 [Ver Colab Notebook](https://colab.research.google.com/drive/1NvpuKVffAYgi2TfAmWPB-FMpXNa2rVbA?usp=sharing)
    """)
    
