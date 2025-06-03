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
        ["Clasificación de Imágenes", "Clasificación de Texto", "Regresión de Vivienda", "Acerca de"], 
        icons=['image', 'chat-right-text', 'house', 'info-circle'],
        menu_icon="cast", 
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
        cr = st.number_input("Crimen per cápita (CRIM)", value=0.1)
        zn = st.number_input("Zonas residenciales (>25k sq.ft.) (ZN)", value=0.0)
        indus = st.number_input("Terrenos no-comerciales (%) (INDUS)", value=10.0)
        chas = st.number_input("Orilla Río Charles? (CHAS=1)", min_value=0, max_value=1, value=0)
        nox = st.number_input("NOx (ppm)", value=0.5)
        rm = st.number_input("Habitaciones promedio (RM)", value=6.0)
        age = st.number_input("Edad (%)", value=68.0)
    with col2:
        dis = st.number_input("Distancias a centros de empleo (DIS)", value=4.0)
        rad = st.number_input("Accesibilidad a autopistas (RAD)", value=4)
        tax = st.number_input("Tasa impuestos ($/10k) (TAX)", value=300)
        ptratio = st.number_input("Alumnos por maestro (PTRATIO)", value=16.0)
        black = st.number_input("1000(B-0.63)^2 (B)", value=390.0)
        lstat = st.number_input("% de población de bajo nivel (LSTAT)", value=12.0)

    if st.button("Predecir Precio"):
        features = np.array([[cr, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]])
        # Supongamos que `pred_price = model.predict(...)`
        pred_price = float(model_reg.predict(features)[0][0])  # <-- extraer valor escalar
        
        st.write(f"Precio predicho (miles $): **{pred_price:.2f}**")
elif seleccion == "Acerca de":
    st.title("Acerca de esta Aplicación")
    st.write("Esta aplicación utiliza modelos de Machine Learning para realizar tareas de clasificación y regresión.")
    st.write("Desarrollada por Sebas como parte del curso de Machine Learning.")
    st.write("Puedes encontrar el código fuente en")

