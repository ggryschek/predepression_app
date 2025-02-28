import streamlit as st
import pandas as pd
import shap
import lime.lime_tabular
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Language dictionary with translations
translations = {
    "English": {
        "app_title": "Depression Prediction for Primary Care",
        "sidebar_header": "Enter Feature Values",
        "age": "Age",
        "gender": "Gender",
        "Male":"Male",
        "Female":"Female",
        "marital_status": "Marital Status",
        "household_size": "Household Size",
        "health_rating": "Health Rating (1=Excellent, 5=Poor)",
        "sleep_hours": "Sleep Hours",
        "drinking_frequency": "Drinking Frequency",
        "disabilities": "Any Disabilities?",
        "predict_button": "Predict",
        "prediction_result": "Prediction Result",
        "predicted_class": "Predicted Class:",
        "prediction_probabilities": "Prediction Probabilities:",
        "class_probabilities_title": "Class Probabilities",
        "non_depression": "Non-Depression",
        "depression": "Depression"
    },
    "Español": {
        "app_title": "Aplicación de Predicción ML con Preprocesamiento",
        "sidebar_header": "Ingrese los Valores de las Características",
        "age": "Edad",
        "gender": "Género",
        "Male":"Masculino",
        "Female":"Feminino",
        "marital_status": "Estado Civil",
        "household_size": "Tamaño del Hogar",
        "health_rating": "Clasificación de Salud (1=Excelente, 5=Pobre)",
        "sleep_hours": "Horas de Sueño",
        "drinking_frequency": "Frecuencia de Consumo de Alcohol",
        "disabilities": "¿Tiene Discapacidad?",
        "predict_button": "Predecir",
        "prediction_result": "Resultado de la Predicción",
        "predicted_class": "Clase Predicha:",
        "prediction_probabilities": "Probabilidades de Predicción:",
        "class_probabilities_title": "Probabilidades de Clase",
        "non_depression": "No Depresión",
        "depression": "Depresión"
    },
    "Português brasileiro": {
        "app_title": "Predição de Depressão na Atenção Primária",
        "sidebar_header": "Insira as Características do Paciente",
        "age": "Idade",
        "gender": "Gênero",
        "Male":"Masculino",
        "Female":"Feminino",
        "marital_status": "Estado Civil",
        "household_size": "Tamanho da Família",
        "health_rating": "Estado de Saúde Autoreferido (1=Excelente, 2-Muito bom, 3-Bom, 4-Regular, 5=Péssimo)",
        "sleep_hours": "Horas de Sono",
        "drinking_frequency": "Frequência de Consumo de Álcool",
        "disabilities": "Possui Deficiência?",
        "predict_button": "Prever",
        "prediction_result": "Resultado da Previsão",
        "predicted_class": "Classe Prevista:",
        "prediction_probabilities": "Probabilidades da Previsão:",
        "class_probabilities_title": "Probabilidades das Classes",
        "non_depression": "Não Depressão",
        "depression": "Depressão"
    }
}

# Function to plot class probabilities
def plot_class_probabilities(probabilities, language):
    """
    Given the class probabilities, this function plots a bar chart.
    
    Parameters:
    - probabilities: The probabilities for each class (array-like).
    - language: The selected language for translations.
    """
    # Get translations for labels
    class_labels = [
        translations[language]["non_depression"],
        translations[language]["depression"]
    ]

    # Create a bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(class_labels, probabilities, color=['blue', 'green'])

    # Add title and labels
    plt.title(translations[language]["class_probabilities_title"])
    plt.xlabel('Classes')
    plt.ylabel('Probability')

    # Show the chart in Streamlit
    st.pyplot(plt)

# Load trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess_input(data):
    """
    Transforms raw user input into the required format for model prediction.
    """
    data['RIAGENDR_2.0'] = 1 if data['gender'] == 'Female' else 0
    data['DMDMARTZ_2.0'] = 1 if data['marital_status'] == 'Divorced' else 0
    data['DMDMARTZ_3.0'] = 1 if data['marital_status'] == 'Never Married' else 0
    data['FNDADI'] = 1 if data['disabilities'] == 'Yes' else 0

    # Selecting only relevant columns
    input_features = ['RIDAGEYR', 'RIAGENDR_2.0', 'DMDMARTZ_2.0', 'DMDMARTZ_3.0', 
                      'DMDHHSIZ', 'HUQ010', 'SLD012', 'ALQ130', 'FNDADI']

    # Convert to DataFrame
    df = pd.DataFrame([data], columns=input_features)

    return df

# Set up layout with two columns
col1, col2 = st.columns([8, 1])  # Left side larger, right side for flags

# Language selectbox
with col1:
    language = st.selectbox(
        'Choose your language / Elige tu idioma / Escolha seu idioma',
        ['English', 'Español', 'Português brasileiro'],
        index=0  # Default to English
    )

# Display flag image beside the language selection
with col2:
    if language == 'English':
        img = mpimg.imread('us_flag.png')  # Replace with your image file
        st.image(img, use_container_width=True)

    elif language == 'Español':
        img = mpimg.imread('es_flag.png')  # Replace with your image file
        st.image(img, use_container_width=True)

    elif language == 'Português brasileiro':
        img = mpimg.imread('br_flag.png')  # Replace with your image file
        st.image(img, use_container_width=True)

# Streamlit app title
st.title(translations[language]["app_title"])

# User input section
st.sidebar.header(translations[language]["sidebar_header"])

# Collect user input
user_input = {
    'RIDAGEYR': st.sidebar.number_input(translations[language]["age"], min_value=0, max_value=100, value=50),
    'gender': st.sidebar.radio(translations[language]["gender"], ["Male", "Female"]),
    'marital_status': st.sidebar.selectbox(translations[language]["marital_status"], ["Married", "Divorced", "Never Married"]),
    'DMDHHSIZ': st.sidebar.number_input(translations[language]["household_size"], min_value=1, max_value=10, value=3),
    'HUQ010': st.sidebar.slider(translations[language]["health_rating"], 1, 5, 3),
    'SLD012': st.sidebar.number_input(translations[language]["sleep_hours"], min_value=0, max_value=12, value=7),
    'ALQ130': st.sidebar.number_input(translations[language]["drinking_frequency"], min_value=0, max_value=30, value=5),
    'disabilities': st.sidebar.radio(translations[language]["disabilities"], ["No", "Yes"])
}

# Convert user input into a DataFrame
input_df = preprocess_input(user_input)

# Predict
if st.button(translations[language]["predict_button"]):
    # Predict the class and probabilities
    prediction = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0]

    st.subheader(translations[language]["prediction_result"])
    st.write(f"{translations[language]['predicted_class']} **{prediction}**")
    st.write(f"{translations[language]['prediction_probabilities']} {probas}")

    # Plot the class probabilities as a bar chart
    plot_class_probabilities(probas, language)