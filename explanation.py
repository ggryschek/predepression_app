import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import shap
import numpy as np

### FORMATTING PARAMETERS ###
st.markdown("""
    <style>
        /* Header Styles */
        h1, .stMarkdown h1 { font-size: 2.5rem; color: royalblue; font-weight: bold; }
        h2, .stMarkdown h2 { font-size: 1.8rem; color: darkorange; font-weight: bold; }
        h3, .stMarkdown h3 { font-size: 1.4rem; color: #008080; font-weight: bold; }
        h4, .stMarkdown h4 { font-size: 1.2rem; color: dodgerblue; }

        /* Sidebar Header Styles */
        .stSidebar h1 { font-size: 2rem; color: royalblue; font-weight: bold; }
        .stSidebar h2 { font-size: 1.6rem; color: darkorange; font-weight: bold; }
        .stSidebar h3 { font-size: 1.3rem; color: #008080; font-weight: bold; }
        .stSidebar h4 { font-size: 1.1rem; color: dodgerblue; }
    </style>
""", unsafe_allow_html=True)

### DICTIONARY ###
# Language dictionary with translations
translations = {
    "English": {
        "app_title": "Depression Screening for Primary Care",
        "app_name": "DepreScan",
        "home_page":"Home",
        "depre_page":"DepreScan",
        "explain_page":"Explanations",
        "more_info_page":"More Info",
        "survey_page":"Survey",
        "sidebar_header": "Inform Patients Data",
        "page1_title": "Depression Prediction",
        "page2_title": "More Information",
        "dont_know":"Don't know",
        "No": "No",
        "Yes": "Yes",
        'Biological Sex':'Biological Sex',
        'Marital Status':'Marital Status',
        'Age':'Age',
        'Education Level':'Education Level',
        'Household Size':'Household Size',
        'Medication Use':'Medication Use',
        'Sleep Habits':'Sleep Habits',
        'Alcohol Use':'Alcohol Use',
        'Disabilities':'Disabilities',
        'Rest Time':'Rest Time',
        'gender':'Biological Sex',
        'marital_status':'Marital Status',
        'age':'Age',
        'education_level':'Education Level',
        'household_size':'Household Size',
        'medication_use':'Medication Use',
        'sleep_hours':'Sleep Hours',
        'drinking_frequency':'Alcohol Use',
        'disabilities':'Disabilities',
        'sedentarism':'Rest Time',
        'training_time':'Training Time',
        'prediction_time':"",
        'accuracy':'',
        'precision':'',
        'recall':'',
        'f1':'',
        'test_accuracy':'Accuracy',
        'test_precision': 'Precision',
        'test_recall': 'Recall',
        'test_f1': 'F1 Score',
        'test_AUC': 'AUC**',
        'average_AUC_CV':'',
        'LGBM_metrics':'LGBM Model Performance Metrics',
        'footnote_1':'*Metric above 0.8 indicates robust model performance; **AUC = Area Under the Curve',
        'threshold':'Threshold*',
        'features':'Features',
        'importance_features':'Importance Features',
        'global_feature_importance':'LGBM - SHAP Global Feature Importance',
        'explanation_title_1':'Understanding how the features might affect the prediction',
        'explanation_intro':'The graph displays the mean absolute SHAP values ​​for each feature, highlighting their overall impact on model predictions as observed in the training data. Higher SHAP values ​​indicate a stronger influence of a feature on model decisions. Features are sorted in descending order, with the most influential at the top. Unlike traditional importance scores, SHAP values ​​capture not only the correlation, but also the actual contribution of each feature to the prediction.',
        'pdd_title':'Partial Dependence Display - How each feature may affect the prediction',
        'pdd_explain':'A Partial Dependence Plot (PDP) helps explain how a single feature (or a pair of features) affects the model’s predictions while keeping all other features constant.',
        'how_interpret':'How to interpret it',
        'how_works':'How it works: ',
        'works_1':'The plot shows how the predicted outcome changes as the feature of interest varies.',
        'works_2':'It helps answer: "If we change this feature, how does it impact the model’s decision?"',
        'works_3':'The y-axis represents the model’s predicted value, while the x-axis represents different values of the feature being analyzed.',
        'why_useful':'Why It’s Useful: ',
        'useful_1':'Interpretability: Helps understand if a feature has a positive, negative, or non-linear relationship with the target.',
        'useful_2':'Fairness & Trust: Ensures the model’s behavior aligns with domain knowledge.',
        'useful_3':'Feature Engineering: Identifies potential interactions or thresholds where the feature’s impact changes.',
        'low_precision':'',
        'recall_title':'',
        'recall_explain':'',

    },
    "Español": {
        "app_name": "DepreScan",
        "app_title": "Detección de depresión en atención primaria",
        "home_page":"Início",
        "depre_page":"DepreScan",
        "explain_page":"Explicaciones",
        "more_info_page":"Más Información",
        "survey_page":"Investigación",
        "sidebar_header": "Introducir datos del paciente",
        "page1_title": "Predicción de la depresión",
        "page2_title": "Más Informatión",
        "dont_know":"No sé",
        "No": "No",
        "Yes": "Sin",
        'Biological Sex':'Sexo biológico',
        'Marital Status':'Estado civil',
        'Age':'Edad',
        'Education Level':'Nivel de Estudios',
        'Household Size':'Tamaño del hogar',
        'Medication Use':'Uso de medicamentos',
        'Sleep Habits':'Horas de sueño',
        'Alcohol Use':'Consumo de Alcohol',
        'Disabilities':'¿Tiene Discapacidad?',
        'Rest Time':'Tiempo de descanso',
        'gender':'Sexo Biológico',
        'marital_status':'Estado Civil',
        'age':'Idade',
        'education_level':'',
        'household_size':'',
        'medication_use':'',
        'sleep_hours':'',
        'drinking_frequency':'',
        'disabilities':'',
        'sedentarism':'',
        'training_time':'Training Time',
        'prediction_time':"",
        'accuracy':'',
        'precision':'',
        'recall':'',
        'f1':'',
        'test_accuracy':'Acurácia',
        'test_precision': 'Precisión',
        'test_recall': 'Recall',
        'test_f1': 'F1 Score',
        'test_AUC': 'AUC**',
        'average_AUC_CV':'',
        'LGBM_metrics':'Métricas de Performance do Modelo LGBM',
        'footnote_1':'*Una métrica por encima de 0,8 indica un rendimiento sólido del modelo; **AUC = Área bajo la curva',
        'threshold':'Limite*',
        'features':'Características',
        'importance_features':'Importância de las Características',
        'global_feature_importance':'LGBM - Importancia global SHAP de las características',
        'how_interpret':'Cómo interpretarlo',
        'explanation_title_1':'Entendiendo la importancia de las características para la predicción',
        'explanation_intro':'El gráfico muestra los valores SHAP medios absolutos para cada característica, resaltando su impacto general en las predicciones del modelo como se observa en los datos de entrenamiento. Los valores SHAP más altos indican una mayor influencia de una característica en las decisiones del modelo. Las características se clasifican en orden descendente, con las más influyentes en la parte superior. A diferencia de los puntajes de importancia tradicionales, los valores SHAP capturan no solo la correlación sino también la contribución real de cada característica a la predicción.',
        'pdd_title':'Visualización de dependencia parcial - Cómo cada característica puede afectar la predicción',
        'pdd_explain':'Un gráfico de dependencia parcial (PDP) ayuda a explicar cómo una sola característica (o un par de características) afecta las predicciones del modelo mientras mantiene todas las demás características constantes.',
        'how_works':'Cómo funciona: ',
        'works_1':'El gráfico muestra cómo cambia el resultado previsto a medida que varía la característica de interés.',
        'works_2':'Ayuda a responder la pregunta: "¿Si cambiamos esta característica, cómo afecta a la decisión del modelo?"',
        'works_3':'El eje Y representa el valor previsto del modelo, mientras que el eje X representa los diferentes valores de la característica analizada.',
        'why_useful':'Por qué es útil: ',
        'useful_1':'Interpretabilidad: Ayuda a comprender si una característica tiene una relación positiva, negativa o no lineal con el objetivo.',
        'useful_2':'Imparcialidad y confianza: Garantiza que el comportamiento del modelo se alinee con el conocimiento del dominio.',
        'useful_3':'Ingeniería de características: Identifica posibles interacciones o umbrales donde cambia el impacto de la característica.'
    },
    "Português Br": {
        "app_name": "DepreScan",
        "app_title": "Rastreio de depressão na Atenção Primária",
        "home_page":"Início",
        "depre_page":"DepreScan",
        "explain_page":"Explicações",
        "more_info_page":"Mais Informações",
        "survey_page":"Pesquisa",
        "sidebar_header": "Informe os dados do paciente",
        "page1_title": "Predição de Depressão",
        "page2_title": "Mais Informações",
        "dont_know":"Não sei",
        "No": "Não",
        "Yes": "Sim",
        'Biological Sex':'Sexo Biológico',
        'Marital Status':'Estado Civil',
        'Age':'Idade',
        'Education Level':'Nível de Estudos',
        'Household Size':'Pessoas morando na casa',
        'Medication Use':'Uso de Medicamentos',
        'Sleep Habits':'Horas de sono',
        'Alcohol Use':'Uso de Álcool',
        'Disabilities':'Possui deficiências?',
        'Rest Time':'Tempo em Repouso',
        'gender':'Sexo Biológico',
        'marital_status':'Estado Civil',
        'age':'Idade',
        'education_level':'',
        'household_size':'',
        'medication_use':'',
        'sleep_hours':'',
        'drinking_frequency':'',
        'disabilities':'',
        'sedentarism':'',
        'training_time':'Tempo de Treinamento',
        'prediction_time':'Tempo de Predição',
        'accuracy':'Acurácia (Treino)',
        'precision':'Precisão (Treino)',
        'recall':'Recall (Treino)',
        'f1':'Escore F1 (Treino)',
        'test_accuracy':'Acurácia',
        'test_precision': 'Precisão',
        'test_recall': 'Recall',
        'test_f1': 'F1 Score',
        'test_AUC': 'AUC**',
        'average_AUC_CV':'AUC (Treino)',
        'accuracy_help':'',
        'precision_help':'',
        'recall_help':'',
        'f1_help':'',
        'auc_help':'',
        'LGBM_metrics':'Métricas de Performance do Modelo LGBM',
        'footnote_1':'*Métrica acima de 0.8 indica performance robusta do modelo; **AUC = Área sob a Curva',
        'threshold':'Limite*',
        'features':'Características',
        'importance_features':'Importância das Características',
        'global_feature_importance':'LGBM - Importância Global SHAP das Variáveis',
        'explanation_title_1':'Entendendo a importância das características para a predição',
        'explanation_intro':'O gráfico exibe os valores médios absolutos de SHAP para cada característica, destacando seu impacto geral nas previsões do modelo, conforme observado nos dados de treinamento. Valores mais altos de SHAP indicam uma influência mais forte de uma característica nas decisões do modelo. As características são classificadas em ordem decrescente, com as mais influentes no topo. Ao contrário das pontuações de importância tradicionais, os valores de SHAP capturam não apenas a correlação, mas também a contribuição real de cada característica para a previsão.',
        'pdd_title':'Partial Dependence Display - Como cada característica pode afetar a previsão',
        'pdd_explain':'Um Partial Dependence Plot (PDP) ajuda a explicar como uma única característica (ou um par de característica) afeta as previsões do modelo, mantendo todas as outras características constantes.',
        'how_works':'Como funciona:',
        'works_1':'O gráfico mostra como o resultado previsto muda conforme a característica de interesse varia.',
        'works_2':'Ajuda a responder: "Se mudarmos essa característica, como isso impacta a decisão do modelo?"',
        'works_3':'O eixo y (vertical) representa o valor previsto do modelo, enquanto o eixo x (horizontal) representa valores diferentes da característica que está sendo analisado.',
        'why_useful':'Porque é Útil: ',
        'useful_1':'Interpretabilidade: Ajuda a entender se uma característica tem uma relação positiva, negativa ou não linear com a variável-alvo (depressão/não-depressão).',
        'useful_2':'Justiça e confiança: Garante que o comportamento do modelo esteja alinhado com o conhecimento do domínio.',
        'useful_3':'Engenharia de variáveis: Identifica interações ou limites potenciais onde o impacto da característica muda.'
    }
}
## LOAD FILES ##

metrics_df = pd.read_pickle("model_metrics.pkl")

feature_importance_df = pd.read_pickle("feature_importance.pkl")

with open("LGBM.pkl", "rb") as f: # Load trained model
    app_model = pickle.load(f)

### LOAD TRAINSET SHAP VALUES ###
# Load the SHAP explanation object from the file
with open('trainset_shap_values.pkl', 'rb') as f:
    grouped_shap_values = pickle.load(f)

## FUNCTIONS ##

def plot_importance_features(df):
    # Ensure the dataframe has the correct format: 'Feature' as the index and 'Importance' as the values column
    fig, ax = plt.subplots(figsize=(10, 6))  # Create the figure and axis
    
    # Make sure the values are in the correct format: 'Feature' column for names and 'Importance' for values
    ax.barh(df['Feature'], df['Importance'], align='center')  # Create horizontal bar plot
    
    # Set the labels and title
    ax.set_title('LGBM - Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()  # Tight layout to ensure everything fits
    st.pyplot(fig)  # Display the plot in Streamlit

def plot_shap_feature_importance(shap_values, language):
    """
    Plot the global feature importance (bar plot) for SHAP values in Streamlit with translated feature names.

    Parameters:
    - shap_values: shap.Explanation
        SHAP values used to compute feature importance.
    - language: str
        The selected language for feature name translation.

    Returns:
    - None
        Displays the SHAP bar plot in the Streamlit app.
    """
    
    # Compute global feature importance (mean absolute SHAP value per feature)
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    
    # Get original feature names
    original_feature_names = shap_values.feature_names

    # Replace original feature names with translated ones
    translated_feature_names = [translations[language].get(f, f) for f in original_feature_names]

    # Create DataFrame with translated feature names
    feature_importance_df = pd.DataFrame({
        "Feature": translated_feature_names,  # Use translated names
        "Importance": feature_importance
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars using the translated feature names
    ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color='royalblue')

    # Customize the plot
    ax.set_title(translations[language]['global_feature_importance'], fontsize=16)
    ax.set_xlabel(translations[language]['importance_features'], fontsize=12)
    ax.set_ylabel(translations[language]['features'], fontsize=12)
    ax.invert_yaxis()  # Ensures the most important feature is at the top

    # Show the plot in Streamlit
    st.pyplot(fig)

# Create a SHAP explainer object for a decision tree model
explainer = shap.Explainer(app_model)

## SETTINGS ##
# Access the selected language from session state
language = st.session_state.get("language", "English")  # Default to English if not set

grouped_features = {
    translations[language]['gender']: ['RIAGENDR_Male', 'RIAGENDR_Female'],
    translations[language]['marital_status']: ['DMDMARTZ_2.0', 'DMDMARTZ_3.0'],
    translations[language]['age']: ['RIDAGEYR'],
    translations[language]['education_level']: ['DMDEDUC2'],
    translations[language]['household_size']: ['DMDHHSIZ'],
    translations[language]['medication_use']: ['RXQ050'],
    translations[language]['sleep_hours']: ['SLD012'],
    translations[language]['drinking_frequency']: ['ALQ130'],
    translations[language]['disabilities']: ['FNDADI'],
    translations[language]['sedentarism']: ['PAD680']
}

## MAIN CONTENT ##
# Streamlit app title
#col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed

# Place the image in the first column
#with col1:
#    st.image("logo_app.jpg", width=100)  # Adjust width as needed

# Place the title in the second column
#with col2:
#    st.title(translations[language]["app_name"])

col3, col4, col5, col6 = st.columns([1,1,1,1])
with col3:
    st.page_link("Home.py", label=translations[language]['home_page'], icon=":material/home:")
with col4:
    st.page_link("DepressionPrediction.py", label=translations[language]['depre_page'], icon=":material/psychology:")
with col5:
    st.page_link("MoreInfo.py", label=translations[language]['more_info_page'], icon=":material/info:")
with col6:
    st.page_link("Survey.py", label=translations[language]['survey_page'], icon=":material/edit:")

st.header(translations[language]['explain_page'])
st.markdown("### " + translations[language]['explanation_title_1'])
#plot_importance_features(feature_importance_df)

plot_shap_feature_importance(grouped_shap_values, language=language)
st.write(translations[language]['explanation_intro'])

st.write('')
st.markdown("### " + translations[language]['pdd_title'])
st.write(translations[language]['pdd_explain'])
st.image("pdp_plot.png", width=800)

col7, col8 = st.columns([1,4])
with col7:
    st.markdown("#### " + translations[language]['how_works'])
with col8:
    st.info(translations[language]['works_1'])
    st.info(translations[language]['works_2'])
    st.info(translations[language]['works_3'])

col9, col10 = st.columns([1,4])
with col9:
    st.markdown("#### " + translations[language]['why_useful'])
with col10:
    st.info(translations[language]['useful_1'])
    st.info(translations[language]['useful_2'])
    st.info(translations[language]['useful_3'])