import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import shap
import numpy as np

### FORMAT PARAMETERS ###
st.markdown("""
    <style>
        /* Main content styling */
        .main {
            margin: 0; /* Remove margins around the main content */
            padding: 10px; /* Add padding around the main content */
            font-family: 'Arial', sans-serif; /* Set font */
            background-color: #f7f7f7; /* Light background color for the main content */
            color: #333; /* Dark text color */
        }

        /* Custom styling for headings */
        h1, .stMarkdown h1 {
            font-size: 2.5rem; /* Large font for the main title */
            color: #1a73e8; /* Title color */
            margin-bottom: 15px; /* Space below the title */
        }

        h2, .stMarkdown h2 {
            font-size: 2rem; /* Medium font for secondary headings */
            color: "snow"; /* Secondary headings */
            margin-bottom: 15px; /* Space below the secondary heading */
        }

        h3, .stMarkdown h3 {
            font-size: 1.5rem; /* Smaller font for third-level headings */
            color: #f39c12; /* Orange color for third-level headings */
            margin-bottom: 10px; /* Space below the third-level heading */
        }

        h4, .stMarkdown h4 {
            font-size: 1.2rem; /* Smaller font for third-level headings */
            color: royalblue; /* blue color for forth-level headings */
            margin-bottom: 10px; /* Space below the forth-level heading */
        }

        /* Customize text size and margin for the general content */
        .stText, .stMarkdown, .stButton, .stRadio {
            font-size: 16px; /* Set text size */
            margin-bottom: 3px; /* Add space below elements */
        }

        /* Style buttons */
        .stButton > button {
            background-color: snow; /* Snow button */
            color: steelblue;
            font-size: 50px; /* Button text size */
            padding: 10px 70px; /* Button padding */
            border-radius: 8px;
            border: none;
        }

        .stButton > button:hover {
            background-color: #c5c8c9; /* Darker green when hovered */
        }

        /* Adjust the layout of elements within the page */
        .stColumns {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
        }

        /* Sidebar styling */
        .stSidebar {
            background-color: #333; /* Dark background color */
            color: white; /* Light text color */
            padding: 10px; /* Padding inside the sidebar */
            font-family: 'Arial', sans-serif; /* Set font for sidebar */
        }

        /* Custom sidebar title */
        .stSidebar h1 {
            font-size: 2rem;
            color: #fff; /* White text color */
            margin-bottom: 5px;
        }

        /* Sidebar headers (for h2 and h3) */
        .stSidebar h2 {
            font-size: 1.5rem;
            color: darkorange; /* Orange color for h2 */
            margin-top: 5px;
            margin-bottom: 5px;
        }

        .stSidebar h3 {
            font-size: 1.2rem;
            color: #1a73e8; /* Blue color for h3 */
            margin-top: 5px;
            margin-bottom: 5px;
        }

        /* Adjust sidebar text elements */
        .stSidebar .stText, .stSidebar .stMarkdown {
            font-size: 16px;
            color: #ccc; /* Lighter text color */
            margin-bottom: 5px;
        }

        /* Sidebar buttons */
        .stSidebar .stButton > button {
            background-color: #4CAF50; /* Green button */
            color: white;
            font-size: 16px;
            padding: 4px 4px;
            border-radius: 4px;
            border: none;
            margin-bottom: 5px; /* Add margin below buttons */
        }

        .stSidebar .stButton > button:hover {
            background-color: #45a049; /* Darker green when hovered */
        }

        /* Reduce padding around elements */
        .stSidebar .stSelectbox, .stSidebar .stCheckbox {
            margin-top: 2px;
            margin-bottom: 2px;
        }

        /* Style links in the sidebar */
        .stSidebar a {
            color: #f39c12; /* Orange links */
            text-decoration: none;
        }

        .stSidebar a:hover {
            text-decoration: underline; /* Underline on hover */
        }

    </style>
""", unsafe_allow_html=True)

### DICTIONARY ###
# Language dictionary with translations
translations = {
    "English": {
        "app_title": "Depression Screenning for Primary Care",
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
        'global_feature_importance':'LGBM - Global Feature Importance',
        'explanation_title_1':'Understanding Model Performance Metrics',
        'explanation_intro':'When looking at a machine learning model’s performance, you may see terms like Accuracy, Precision, Recall, F1 Score, and AUC Score. These metrics help us understand how well the model is making predictions',
        'accuracy_title':'Accuracy – “How often is the model right?”',
        'accuracy_explain':'What it means: Accuracy tells us the percentage of total predictions that were correct. If the model has 90% \accuracy, it means that out of 100 cases, 90 were correctly classified',
        'how_interpret':'How to interpret it',
        'high_accuracy':'High accuracy (close to 100%) → The model is generally making correct predictions',
        'low_accuracy':'Low accuracy (closer to 50% \or lower) → The model is making a lot of mistakes',
        'precision_title':'Precision – “When the model predicts positive, how often is it correct?”',
        'precision_explain':'What it means: Precision is about avoiding false positives (wrong positive predictions). If a model predicts someone has a disease, precision tells us how often that prediction is actually correct',
        'high_precision':'',
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
        'global_feature_importance':'LGBM - Importancia global de las características',
        'how_interpret':'How to interpret it',
        'explanation_title_1':'Comprensión de las métricas de rendimiento del modelo',
        'explanation_intro':'Al observar el rendimiento de un modelo de aprendizaje automático, es posible que vea términos como exactitud, precisión, recuperación, puntuación F1 y puntuación AUC. Estas métricas nos ayudan a comprender qué tan bien realiza predicciones el modelo',
        'accuracy_title':'',
        'accuracy_explain':'',
        'high_accuracy':'',
        'low_accuracy':'',
        'precision_title':'',
        'precision_explain':'',

    },
    "Português brasileiro": {
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
        'global_feature_importance':'LGBM - Importância Global das Variáveis',
        'explanation_title_1':'Entendendo as métricas de performance dos modelos de Machine Learning',
        'explanation_intro':'Ao observar o desempenho de um modelo de machine learning, você pode ver termos como Accuracy, Precision, Recall, F1 Score e AUC Score. Essas métricas nos ajudam a entender o quão bem o modelo está fazendo previsões',
        'accuracy_title':'',
        'accuracy_explain':'',
        'high_accuracy':'',
        'low_accuracy':'',
        'precision_title':'',
        'precision_explain':'',
    }
}
## LOAD FILES ##

metrics_df = pd.read_pickle("model_metrics.pkl")

feature_importance_df = pd.read_pickle("feature_importance.pkl")

with open("best_model.pkl", "rb") as f: # Load trained model
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
    Plot the global feature importance (bar plot) for SHAP values in Streamlit.

    Parameters:
    - shap_values: shap.Explanation
        SHAP values used to compute feature importance.
    - title: str, optional
        Title of the plot (default is "Global Feature Importance").
    - figsize: tuple, optional
        Size of the plot (default is (10, 6)).

    Returns:
    - None
        Displays the SHAP bar plot in the Streamlit app.
    """
    # Calculate the global feature importance (max absolute value across all samples)
    feature_importance = shap_values.abs.max(0)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the SHAP bar plot
    shap.plots.bar(feature_importance, show=False)

    # Customize the title and axis labels
    ax.set_title(label=translations[language]['global_feature_importance'], fontsize=16)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)

    # Show the plot in Streamlit
    st.pyplot(fig)

# Create a SHAP explainer object for a decision tree model
explainer = shap.Explainer(app_model)

## SETTINGS ##
# Access the selected language from session state
language = st.session_state.get("language", "English")  # Default to English if not set

## MAIN CONTENT ##

# Streamlit app title
col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed

# Place the image in the first column
with col1:
    st.image("logo_app.jpg", width=100)  # Adjust width as needed

# Place the title in the second column
with col2:
    st.title(translations[language]["app_name"])

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
st.subheader(translations[language]['explanation_title_1'])

#plot_importance_features(feature_importance_df)

plot_shap_feature_importance(grouped_shap_values, language)

st.image("pdp_plot.png", width=800)