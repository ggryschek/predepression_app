import streamlit as st

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
        "link_1":'Check the dataset source for training this model in NATIONAL CENTER FOR HEALTH STATISTICS/Center for Control and Prevention of Diseases (CDC): ',
        'link_2':'National Health and Nutrition Examination Survey (NHANES 21-23)',
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
        'auc':'AUC',
        'auc_table_explain': 'Varies by calculation method',
        'LGBM_metrics':'LGBM Model Performance Metrics',
        'footnote_1':'*Metric above 0.8 indicates robust model performance; **AUC = Area Under the Curve',
        'threshold':'Threshold*',
        'global_feature_importance':'LGBM - Global Feature Importance',
        'term': 'Term',
        'meaning': 'Meaning',
        'math_representation': 'Mathematical Representation',
        'confusion_matrix_terminology':'Confusion Matrix Terminology',
        'tp': 'True Positive (TP)',
        'tn': 'True Negative (TN)',
        'fp': 'False Positive (FP)',
        'fn': 'False Negative (FN)',
        'tp_desc': 'Model predicted **positive**, and the actual value was **positive**',
        'tn_desc': 'Model predicted **negative**, and the actual value was **negative**',
        'fp_desc': 'Model predicted **positive**, but the actual value was **negative** (**Type I Error**)',
        'fn_desc': 'Model predicted **negative**, but the actual value was **positive** (**Type II Error**)',
        'explanation_title_1':'Understanding Model Performance Metrics',
        'explanation_intro':'When looking at a machine learning model’s performance, you may see terms like Accuracy, Precision, Recall, F1 Score, and AUC Score. These metrics help us understand how well the model is making predictions',
        'accuracy_title':'Accuracy – “How often is the model right?”',
        'accuracy_explain':'What it means: Accuracy tells us the percentage of total predictions that were correct. If the model has 90% accuracy, it means that out of 100 cases, 90 were correctly classified',
        'how_interpret':'How to interpret it',
        'metrics_summary':'Metrics Summary',
        'high_accuracy':'High accuracy (close to 100%) → The model is generally making correct predictions',
        'low_accuracy':'Low accuracy (closer to 50% or lower) → The model is making a lot of mistakes',
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
        "link_1":'mmm',
        'link_2':'kkk',
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
        'auc':'AUC',
        'auc_table_explain': 'Varies by calculation method',
        'LGBM_metrics':'Métricas de Performance do Modelo LGBM',
        'footnote_1':'*Una métrica por encima de 0,8 indica un rendimiento sólido del modelo; **AUC = Área bajo la curva',
        'threshold':'Limite*',
        'global_feature_importance':'LGBM - Importancia global de las características',
        'term': 'Term',
        'meaning': 'Meaning',
        'math_representation': 'Mathematical Representation',
        'confusion_matrix_terminology':'Confusion Matrix Terminology',
        'tp': 'True Positive (TP)',
        'tn': 'True Negative (TN)',
        'fp': 'False Positive (FP)',
        'fn': 'False Negative (FN)',
        'tp_desc': 'Model predicted **positive**, and the actual value was **positive**',
        'tn_desc': 'Model predicted **negative**, and the actual value was **negative**',
        'fp_desc': 'Model predicted **positive**, but the actual value was **negative** (**Type I Error**)',
        'fn_desc': 'Model predicted **negative**, but the actual value was **positive** (**Type II Error**)',
        'how_interpret':'How to interpret it',
        'metrics_summary':'Metrics Summary',
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
        "link_1":'mmm',
        'link_2':'kkk',
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
        'auc':'AUC',
        'average_AUC_CV':'AUC (Treino)',
        'auc_table_explain': 'Varies by calculation method',
        'accuracy_help':'',
        'precision_help':'',
        'recall_help':'',
        'f1_help':'',
        'auc_help':'',
        'LGBM_metrics':'Métricas de Performance do Modelo LGBM',
        'footnote_1':'*Métrica acima de 0.8 indica performance robusta do modelo; **AUC = Área sob a Curva',
        'threshold':'Limite*',
        'global_feature_importance':'LGBM - Importância Global das Variáveis',
        'how_interpret':'Como Interpretar',
        'term': 'Termo',
        'meaning': 'Significado',
        'math_representation': 'Representação Matemática',
        'confusion_matrix_terminology':'Terminologia da Matriz de Confusão',
        'tp': 'True Positive (TP)',
        'tn': 'True Negative (TN)',
        'fp': 'False Positive (FP)',
        'fn': 'False Negative (FN)',
        'tp_desc': 'Model predicted **positive**, and the actual value was **positive**',
        'tn_desc': 'Model predicted **negative**, and the actual value was **negative**',
        'fp_desc': 'Model predicted **positive**, but the actual value was **negative** (**Type I Error**)',
        'fn_desc': 'Model predicted **negative**, but the actual value was **positive** (**Type II Error**)',
        'metrics_summary':'Resumo das Medidas de Performance',
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

with open('DT_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

### LOAD TRAINSET SHAP VALUES ###
# Load the SHAP explanation object from the file
with open('trainset_shap_values.pkl', 'rb') as f:
    grouped_shap_values = pickle.load(f)

## FUNCTIONS ##

def plot_model_results(df, language):
    # Define the classifiers' names (assuming 'Model' is the index)
    classifiers = df['Model'].tolist()

    # Define the metrics to plot
    #timing_cols = ['training_time', 'prediction_time']
    test_metrics_cols = ['test_accuracy','test_precision', 'test_recall', 'test_f1', 'test_AUC']

    df_plot = df[test_metrics_cols].copy()

    # Define new column names
    new_column_names = {
        'training_time':translations[language]['training_time'],
        'prediction_time':translations[language]['prediction_time'],
        'accuracy':translations[language]['accuracy'],
        'precision':translations[language]['precision'],
        'recall':translations[language]['recall'],
        'f1':translations[language]['f1'],
        'test_accuracy': translations[language]['test_accuracy'],
        'test_precision': translations[language]['test_precision'],
        'test_recall': translations[language]['test_recall'],
        'test_f1': translations[language]['test_f1'],
        'test_AUC': translations[language]['test_AUC'],
        'average_AUC_CV':translations[language]['average_AUC_CV']
    }

    # Rename columns
    df_plot = df_plot.rename(columns=new_column_names)

    # Use a color theme
    plt.style.use("default")

    # Plot using new column names
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.get_cmap("Set2").colors
    #colors = ['purple','red', 'green', 'blue', 'orange']  
    df_plot.plot(kind='bar', ax=ax, width=0.8, color=colors)

    threshold = 0.8
    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=translations[language]['threshold'])
    ax.legend()

    # Display the values on top of the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    xytext=(0, 9),  # Vertical offset
                    textcoords='offset points', 
                    ha='center', 
                    va='bottom', 
                    fontsize=12, color='black')

    # Title and labels
    ax.set_title(translations[language]['LGBM_metrics'], fontsize=24)
    #ax.set_ylabel('Score', fontsize=12)
    ax.set_xticklabels('', rotation=45)
    #ax.set_xlabel('LGBM')

    # Add footnote
    plt.figtext(0.1, -0.02, translations[language]['footnote_1'], fontsize=12, ha="left", color="gray")

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

from sklearn.tree import plot_tree

def plot_custom_decision_tree(model, feature_names, class_names, figsize=(12, 8), title="Decision Tree", fontsize=12):
    """
    Function to plot a customized decision tree.

    Parameters:
    - model: Trained decision tree model (e.g., DecisionTreeClassifier).
    - feature_names: List or DataFrame columns containing the feature names.
    - class_names: List containing the class names for classification (e.g., ['Non Depression', 'Depression']).
    - figsize: Tuple specifying the figure size (default is (12, 8)).
    - title: Title for the plot (default is 'Decision Tree').
    - fontsize: Font size for labels and title (default is 12).
    """
    plt.figure(figsize=figsize)  # Adjust figure size
    plot_tree(
        model,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=class_names,
        fontsize=fontsize,
        precision=2,
        label='all',
        proportion=True,
        node_ids=True,
        impurity=False
        #color_map='Blues'
    )

    # Add a title to the plot
    plt.title(title, fontsize=16)

    # Show the plot
    plt.show()

# Create a SHAP explainer object from the model
explainer = shap.Explainer(app_model)

features_names = ['RIDAGEYR', 'RIAGENDR_Female', 'RIAGENDR_Male', 'DMDMARTZ_2.0',
       'DMDMARTZ_3.0', 'DMDEDUC2', 'DMDHHSIZ', 'RXQ050', 'SLD012', 'ALQ130',
       'FNDADI', 'PAD680']

## SETTINGS ##
# Access the selected language from session state
language = st.session_state.get("language", "English")  # Default to English if not set

# Create Confusion Matrix Terminology Table
confusion_matrix_data = pd.DataFrame({
    translations[language]['term']: [
        translations[language]['tp'],
        translations[language]['tn'],
        translations[language]['fp'],
        translations[language]['fn']
    ],
    translations[language]['meaning']: [
        translations[language]['tp_desc'],
        translations[language]['tn_desc'],
        translations[language]['fp_desc'],
        translations[language]['fn_desc']
    ]
})

## MAP FEATURES NAMES ##
#grouped_features = {
#    translations[language]['gender']: ['RIAGENDR_Male', 'RIAGENDR_Female'],
#    translations[language]['marital_status']: ['DMDMARTZ_2.0', 'DMDMARTZ_3.0'],
#    translations[language]['age']: ['RIDAGEYR'],
#    translations[language]['education_level']: ['DMDEDUC2'],
#    translations[language]['household_size']: ['DMDHHSIZ'],
#    translations[language]['medication_use']: ['RXQ050'],
#    translations[language]['sleep_hours']: ['SLD012'],
#    translations[language]['drinking_frequency']: ['ALQ130'],
#    translations[language]['disabilities']: ['FNDADI'],
#    translations[language]['sedentarism']: ['PAD680']}

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
    st.page_link("explanation.py", label=translations[language]['explain_page'], icon=":material/help:")
with col6:
    st.page_link("Survey.py", label=translations[language]['survey_page'], icon=":material/edit:")

st.header(translations[language]['more_info_page'])

plot_custom_decision_tree(
    dt_model, 
    feature_names=features_names, 
    class_names=['Non Depression', 'Depression'],
    title="Decision Tree - Depression Classification"
)

st.subheader(translations[language]['explanation_title_1'])

st.write(translations[language]['explanation_intro'])

plot_model_results(metrics_df, language)

# Display in Streamlit
st.write(translations[language]['confusion_matrix_terminology'])
st.table(confusion_matrix_data)


st.markdown(f"<h3>{translations[language]['accuracy_title']}</h3>", unsafe_allow_html=True)

st.write(translations[language]['accuracy_explain'])

col5, col6 = st.columns([1, 4])  # Adjust the ratio as needed
with col5:
    st.write(translations[language]['test_accuracy'])
with col6:
    st.latex(r" = \frac{TP + TN}{TP + TN + FP + FN}")

col5, col6 = st.columns([1, 4])  # Adjust the ratio as needed

# Place the image in the first column
with col5:
    st.write(translations[language]['how_interpret'] + ': ')

# Place the title in the second column
with col6:
    st.info(translations[language]['high_accuracy'])

    st.info(translations[language]['low_accuracy'])

st.markdown(f"<h3>{translations[language]['metrics_summary']}</h3>", unsafe_allow_html=True)
st.table([
    [translations[language]['test_accuracy'],r"$ \frac{TP + TN}{TP + TN + FP + FN} $"],
    [translations[language]['test_precision'], r"$ \frac{TP}{TP + FP} $"],
    [translations[language]['test_recall'], r"$ \frac{TP}{TP + FN} $"],
    [translations[language]['test_f1'], r"$ 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $"],
    [translations[language]['auc'], translations[language]['auc_table_explain']]
])

# Adding links for more information
link ="https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023"
st.markdown(f"{translations[language]['link_1']}")
st.markdown(f"[{translations[language]['link_2']}]({link})")