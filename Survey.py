import streamlit as st

import streamlit as st

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
        "title_1": "Welcome to DepreScan, a tool to assist healthcare professionals in detecting depression in Primary Care",
        "title_2": "Survey on Machine Learning Explanatory Models",
        "text_1":"This tool is based on the use of Machine Learning (ML) to assess the likelihood of a person having depression from readily available data",
        "text_2":"For more information about the ML model used, the training data and performance metrics, see the More Information page",
        "text_3": "If you would like to participate in the survey, on the Questionnaire page, you will find some tasks and questions about DepreScan, the ML model and the explanations",
        "text_4":"Thank you for your support and interest in this Project!"
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
        "title_1": "Bienvenido a DepreScan, una herramienta para ayudar a los profesionales sanitarios a detectar la depresión en Atención Primaria",
        "title_2": "Investigación sobre modelos explicativos de aprendizaje automático",
        "text_1":"Esta herramienta se basa en el uso de aprendizaje automático (ML) para evaluar la probabilidad de que una persona tenga depresión a partir de datos fácilmente disponibles",
        "text_2":"Para obtener más información sobre el modelo ML utilizado, los datos de entrenamiento y las métricas de rendimiento, consulte la página Más información",
        "text_3": "Si desea participar en la encuesta, en la página Quiz encontrará algunas tareas y preguntas sobre DepreScan, el modelo ML y explicaciones",
        "text_4":"¡Gracias por su apoyo e interés en este proyecto!"
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
        "title_1": "Bem vindos ao DepreScan, uma ferramenta para auxiliar profissionais da saúde na detecção de depressão na Atenção Primária",
        "title_2": "Pesquisa em Modelos Explicativos de Machine Learning",
        "text_1":"Essa ferramenta se baseia no uso de Machine Learning (ML) para avaliar a probabilidade de uma pessoa ter depressão a partir de dados facilmente disponíveis",
        "text_2":"Para mais informações sobre o modelo de ML utilizado, os dados de treino e métricas de performance, veja a página Mais Informações",
        "text_3": "Se você deseja participar da pesquisa na página Questionário, você encontrará algumas tarefas e perguntas sobre o DepreScan, o modelo de ML e as explicações",
        "text_4":"Obrigado pelo apoio e interesse neste Projeto!"
    }
}

## SETTINGS ##
# Access the selected language from session state
language = st.session_state.get("language", "English")  # Default to English if not set

# Define Colors
COLORS = {
    "title": "#0D47A1",       # Deep Blue
    "header": "#1976D2",      # Medium Blue
    "subheader": "#009688",   # Teal
    "text": "#455A64",        # Dark Gray-Blue
    "success": "#43A047",     # Green
    "warning": "#FF9800",     # Orange
    "error": "#E53935",       # Red
}

# Streamlit Page Title
st.markdown(f"<h1 style='color: {COLORS['title']};'>Health App Dashboard</h1>", unsafe_allow_html=True)

# Headers and Subheaders
st.markdown(f"<h2 style='color: {COLORS['header']};'>Overview</h2>", unsafe_allow_html=True)
st.markdown(f"<h3 style='color: {COLORS['subheader']};'>User Statistics</h3>", unsafe_allow_html=True)
st.markdown(f"<h4 style='color: {COLORS['text']};'>Recent Activity</h4>", unsafe_allow_html=True)

# Information Messages
st.markdown(f"<p style='color: {COLORS['success']}; font-weight: bold;'>✅ All systems are running smoothly.</p>", unsafe_allow_html=True)
st.markdown(f"<p style='color: {COLORS['warning']}; font-weight: bold;'>⚠️ Warning: Some users haven't completed their profiles.</p>", unsafe_allow_html=True)
st.markdown(f"<p style='color: {COLORS['error']}; font-weight: bold;'>❌ Error: Failed to fetch patient records.</p>", unsafe_allow_html=True)

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
    st.page_link("MoreInfo.py", label=translations[language]['more_info_page'], icon=":material/info:")

st.header(translations[language]['survey_page'])

st.markdown(f"<h2>{translations[language]['app_title']}</h2>", unsafe_allow_html=True)

st.markdown(f"<h3>{translations[language]['title_1']}</h3>", unsafe_allow_html=True)

st.write(f"{translations[language]['text_1']}")
st.write(f"{translations[language]['text_2']}")

st.markdown(f"<h3>{translations[language]['title_2']}</h3>", unsafe_allow_html=True)

st.write(f"{translations[language]['text_3']}")

st.write(f"{translations[language]['text_4']}")