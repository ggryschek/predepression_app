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
        "app_title": "Depression Screening for Primary Care",
        "app_name": "DepreScan",
        "sidebar_header": "Inform Patients Data",
        "page1_title": "Depression Prediction",
        "page2_title": "More Information",
        "dont_know":"Don't know",
        "No": "No",
        "Yes": "Yes",
        "survey":"Survey",
        "more_info": "More Information",
        "explanations":"Explanations",
        "title_1": "Welcome to DepreScan, a tool to assist healthcare professionals in detecting depression in Primary Care",
        "title_2": "Survey on Machine Learning Explanatory Models",
        "text_1":"This tool is based on the use of Machine Learning (ML) to assess the likelihood of a person having depression from readily available data",
        "text_tool_use":"This tool uses readily available data to predict whether a person with these characteristics may have depression. In the 'DepreScan' tab, you can enter the information and see the prediction result, as well as understand how each piece of data contributed to this specific prediction (i.e. local explanation)",
        "text_explainpage":"To understand the importance of each variable in training the model, access the 'Explanations' tab. It is worth remembering that no variable influences the result in isolation. The Machine Learning model learns the patterns between the data and makes the prediction based on these relationships, indicating the possibility of depression or not. This is called the global explanation of the model, which shows how it learned from the training data and the importance of each of the variables in making the predictions",
        "text_infopage_1":"There are several Machine Learning models used for prediction tasks, such as the one used in DepreScan. In this case, the model that performed best was the LGBM, and it is this that is behind the tool's predictions",
        "text_infopage_2":"To learn more about the LGBM model, training data, performance metrics, and other details, visit the 'More Information' tab",
        "text_survey": "This online tool is part of a research project on explanatory models for Machine Learning applied to Healthcare. If you are a Primary Care healthcare professional and would like to contribute to this research, see the 'Survey' tab to learn how to participate",
        "text_thanks":"Thank you for your support and interest in this Project!"
    },
    "Español": {
        "app_name": "DepreScan",
        "app_title": "Detección de Depresión en atención primaria",
        "sidebar_header": "Introducir datos del paciente",
        "page1_title": "Predicción de la depresión",
        "page2_title": "Más Informatión",
        "dont_know":"No sé",
        "No": "No",
        "Yes": "Sin",
        "survey":"Pesquisa",
        "more_info": "Más Información",
        "explanations":"Explicações",
        "title_1": "Bienvenido a DepreScan, una herramienta para ayudar a los profesionales sanitarios a detectar la depresión en Atención Primaria",
        "title_2": "Investigación sobre modelos explicativos de Machine Learning",
        "text_1":"Esta herramienta se basa en el uso de Machine Learning (ML) para evaluar la probabilidad de que una persona tenga depresión a partir de datos fácilmente disponibles",
        "text_tool_use":"Esta herramienta utiliza datos fácilmente disponibles para predecir si una persona con estas características puede tener depresión. En la pestaña 'DepreScan', puede ingresar la información y ver el resultado de la predicción, así como comprender cómo cada dato contribuyó a esa predicción específica (es decir, explicación local)",
        "text_explainpage":"Para comprender la importancia de cada variable en el entrenamiento del modelo, acceda a la pestaña “Explicaciones”. Vale la pena recordar que ninguna variable influye en el resultado de forma aislada. El modelo de Machine Learning aprende los patrones entre los datos y hace predicciones basadas en estas relaciones, indicando la posibilidad de depresión o no. Esto se llama explicación global del modelo, que muestra cómo aprendió de los datos de entrenamiento y la importancia de cada una de las variables para realizar las predicciones",
        "text_infopage_1":"Hay varios modelos de Machine Learning que se utilizan para tareas de predicción, como el utilizado en DepreScan. En este caso, el modelo que mejor se comportó fue LGBM, y es el que está detrás de las predicciones de la herramienta",
        "text_infopage_2":"Para obtener más información sobre el modelo LGBM, datos de entrenamiento, métricas de rendimiento y otros detalles, visite la pestaña 'Más información'",
        "text_survey": "Esta herramienta online forma parte de un proyecto de investigación sobre modelos explicativos de Machine Learning aplicado a la Salud. Si eres un profesional sanitario de Atención Primaria y quieres contribuir a esta investigación, consulta la pestaña 'Survey / Pesquisa' para saber cómo participar",
        "text_thanks":"¡Gracias por su apoyo e interés en este proyecto!"
    },
    "Português brasileiro": {
        "app_name": "DepreScan",
        "app_title": "Rastreio de Depressão na Atenção Primária",
        "sidebar_header": "Informe os dados do paciente",
        "page1_title": "Predição de Depressão",
        "page2_title": "Mais Informações",
        "dont_know":"Não sei",
        "No": "Não",
        "Yes": "Sim",
        "survey":"Pesquisa",
        "more_info": "Mais Informações",
        "explanations":"Explicações",
        "title_1": "Bem vindos ao DepreScan, uma ferramenta para auxiliar profissionais da saúde na detecção de depressão na Atenção Primária",
        "title_2": "Pesquisa em Modelos Explicativos de Machine Learning",
        "text_1":"Essa ferramenta se baseia no uso de Machine Learning (ML) para avaliar a probabilidade de uma pessoa ter depressão a partir de dados facilmente disponíveis",
        "text_tool_use":"Essa ferramenta usa dados facilmente disponíveis para prever se uma pessoa com essas características pode ter depressão. Na aba 'DepreScan', você pode inserir as informações e ver o resultado da predição, além de entender como cada dado contribuiu para essa previsão específica (i.e. explicação local)",
        "text_explainpage":"Para entender a importância de cada variável no treinamento do modelo, acesse a aba 'Explanations'. Vale lembrar que nenhuma variável influencia o resultado de forma isolada. O modelo de Machine Learning aprende os padrões entre os dados e faz a predição com base nessas relações, indicando a possibilidade de depressão ou não. Essa é chamada explicação global do modelo, que mostra como ele aprendeu dos dados de treinamento e a importância de cada uma das variáveis para fazer as previsões",
        "text_infopage_1":"Existem vários modelos de Machine Learning usados para tarefas de predição, como o usado no DepreScan. Neste caso, o modelo que apresentou melhor desempenho foi o LGBM, e é ele que está por trás das previsões da ferramenta",
        "text_infopage_2":"Para saber mais sobre o modelo LGBM, os dados de treinamento, métricas de desempenho e outros detalhes, acesse a aba 'Mais Informações'",
        "text_survey": "Essa ferramenta online é parte de um projeto de pesquisa sobre modelos expicativos para Machine Learning aplicado à Saúde. Se você é um profissional de saúde da Atenção Primária e deseja contribuir para essa pesquisa, veja a aba 'Survey / Pesquisa' para saber como participar",
        "text_thanks":"Obrigado pelo apoio e interesse neste Projeto!"
    }
}

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

st.markdown(f"<h2>{translations[language]['app_title']}</h2>", unsafe_allow_html=True)

st.markdown(f"<h3>{translations[language]['title_1']}</h3>", unsafe_allow_html=True)

# Streamlit main text
col3, col4 = st.columns([2,7])  # Adjust the ratio as needed

with col3:
    st.page_link("DepressionPrediction.py", label="🧠 DepreScan")

with col4:
    st.write(f"{translations[language]['text_tool_use']}")

col5, col6 = st.columns([2,7])  # Adjust the ratio as needed

with col5:
    st.page_link("explanation.py", label=f"❓+ {translations[language]['explanations']}")

with col6:
    st.write(f"{translations[language]['text_explainpage']}")

st.markdown(f"<h4>Light Gradient-Boosting Machine (LGBM)</h4>", unsafe_allow_html=True)

col7, col8 = st.columns([2,7])  # Adjust the ratio as needed

with col7:
    st.page_link("MoreInfo.py", label=f"ℹ️ {translations[language]['more_info']}")

with col8:
    st.write(f"{translations[language]['text_infopage_1']}")
    st.write(f"{translations[language]['text_infopage_2']}")

st.markdown(f"<h3>{translations[language]['title_2']}</h3>", unsafe_allow_html=True)

col9, col10 = st.columns([2,7])  # Adjust the ratio as needed

with col9:
    st.page_link("Survey.py", label=f"✏️ {translations[language]['survey']}")

with col10:
    st.write(f"{translations[language]['text_survey']}")

st.write('')
#st.write(f"{translations[language]['text_thanks']}")
st.markdown(f"<h4>{translations[language]['text_thanks']}</h4>", unsafe_allow_html=True)
#