import streamlit as st

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
        .stSidebar h4 { font-size: 1.1rem; color: #444444; }
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
        "text_1":"Help Us Improve Explainable AI for Depression Screening in Primary Care",
        "text_2":"Machine Learning (ML) models are transforming healthcare, but they are often seen as 'black boxes'— making predictions without clear explanations. Explanatory models help bridge this gap by providing insights into how these models reach their conclusions. Techniques like SHAP values, partial dependence plots, and feature importance analysis can enhance transparency, build trust, and support better decision-making in clinical practice.",
        "text_3": "This research focuses on depression screening in primary care and aims to understand whether such ML-based tools would be practical and valuable in real-world scenarios.",
        "search":"We are looking for",
        "psycho":"Psychologists", 
        "nurse": "Nurses",
        "clinic":"General Practitioners/Family Physicians",
        "experience": "with experience in identifying and treating depressive patients in primary care to share their perspectives.",
        "how_participate":"How to Participate",
        "text_4":"If you are willing to contribute to this research, please follow these steps:",
        "step_1":"1. Click the link – A new tab will open with the questionnaire. Keep the DepreScan tab open, as you will use it during the study.",
        "link1":"To participate in the Research click",
        "link2":"HERE",
        "step_2":"2. Read the consent form – If you agree, provide your consent to proceed.",
        "step_3":"3. Complete the questionnaire – Follow the instructions carefully to share your insights.",
        "text_5":"Your participation will help refine AI-driven depression screening, making it more interpretable, reliable, and aligned with real-world needs.",
        "thanks":"Thank you for your time and contribution!"
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
        "title_2": "Encuesta sobre modelos explicativos de Machine Learning",
        "text_1":"Ayúdenos a mejorar la IA explicable para la detección de la depresión en atención primaria",
        "text_2":"Los modelos de Machine Learning (ML) están transformando la atención médica, pero a menudo se consideran 'cajas negras' que hacen predicciones sin explicaciones claras. Los modelos explicativos ayudan a superar esta brecha al proporcionar información sobre cómo estos modelos llegan a sus conclusiones. Técnicas como los valores SHAP, los gráficos de dependencia parcial y el análisis de importancia de características pueden mejorar la transparencia, generar confianza y facilitar una mejor toma de decisiones en la práctica clínica.",
        "text_3": "Esta investigación se centra en la detección de la depresión en atención primaria y busca comprender si estas herramientas basadas en ML serían prácticas y valiosas en situaciones reales.",
        "search":"Buscamos",
        "psycho":"Psicólogos", 
        "nurse": "Enfermeros",
        "clinic":"Médicos de cabecera/familia",
        "experience": "con experiencia en la identificación y tratamiento de pacientes deprimidos en atención primaria para compartir sus perspectivas",
        "how_participate":"Cómo participar",
        "text_4":"Si desea contribuir a esta investigación, siga estos pasos:",
        "step_1":"1. Haga clic en el enlace: se abrirá una nueva pestaña con el cuestionario. Mantenga abierta la pestaña DepreScan, ya que la utilizará durante el estudio.",
        "link1":"Para participar en la investigación, haga clic",
        "link2":"AQUÍ",
        "step_2":"2. Lea el formulario de consentimiento. Si está de acuerdo, dé su consentimiento para continuar.",
        "step_3":"3. Complete el cuestionario. Siga atentamente las instrucciones para compartir sus observaciones.",
        "text_5":"Su participación ayudará a perfeccionar la detección de la depresión basada en IA, haciéndola más interpretable, fiable y adaptada a las necesidades del mundo real.",
        "thanks":"¡Gracias por su tiempo y contribución!"
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
        "text_1":"Ajude-nos a melhorar a IA explicável para triagem de depressão na atenção primária",
        "text_2":"Os modelos de Machine Learning (ML) estão transformando a assistência médica, mas geralmente são vistos como 'caixas pretas' — fazendo previsões sem explicações claras. Os modelos explicativos ajudam a preencher essa lacuna, fornecendo insights sobre como esses modelos chegam às suas conclusões. Técnicas como valores SHAP, gráficos de dependência parcial e análise de importância de variáveis podem aumentar a transparência, criar confiança e dar suporte a uma melhor tomada de decisão na prática clínica.",
        "text_3": "Esta pesquisa se concentra no screening de depressão na atenção primária e visa entender se essas ferramentas baseadas em ML seriam práticas e valiosas em cenários do mundo real.",
        "search":"Estamos procurando",
        "psycho":"Psicólogos", 
        "nurse": "Enfermeiros",
        "clinic":"Clínicos gerais/Médicos de família",
        "experience": "com experiência na identificação e tratamento de pacientes deprimidos na atenção primária para compartilhar suas perspectivas",
        "how_participate":"Como participar",
        "text_4":"Se você estiver disposto a contribuir para esta pesquisa, siga estas etapas:",
        "step_1":"1. Clique no link – Uma nova guia será aberta com o questionário. Mantenha a guia DepreScan aberta, pois você a usará durante o estudo.",
        "link1":"Para participar na Pesquisa clique",
        "link2":"AQUI",
        "step_2":"2. Leia o formulário de consentimento – Se concordar, dê seu consentimento para prosseguir.",
        "step_3":"3. Preencha o questionário – Siga as instruções cuidadosamente para compartilhar seus insights.",
        "text_5":"Sua participação ajudará a aprimorar o screening de depressão orientada por IA, tornando-a mais interpretável, confiável e alinhada às necessidades do mundo real.",
        "thanks":"Obrigado pelo seu tempo e participação!"
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
#st.markdown(f"<h1 style='color: {COLORS['title']};'>Health App Dashboard</h1>", unsafe_allow_html=True)

# Headers and Subheaders
#st.markdown(f"<h2 style='color: {COLORS['header']};'>Overview</h2>", unsafe_allow_html=True)
#st.markdown(f"<h3 style='color: {COLORS['subheader']};'>User Statistics</h3>", unsafe_allow_html=True)
#st.markdown(f"<h4 style='color: {COLORS['text']};'>Recent Activity</h4>", unsafe_allow_html=True)

# Information Messages
#st.markdown(f"<p style='color: {COLORS['success']}; font-weight: bold;'>✅ All systems are running smoothly.</p>", unsafe_allow_html=True)
#st.markdown(f"<p style='color: {COLORS['warning']}; font-weight: bold;'>⚠️ Warning: Some users haven't completed their profiles.</p>", unsafe_allow_html=True)
#st.markdown(f"<p style='color: {COLORS['error']}; font-weight: bold;'>❌ Error: Failed to fetch patient records.</p>", unsafe_allow_html=True)

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

st.header(translations[language]['title_2'])

#st.markdown(f"<h2>{translations[language]['app_title']}</h2>", unsafe_allow_html=True)

#st.markdown(f"<h3>{translations[language]['title_1']}</h3>", unsafe_allow_html=True)

st.write(f"### {translations[language]['text_1']}")
st.write(f"{translations[language]['text_2']}")

#st.markdown(f"<h2>{translations[language]['title_2']}</h2>", unsafe_allow_html=True)
link = "https://developer.mozilla.org/pt-BR/docs/Web/CSS/color_value#palavras-chave_de-cores"

st.write(f"{translations[language]['text_3']}")
st.write(" ")

col7, col8, col9 = st.columns([2,3,4])
with col7:
    st.write(f"#### {translations[language]['search']}")
with col8:
    st.info(f" - {translations[language]['psycho']}")

    st.info(f" - {translations[language]['nurse']}")

    st.info(f" - {translations[language]['clinic']}")
with col9:
    st.write(f"#### {translations[language]['experience']}")

st.write(f"### {translations[language]['text_4']}")

st.info(f"{translations[language]['step_1']}")

col10, col11 = st.columns([1,8])
with col10:
    st.write(" ") #👉➡️
with col11:
#st.write(" ")
    st.write(f"## ➡️{translations[language]['link1']} [{translations[language]['link2']}]({link}) ⬅️", unsafe_allow_html=True)
#with col12:
    #st.write("# ⬅️") #👈

st.info(f"{translations[language]['step_2']}")

st.info(f"{translations[language]['step_3']}")

st.write(f"{translations[language]['text_5']}")

col13, col14 = st.columns([1,4])
with col13:
    st.write(" ")
with col14:
    st.write(f"### {translations[language]['thanks']}")