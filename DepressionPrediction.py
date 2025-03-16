import streamlit as st
import pandas as pd
import shap
import lime.lime_tabular
import lime
import pickle
import numpy as np
import streamlit.components.v1 as components
import matplotlib as mplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg
import joblib

### FORMATTING PARAMETERS ###
st.markdown("""
    <style>
        /* Header Styles */
        h1, .stMarkdown h1 { font-size: 2.5rem; color: royalblue; font-weight: bold; }
        h2, .stMarkdown h2 { font-size: 1.8rem; color: darkorange; font-weight: bold; }
        h3, .stMarkdown h3 { font-size: 1.4rem; color: #008080; font-weight: bold; }
        h4, .stMarkdown h4 { font-size: 1.2rem; color: #444444; }

        /* Sidebar Header Styles */
        .stSidebar h1 { font-size: 2rem; color: royalblue; font-weight: bold; }
        .stSidebar h2 { font-size: 1.6rem; color: darkorange; font-weight: bold; }
        .stSidebar h3 { font-size: 1.3rem; color: #008080; font-weight: bold; }
        .stSidebar h4 { font-size: 1.1rem; color: #444444; }
    </style>
""", unsafe_allow_html=True)

# Check colors: https://developer.mozilla.org/pt-BR/docs/Web/CSS/color_value#palavras-chave_de-cores

### DICTIONARY FOR AUTOMATED TRANSLATION ###
# Language dictionary with translations
translations = {
    "English": {
        "app_name": "DepreScan",
        "app_title": "Depression Screening for Primary Care",
        "sidebar_header": "⚠️ Inform HERE Patients Data",
        "home_page":"Home",
        "depre_page":"DepreScan",
        "explain_page":"Explanations",
        "more_info_page":"More Info",
        "survey_page":"Survey",
        "dont_know":"Don't know",
        "No": "No",
        "Yes": "Yes",
        "age": "Age (in years)",
        "age_help": "Enter the age in years (18-100). This tool was trained with adults data",
        "gender": "Biological Sex",
        "male":"Male",
        "female":"Female",
        "gender_help": "Select the biological sex",
        "marital_status": "Marital Status",
        "Married": "Married/Living with partner",
        "Divorced": "Divorced/Separated/Widowed", 
        "Never Married": "Never Married",
        "marital_help": "Inform the marital status among the options. If unsure, choose 'Don't Know' above",
        "education_level": "Educational Level",
        "educ_1": "Less than 9th grade",
        "educ_2":"9-11th grade",
        "educ_3":"High school graduate/GED or equivalent",
        "educ_4":"Some college or AA degree",
        "educ_5":"College graduate or above",
        "educ_help":"Educational level depends on the local context. Consider possible equivalencies in your case. The College or above option includes master's, doctorate and other postgraduate degrees. If unsure, choose 'Don't Know'", 
        "household_size": "People living in the Household",
        "household_slider": "How many people living in the same house?",
        "household_help": "Inform the total number of people living in the household, including yourself/the patient. If 7 or more, choose 7. If unsure, check 'Don't know' above",
        "medication_use": "Medication Use",
        "medication_slider": "How many prescription medications {have you/has SP} taken in the past 30 days?",
        "medication_help": "Consider only medications prescribed by a doctor or other health care professional. Do not consider medications you have taken on your own. If unsure, check 'Don't know' above",
        "health_rating": "Self-reported General Health",
        "Excellent": "Excellent",
        "Very Good": "Very Good",
        "Good": "Good",
        "Fair": "Fair",
        "Poor": "Poor",
        "health_help": "Would (you/the patient) say (your/his/her) health in general is: Excellent, Very Good, Good, Fair, Poor? If unsure, check 'Don't Know' above",
        "sedentarism":"Rest Time",
        "sedentarism_input": "How many hours do you sit per day, on average? Do not include time spent sleeping! (Assume a maximum of 23h)",
        'sedentarism_help': "Please enter the hours you spend sitting on average on a typical day, at school, at home, getting to and from places, or with friends including time spent sitting at a desk, traveling in a car or bus, reading, playing cards, watching television, or using a computer. If you are unsure, check 'Don't know' above",
        "sleep_hours": "Sleep Hours",
        "sleep_habits": "Sleep Habits",
        "sleep_slider": "Number of hours usually sleep on weekdays or workdays",
        "sleep_help":"Inform number of hours of sleep in a typical workday. If less than 3, choose 2. If 14 or more, choose 14. If unsure, check 'Don't know' above",
        "drinking_frequency": "Alcohol Use",
        "drinking_slider": "In the last 12 months, on the days when (you/the patient) drank alcoholic beverages, on average, how many drinks did (you/the patient) have?",
        "drinking_help": "One drink: 350 ml beer or a 140 ml glass of wine, or a 40 ml shot of spirits; If 15 or more, choose 15. If unsure, check 'Don't know' above",
        "disabilities": "Any Disabilities?",
        "disab_help": "Disabilities may include visual, auditory, locomotion, communication, self-care, and memory difficulties. If unsure, check 'Don't know' above. For more information about disabilities check the link below",
        "link": "click here",
        "more_info": "For more on how evaluate disabilities",
        "default_info": "You must inform at least Age and Biological Sex. By default, all other information were set to 'Don't know': updated it once you have the right data. The more data provided, the better the prediction will be!",
        "feature":"Features", 
        "user_selection": "Informed Data",
        "user_review": "✅ Data Summary",
        "predict_button": "Predict",
        "prediction_result": "✅ Prediction Result",
        "predicted_class": "🔍 Predicted Classification",
        "prediction_probabilities": "Prediction Probabilities: ",
        "class_probabilities_title": "Condition Probabilities",
        "probability": "Probability",
        "True": "Depression",
        "False": "Non-depression",
        "non_depression": "Non-Depression",
        "depression": "Depression",
        "check_table_fail":"⚠️ We still have 'Don't know': remember this might affect the prediction results!",
        'check_table_ok':'✅ Thanks for providing as much data as possible: this will make the prediction more accurate!',
        "shap":"Understand this result with SHAP (SHapley Additive exPlanations)",
        "shap_waterfall_title":"SHAP Waterfall Plot",
        "waterfall_intro":"🔍 Check how each feature affects the model’s prediction",
        "waterfall_x_label":"SHAP Values: f(x) = 0 means 50% probabilities for each condition",
        "shap_help":"This method uses concepts from game theory to explain how machine learning models make predictions. It works by fairly distributing the importance of each feature (input variable) in a prediction, similar to how rewards are shared among players in a cooperative game. This approach is based on Shapley values, a game theory concept that ensures each feature gets credit based on its contribution to the final prediction. By doing this, it provides clear and interpretable explanations of model decisions at an individual level"
    },
    "Español": {
        "app_name": "DepreScan",
        "app_title": "Detección de la depresión en atención primaria",
        "sidebar_header": "⚠️ Informe AQUÍ los datos del paciente",
        "home_page":"Início",
        "depre_page":"DepreScan",
        "explain_page":"Explicaciones",
        "more_info_page":"Más Información",
        "survey_page":"Investigación",
        "age": "Edad (en años)",
        "age_help": "Ingresa la edad en años (18-100). Esta herramienta fue entrenada con datos de adultos",
        "gender": "Sexo biológico",
        "male":"Masculino",
        "female":"Feminino",
        "dont_know":"No sé",
        "No": "No",
        "Yes": "Sin",
        "gender_help": "Seleccione el sexo biológico",
        "marital_status": "Estado Civil",
        "Married": "Casado/Vive con su pareja",
        "Divorced": "Divorciado/Separado/Viudo", 
        "Never Married": "Nunca casado",
        "marital_help": "Indique el estado civil entre las opciones. Si no está seguro, seleccione 'No sé' arriba",
        "education_level": "Nivel de estudios",
        "educ_1": "Menos de 9º grado",
        "educ_2":"Grado 9º a 11º",
        "educ_3":"Graduado de la escuela secundaria o equivalente",
        "educ_4":"Algún título universitario",
        "educ_5":"Graduado universitario o superior",
        "educ_help":"El nivel educativo depende del contexto local. Considere las posibles equivalencias en su caso. La opción de nivel universitario o superior incluye maestrías, doctorados y otros títulos de posgrado. Si no está seguro, seleccione 'No sé' arriba.", 
        "household_size": "Tamaño del Hogar",
        "household_help": "Indique el número total de personas que viven en el hogar, incluido usted/el paciente. Si son 7 o más, seleccione 7. Si no está seguro, marque 'No sé' arriba",
        "household_slider": "¿Cuántas personas viven en la misma casa?",
        "medication_use": "Uso de medicamentos",
        "medication_slider": "¿Cuántos medicamentos recetados ha tomado usted o su pareja en los últimos 30 días?",
        "medication_help": "Considere solo los medicamentos recetados por un médico u otro profesional de la salud. No considere los medicamentos que haya tomado por su cuenta. Si no está seguro, marque 'No sé' arriba",
        "health_rating": "Salud general autoinformada",
        "Excellent": "Excelente",
        "Very Good": "Muy buena",
        "Good": "Buena",
        "Fair": "Regular",
        "Poor": "Mala",
        "health_help": "¿Diría (usted/el paciente) que su salud en general es: Excelente, Muy buena, Buena, Regular, Mala? Si no está seguro, marque 'No sé' arriba",
        "sedentarism":"Tiempo de Descanso",
        "sedentarism_input": "¿Cuántas horas pasas sentado al día, en promedio? ¡No incluya el tiempo que pasa durmiendo! (Considere un máximo de 23 h)",
        'sedentarism_help': "Indique el número promedio aproximado de horas que pasa sentado en un día típico, en la escuela, en casa, yendo y viniendo de un lugar o con amigos, incluyendo el tiempo que pasa sentado en un escritorio, viajando en un automóvil o autobús, leyendo, jugando a las cartas, mirando televisión o usando una computadora. Si no está seguro, marque 'No sé' arriba",
        "sleep_hours": "Horas de Sueño",
        "sleep_habits": "Hábito de Sono",
        "sleep_slider": "Número de horas que suele dormir entre semana o en días laborables",
        "sleep_help":"Indique el número de horas de sueño en un dia habitual. Si es menos de 3, elija 2. Si es 14 o más, seleccione 14. Si no está seguro, marque 'No sé' arriba",
        "drinking_frequency": "Consumo de alcohol",
        "drinking_slider": "En los últimos 12 meses, en los días en que (usted/el paciente) bebió bebidas alcohólicas, en promedio, ¿cuántas bebidas tomó (usted/el paciente)?",
        "drinking_help": "Una bebida: 350 ml de cerveza o una copa de vino de 140 ml, o un chupito de 40 ml de bebida espirituosa; Si son 15 o más, seleccione 15. Si no está seguro, marque 'No sé' arriba",
        "disabilities": "¿Tiene Discapacidad?",
        "disab_help": "Entre las discapacidades se incluyen dificultades visuales, auditivas, de locomoción, de comunicación, de autocuidado y de memoria. Si no está seguro, marque 'No sé' arriba. Para obtener más información sobre discapacidades, consulte el link abajo",
        "link": "clic aquí",
        "more_info": "Para ver más sobre cómo evaluar las discapacidades",
        "default_info": "Debes indicar al menos la edad y el sexo. De forma predeterminada, el resto de la información se establece en 'No sé': actualízala una vez que tengas los datos correctos. Cuantos más datos se proporcionen, mejor será la predicción!",
        "feature":"Características", 
        "user_selection": "Datos informados",
        "user_review": "✅ Resumen de datos",
        "predict_button": "Predecir",
        "prediction_result": "✅ Resultado de la Predicción",
        "predicted_class": "🔍 Clasificación Predicha",
        "prediction_probabilities": "Probabilidades de Predicción:",
        "class_probabilities_title": "Probabilidades de condiciones",
        "probability": "Probabilidad",
        "True": "Depresión",
        "False": "No Depresión",
        "non_depression": "No Depresión",
        "depression": "Depresión",
        "check_table_fail":"⚠️ Aún tenemos 'No sé': ¡recuerde que esto podría afectar los resultados de la predicción!",
        'check_table_ok':'✅ ¡Gracias por proporcionar la mayor cantidad de datos posible: esto hará que la predicción sea más precisa!',
        "shap":"SHAP (SHapley Additive exPlanations)",
        "shap_waterfall_title":"SHAP Waterfall",
        "waterfall_intro":"🔍 Comprueba cómo cada característica afecta la predicción del modelo",
        "waterfall_x_label":"SHAP Values: f(x) = 0 means 50% probabilities for each condition",
        "shap_help": "Este método utiliza conceptos de la teoría de juegos para explicar cómo los modelos de Machine Learning hacen predicciones. Funciona distribuyendo equitativamente la importancia de cada característica (variable de entrada) en una predicción, de manera similar a cómo se reparten las recompensas entre los jugadores en un juego cooperativo. Este enfoque se basa en los valores de Shapley, un concepto de la teoría de juegos que garantiza que cada característica reciba crédito en función de su contribución a la predicción final. De esta manera, proporciona explicaciones claras e interpretables de las decisiones del modelo a nivel individual"
    },
    "Português brasileiro": {
        "app_name": "DepreScan",
        "app_title": "Detecção de Depressão na Atenção Primária",
        "sidebar_header": "⚠️ Informe AQUI os dados do paciente",
        "home_page":"Início",
        "depre_page":"DepreScan",
        "explain_page":"Explicações",
        "more_info_page":"Mais Informações",
        "survey_page":"Pesquisa",
        "dont_know":"Não sei",
        "No": "Não",
        "Yes": "Sim",
        "age": "Idade (em anos)",
        "age_help": "Insira a idade em anos (18-100). Esta ferramenta foi treinada com dados de adultos",
        "gender": "Sexo Biológico",
        "male":"Masculino",
        "female":"Feminino",
        "gender_help": "Selecione o sexo biológico",
        "marital_status": "Estado Civil",
        "Married": "Casado/Vive com Parceiro",
        "Divorced": "Divorciado/Separado/Viúvo", 
        "Never Married": "Nunca casou",
        "marital_help": "Informe o estado civil entre as opções. Se não tiver certeza, escolha 'Não sei' acima",
        "education_level": "Nível de Estudos",
        "educ_1": "Ensino Fundamental (1º a 9º série) ou menos",
        "educ_2":"Ensino Médio incompleto",
        "educ_3":"Ensino Médio completo",
        "educ_4":"Ensino Superior incompleto (Faculdade/Graduação) incompleto",
        "educ_5":"Ensino Superior completo (Faculdade/Graduação) ou mais (Pós-graduação, Mestrado, Doutorado)",
        "educ_help":"Nível educacional depende do contexto local. Considere possíveis equivalências no seu caso. A opção Faculdade ou acima inclui mestrado, doutorado e outras pós-graduações. Se incerto, escolha 'Não sei' acima", 
        "household_size": "Pessoas morando na casa",
        "household_help": "Informe o número total de pessoas que moram no domicílio, incluindo você/o paciente. Se forem 7 ou mais, escolha 7. Se não tiver certeza, marque 'Não sei' acima",
        "household_slider": "Quantas pessoas vivendo na mesma casa?",
        "medication_use": "Uso de Medicações",
        "medication_slider": "Quantos medicamentos prescritos tomou nos últimos 30 dias?",
        "medication_help": "Considere apenas medicamentos prescritos por um médico ou outro profissional de saúde. Não considere medicamentos que você tomou por conta própria. Se não tiver certeza, marque 'Não sei' acima",
        "health_rating": "Estado Geral de Saúde Auto-referido",
        "Excellent": "Excelente",
        "Very Good": "Muito boa",
        "Good": "Boa",
        "Fair": "Razoável",
        "Poor": "Ruim",
        "health_help": "(Você/o paciente) diria que sua saúde em geral é: Excelente, Muito boa, Boa, Razoável, Ruim? Se não tiver certeza, marque 'Não sei' acima",
        "sedentarism":"Tempo em Repouso",
        "sedentarism_input": "Quantas horas fica sentado/deitado por dia, em média? (Considere máximo de 23h)",
        'sedentarism_help': "Informe a quantidade de horas aproximada que passa sentado ou deitado, em média, em um dia comum, na escola, em casa, indo e vindo de lugares, ou com amigos, incluindo tempo gasto sentado em uma mesa, viajando em um carro ou ônibus, lendo, jogando cartas, assistindo televisão ou usando um computador. Se não tiver certeza, marque 'Não sei' acima",
        "sleep_hours": "Horas de Sono",
        "sleep_habits": "Hábito de Sono",
        "sleep_slider": "Número de horas que costuma dormir durante a semana ou dias úteis",
        "sleep_help":"Informe o número de horas de sono em um dia de semana típico. Se for menor que 3, escolha 2. Se 14 ou mais, escolha 14. Se não tiver certeza, marque 'Não sei' acima",
        "drinking_frequency": "Consumo de Álcool",
        "drinking_slider": "Nos últimos 12 meses, nos dias em que (você/o paciente) bebeu bebidas alcoólicas, em média, quantas doses (você/o paciente) tomou?",
        "drinking_help": "Uma dose de álcool: 350 ml de cerveja ou uma taça de vinho de 140 ml, ou uma dose de 40 ml de destilado; Se 15 ou mais, escolha 15. Se não tiver certeza, marque 'Não sei' acima",
        "disabilities": "Possui Deficiência?",
        "disab_help": "As deficiências incluem dificuldades visuais, auditivas, de locomoção, de comunicação, de auto-cuidado e de memória. Se não tiver certeza, marque 'Não sei' acima. Para mais informações veja o link abaixo",
        "link": "clique aqui",
        "more_info": "Para ver mais como avaliar deficiências",
        "default_info": "Você deve informar pelo menos Idade e Sexo Biológico. Por padrão, todas as outras informações foram definidas como 'Não sei': atualize-as quando tiver os dados corretos. Quanto mais dados fornecidos, melhor será a predição!",
        "feature":"Características", 
        "user_selection": "Dados informados",
        "user_review": "✅ Resumo dos Dados",
        "predict_button": "Prever",
        "prediction_result": "✅ Resultado da Previsão",
        "predicted_class": "🔍 Classificação Prevista",
        "prediction_probabilities": "Probabilidades da Previsão:",
        "class_probabilities_title": "Probabilidades das Condições",
        "probability": "Probabilidade",
        "True": "Depressão",
        "False": "Sem depressão",
        "non_depression": "Não Depressão",
        "depression": "Depressão",
        "check_table_fail":"⚠️ Ainda temos 'Não sei': lembre-se de que isso pode afetar os resultados da previsão!",
        'check_table_ok':'✅ Obrigado por fornecer o máximo de dados possível: isso tornará a previsão mais precisa!',
        "shap":"Entenda esse resultado com SHAP (SHapley Additive exPlanations)",
        "shap_waterfall_title":"Cascata de Valores SHAP (Waterfall Plot)",
        "waterfall_intro":"🔍 Verifique como cada característica afeta a previsão do modelo",
        "waterfall_x_label":"Valores SHAP: f(x) = 0 significa 50% probabilidades para cada condição",
        "shap_help":"Este método usa conceitos da teoria dos jogos para explicar como os modelos de Machine Learning fazem previsões. Ele funciona distribuindo de forma justa a importância de cada característica (variável de entrada) em uma previsão, semelhante a como as recompensas são compartilhadas entre os jogadores em um jogo cooperativo. Esta abordagem é baseada em valores de Shapley, um conceito da teoria dos jogos que garante que cada recurso receba crédito com base em sua contribuição para a previsão final. Ao fazer isso, ele fornece explicações claras e interpretáveis ​​das decisões do modelo em um nível individual."
    }
}

#@st.cache_data # Rerun automatically

## SETTINGS ##
# Access the selected language from session state
language = st.session_state.get("language", "English")  # Default to English if not set

### LOAD ML MODELS ###
with open("best_model.pkl", "rb") as f: # Load trained model
    app_model = pickle.load(f)

### LOAD SCALER ###
# Load the saved scaler
scaler = joblib.load('scaler.pkl')

### LOAD TRAINSET SHAP VALUES ###
# Load the SHAP explanation object from the file
with open('trainset_shap_values.pkl', 'rb') as f:
    grouped_shap_values = pickle.load(f)

### CREATE EXPLAINER ###
# Create a SHAP explainer object for a decision tree model
explainer = shap.Explainer(app_model)

### FUNCTIONS ###

# Function for prediction logic
def predict_class_and_probabilities(model, input_df, translations, language):
    """Predicts the class and probability from a trained model."""
    
    # Predict class and probabilities
    prediction = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0]

    # Get the class with the highest probability
    max_index = np.argmax(probas)
    max_prob = probas[max_index]

    # Define class labels safely
    class_labels = {
        0: translations[language]["non_depression"], 
        1: translations[language]["depression"]
    }

    # Construct the prediction message
    prediction_message = f"**{max_prob*100:.2f}%** *({class_labels.get(max_index, 'Unknown')})*"

    return prediction, prediction_message, class_labels.get(int(prediction), "Unknown"), probas

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
    plt.figure(figsize=(2, 1))

    # Set figure background color (outside the plot)
    plt.gcf().set_facecolor("snow")  # Light gray background

    # Set plot (axes) background color and opacity
    ax = plt.gca()
    ax.set_facecolor("#e0f7fa")  # Light blue background
    ax.patch.set_alpha(0.6)  # Opacity (0 = fully transparent, 1 = fully opaque)

    plt.bar(class_labels, probabilities, color=['darkseagreen', 'lightsalmon'], alpha=0.8)

    # Add title and labels
    plt.title(translations[language]["class_probabilities_title"], fontsize=7)
    #plt.xlabel('Classes')
    plt.ylabel(translations[language]["probability"], fontsize=5)
    plt.xticks(fontsize=5)  
    plt.yticks(fontsize=5) 

    # Show the chart in Streamlit
    st.pyplot(plt)

def plot_gauge(probabilities, language):
    class_labels = [
        translations[language]["non_depression"],
        translations[language]["depression"]
    ]
    
    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probabilities[1] * 100,  # Convert probability to percentage
    title={'text': translations[language]["class_probabilities_title"], 
           'font': {'size': 20, 'color': "black"}},
    number={'suffix': "%", 'font': {'size': 24, 'color': "black"}},

    gauge={
        'axis': {
            'range': [0, 100],
            'tickvals': [0, 25, 50, 75, 100],
            'ticktext': [
                translations[language]["non_depression"], 
                "", 
                "Neutral", 
                "", 
                translations[language]["depression"]
            ],
            'tickfont': {'color': 'black', 'size': 14}  # ✅ Set tick labels color
        },
        'bgcolor': 'black',  # Background inside the gauge
        'bar': {'color': "black", 'thickness': 0.05},  # Fix thickness
        'threshold': {
            'line': {'color': "black", 'width': 4},  # Adds a marker for the current value
            'value': probabilities[1] * 100
        },
        'steps': [
            {'range': [0, 50], 'color': "darkseagreen"},
            {'range': [50, 75], 'color': "lightsalmon"},
            {'range': [75, 90], 'color': "orangered"},
            {'range': [90, 100], 'color': "firebrick"}
        ]
    }
))

    fig.update_layout(
        width=500, height=420, 
        paper_bgcolor="snow",  # Background color outside the gauge
        plot_bgcolor="#e0f7fa"     # Background inside the gauge
    )

    st.plotly_chart(fig, use_container_width=True)

# Create SHAP waterfall plot and display in Streamlit
def plot_shap_waterfall(shap_values, language):
    # Create SHAP waterfall plot and force it to use a different colormap
    fig, ax = plt.subplots()  # Create a figure

    # Plot Waterfall Plot
    shap.plots.waterfall(shap_values[0], show = False)

    plt.title(translations[language]["shap_waterfall_title"]) 
    plt.xlabel(translations[language]["waterfall_x_label"])
    plt.xlim([-12,12])

    # Generate the waterfall plot and override SHAP's color settings
    # Default SHAP colors
    default_pos_color = "#ff0051"
    default_neg_color = "#008bfb"
    # Custom colors
    positive_color = "darkseagreen"
    negative_color = "lightsalmon"

    # Change the colormap of the artists
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if (isinstance(fcc, mplot.patches.FancyArrow)):
                if (mplot.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                    fcc.set_facecolor(positive_color)
                elif (mplot.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                    fcc.set_color(negative_color)
            elif (isinstance(fcc, plt.Text)):
                if (mplot.colors.to_hex(fcc.get_color()) == default_pos_color):
                    fcc.set_color(positive_color)
                elif (mplot.colors.to_hex(fcc.get_color()) == default_neg_color):
                    fcc.set_color(negative_color)
    
    st.pyplot(fig)  # Display in Streamlit

def st_shap(plot, height=130, width="90%", bgcolor="#f8f9fa"):
    """Render SHAP force plot in Streamlit with customization."""
    shap_html = f"""
    <head>{shap.getjs()}</head>
    <body style="display: flex; justify-content: center; background-color: {bgcolor};">
        <div style="width: {width};">
            {plot.html()}
        </div>
    </body>
    """
    components.html(shap_html, height=height)


def group_shap_features(shap_values, grouped_features):
    """
    Groups SHAP values for one-hot encoded features by summing their contributions.

    Parameters:
    - shap_values: shap.Explanation
        The original SHAP values object.
    - grouped_features: dict
        Dictionary where keys are new feature names and values are lists of original feature names to group.

    Returns:
    - shap.Explanation
        A new SHAP object with grouped features.
    """
    # Extract feature names from SHAP values
    original_feature_names = shap_values.feature_names

    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=original_feature_names)

    # Create a new DataFrame for grouped features
    grouped_shap_df = pd.DataFrame()

    # Aggregate SHAP values by summing grouped categories
    for new_feature, old_features in grouped_features.items():
        grouped_shap_df[new_feature] = shap_df[old_features].sum(axis=1)

    # Identify remaining features that are not grouped
    grouped_feature_list = sum(grouped_features.values(), [])  # Flatten the list
    remaining_features = [f for f in original_feature_names if f not in grouped_feature_list]

    # Add non-grouped features back
    grouped_shap_df[remaining_features] = shap_df[remaining_features]

    # Convert back to SHAP object
    return shap.Explanation(
        values=grouped_shap_df.values,
        base_values=shap_values.base_values,
        data=shap_values.data,
        feature_names=grouped_shap_df.columns.tolist(),
    )

# Example usage
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

def plot_shap_feature_importance(shap_values, title="Global Feature Importance", figsize=(10, 6)):
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
    fig, ax = plt.subplots(figsize=figsize)

    # Generate the SHAP bar plot
    shap.plots.bar(feature_importance, show=False)

    # Customize the title and axis labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)

    # Show the plot in Streamlit
    st.pyplot(fig)

def update_prediction(input, model, grouped_features, explainer, language): # Collect user inputs dynamically
    # Predict the class and probabilities
    prediction = model.predict(input)[0]
    probas = model.predict_proba(input)[0]
    # Get probabilities
    max_index = probas.argmax()  # Find the class with highest probability
    max_prob = probas[max_index]  # Get the highest probability

    # Define class labels (ensure they match translations)
    class_labels = {
        0: translations[language]["non_depression"], 
        1: translations[language]["depression"]
    }

    # Construct the output message
    prediction_message = f"**{max_prob*100:.2f}%** *({class_labels.get(max_index, 'Unknown')})*"

    # Display prediction result dynamically
    #st.write(f"{translations[language]['predicted_class']}: **{class_labels.get(int(prediction), 'Unknown')}**")
    #st.write(f"{translations[language]['prediction_probabilities']} {prediction_message}")

    # Define color based on prediction probability (you can use percentage to determine this)
    if prediction == 1:  # Class 1 (e.g., high risk)
        color = "orangered"
    else:  # Class 0 (e.g., low risk)
        color = "chartreuse"

    # Display text with color dynamically
    st.markdown(f"""
        <p style="font-size:18px;">
            {translations[language]['predicted_class']}: 
            <strong style="color:{color};">{class_labels.get(int(prediction), 'Unknown')}</strong>
    </p>
""", unsafe_allow_html=True)

    # Plot the class probabilities as a bar chart
    #plot_class_probabilities(probas, language)
    plot_gauge(probas, language)
    
    st.write('')
    st.write('')
    st.write('### ' + translations[language]['shap'])

    # Calculate SHAP values for the test set
    shap_values = explainer(input_df)

    shap_plot = group_shap_features(shap_values, grouped_features)

    st.write(f"{translations[language]['waterfall_intro']}")
    
    plot_shap_waterfall(shap_plot, language)

    #st_shap(shap.force_plot(
    #    shap_plot.base_values, 
    #    shap_plot.values, 
    #    shap_plot.feature_names,
    #    plot_cmap="coolwarm"))# height=130)

# Preprocessing Function
def preprocess_input(data, translations, language, scaler=None):
    """
    Transforms raw user input into the required format for model prediction, including normalization.
    """
    # One-hot encoding for gender
    data['RIAGENDR_Female'] = 1 if data['gender'] == translations[language]['female'] else 0
    data['RIAGENDR_Male'] = 1 if data['gender'] == translations[language]['male'] else 0

    # One-hot encoding for marital status
    data['DMDMARTZ_2.0'] = 1 if data['marital_status'] == translations[language]['Divorced'] else 0
    data['DMDMARTZ_3.0'] = 1 if data['marital_status'] == translations[language]['Never Married'] else 0

    # Map education level to numbers
    education_mapping = {
        'Less than 9th grade': 0,
        '9-11th grade': 1,
        'High school graduate/GED or equivalent': 2,
        'Some college or AA degree': 3,
        'College graduate or above': 4,
    }
    data['DMDEDUC2'] = education_mapping.get(data['education_level'])  # -1 if unknown

    # Boolean conversion for disabilities
    data['FNDADI'] = data['disabilities']

    # Selecting only relevant columns
    input_features = ['RIDAGEYR', 'RIAGENDR_Female', 'RIAGENDR_Male', 'DMDMARTZ_2.0', 'DMDMARTZ_3.0', 'DMDEDUC2',
                      'DMDHHSIZ', 'RXQ050', 'SLD012', 'ALQ130', 'FNDADI', 'PAD680']

    # Convert to DataFrame to ensure the proper format
    df = pd.DataFrame(scaler.transform(pd.DataFrame([data], columns=input_features)), 
                  columns=input_features, dtype=float)

    return df

def generate_lime_explanation(input_data, model, class_names=['non_depression', 'depression'], num_features=10):
    """
    This function takes in the model and new data input (DataFrame) and generates the LIME explanation.
    It displays the explanation and visualizes it using Streamlit.
    """
    # Ensure input data is in the correct format (Pandas DataFrame)
    if isinstance(input_data, pd.DataFrame):
        # Handle missing values by imputing with mean (for numerical data)
        input_data = input_data.fillna(input_data.mean())
        input_data_values = input_data.values
    else:
        raise ValueError("The input data must be a Pandas DataFrame.")

    # Initialize the LIME explainer with user input data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=input_data_values,  # Using DataFrame values
        feature_names=input_data.columns,  # Feature names from the DataFrame
        class_names=class_names,  # Class labels
        mode='classification'  # Assuming a classification model
    )

    # Generate the explanation for the first instance in the input data
    explanation = explainer.explain_instance(input_data_values[0], model.predict_proba, num_features=num_features)

    # Display the explanation as a list
    st.write("### LIME Explanation:")
    st.write(explanation.as_list())

    # Plot the explanation using LIME's internal plotter and display it with Streamlit
    fig = explanation.as_pyplot_figure()
    
    # Display the plot using st.pyplot() to show in the Streamlit app
    st.pyplot(fig)

### SIDE BAR SETTINGS ###
# User input section
st.sidebar.markdown("## " + translations[language]["sidebar_header"])

user_input = {} # Initialize user input dictionary
user_choice = {} # Initialize user choice for user feedback

# Age Input (Number Input)
st.sidebar.markdown("### " + "1. " + translations[language]["age"])
#dont_know_label_age = translations[language]["dont_know"]
#age_dontknow = st.sidebar.checkbox(dont_know_label_age, key="age_dontknow", value=True) # Checkbox for "Don't know"
age = st.sidebar.slider(label='', min_value=18, max_value=100, value=50, key="age", #disabled=age_dontknow, # Number input for age
    help=translations[language]["age_help"])

user_input['RIDAGEYR'] = age # np.nan if age_dontknow else age # Store user input (keeping None as the standard representation)
user_choice[translations[language]["age"]] = age #translations[language]["dont_know"] if age_dontknow else age

# Gender Input (Radio)
st.sidebar.markdown("### " + "2. " + translations[language]["gender"])
#dont_know_label_gender = translations[language]["dont_know"]
gender_map = { # Define translation mappings
    translations[language]["male"]: "Male",
    translations[language]["female"]: "Female"}

gender_options = list(gender_map.keys()) # Translated options
#gender_dontknow = st.sidebar.checkbox(dont_know_label_gender, key="gender_dontknow", value=True)
selected_gender = st.sidebar.radio(label='', options=gender_options, key="gender", index=1, #disabled=gender_dontknow,
                                   help=translations[language]["gender_help"])

user_input['gender'] = selected_gender #np.nan if gender_dontknow else selected_gender
user_choice[translations[language]["gender"]] = selected_gender #translations[language]["dont_know"] if gender_dontknow else selected_gender

# Marital Status Input (Dropdown)
st.sidebar.markdown("### " + "3. " + translations[language]["marital_status"])
dont_know_label_marital = translations[language]["dont_know"]
marital_dontknow = st.sidebar.checkbox(dont_know_label_marital, key="marital_dontknow", value=True)
marital_map= {
    translations[language]["Married"]: "Married",
    translations[language]["Divorced"]: "Divorced",
    translations[language]["Never Married"]: "Never Married"}
marital_options = list(marital_map.keys())
marital_status = st.sidebar.radio(label="",options=marital_options, 
        key="marital_status", 
        disabled=marital_dontknow,
        help=translations[language]["marital_help"])
user_input['marital_status'] = np.nan if marital_dontknow else marital_status
user_choice[translations[language]["marital_status"]] =  translations[language]["dont_know"] if marital_dontknow else marital_status

# Education Level (Dropdown)
st.sidebar.markdown("### " + "4. " + translations[language]["education_level"])
dont_know_label_edu = translations[language]["dont_know"]
edu_dontknow = st.sidebar.checkbox(dont_know_label_edu, key="edu_dontknow", value=True)

edu_map = {
    translations[language]["educ_1"]: "Less than 9th grade",
    translations[language]["educ_2"]: "9-11th grade",
    translations[language]["educ_3"]: "High school graduate/GED or equivalent",
    translations[language]["educ_4"]: "Some college or AA degree",
    translations[language]["educ_5"]: "College graduate or above"
}

edu_options = list(edu_map.keys())
education_level = st.sidebar.selectbox(label='', options=edu_options, key="education", disabled=edu_dontknow,
                                       help=translations[language]["educ_help"])

user_input['education_level'] = np.nan if edu_dontknow else edu_map[education_level]
user_choice[translations[language]["education_level"]] = translations[language]["dont_know"] if edu_dontknow else education_level

# Household Size (Slider)
st.sidebar.markdown("### " + "5. " + translations[language]["household_size"])
dont_know_label_household = translations[language]["dont_know"]
household_dontknow = st.sidebar.checkbox(dont_know_label_household, key="household_dontknow", value=True)
household_size = st.sidebar.slider(translations[language]["household_slider"], 1, 7, 3, 
                                    key="household_size", 
                                    disabled=household_dontknow,
                                    help=translations[language]["household_help"])
user_input['DMDHHSIZ'] = np.nan if household_dontknow else household_size
user_choice[translations[language]["household_size"]] =  translations[language]["dont_know"] if household_dontknow else household_size

# Medication Use (Slider)
st.sidebar.markdown("### " + "6. " + translations[language]["medication_use"])
dont_know_label_medic = translations[language]["dont_know"]
medication_dontknow = st.sidebar.checkbox(dont_know_label_medic, key="medication_dontknow", value=True)
medication_use = st.sidebar.slider(translations[language]["medication_slider"], 0, 5, 0, 
                                    key="medication_use", 
                                    disabled=medication_dontknow,
                                    help=translations[language]["medication_help"])
user_input['RXQ050'] = np.nan if medication_dontknow else medication_use
user_choice[translations[language]["medication_use"]] =  translations[language]["dont_know"] if medication_dontknow else medication_use

# Sedentarism (Number Input)
st.sidebar.markdown("### " + "7. " + translations[language]["sedentarism"])
dont_know_label_sedent = translations[language]["dont_know"]
sedent_dontknow = st.sidebar.checkbox(dont_know_label_sedent, key="sedent_dontknow", value=True) # Checkbox for "Don't know"
sedentarism = st.sidebar.slider(translations[language]["sedentarism_input"], min_value=0.0, max_value=23.0, value=4.0, step=0.5, key="sedentarism", disabled=sedent_dontknow, # Number input for age
    help=translations[language]["sedentarism_help"])

user_input['PAD680'] = np.nan if sedent_dontknow else (sedentarism*60) # Store user input (keeping None as the standard representation)
user_choice[translations[language]["sedentarism"]] = translations[language]["dont_know"] if sedent_dontknow else sedentarism

# Sleep Hours (Slider)
st.sidebar.markdown("### " + "8. " + translations[language]["sleep_hours"])
dont_know_label_sleep = translations[language]["dont_know"]
sleep_dontknow = st.sidebar.checkbox(dont_know_label_sleep, key="sleep_dontknow", value=True)
sleep_hours = st.sidebar.slider(translations[language]["sleep_slider"], 2.0, 14.0, 7.0, 0.5, key="sleep", disabled=sleep_dontknow,
                                help=translations[language]["sleep_help"])
user_input['SLD012'] = np.nan if sleep_dontknow else sleep_hours
user_choice[translations[language]["sleep_hours"]] =  translations[language]["dont_know"] if sleep_dontknow else sleep_hours

# Drinking Frequency (Slider)
st.sidebar.markdown("### " + "9. " + translations[language]["drinking_frequency"])
dont_know_label_drink = translations[language]["dont_know"]
drink_dontknow = st.sidebar.checkbox(dont_know_label_drink, key="drink_dontknow", value=True)
drinking_frequency = st.sidebar.slider(translations[language]["drinking_slider"], 0, 15, 5, key="drink", disabled=drink_dontknow,
                                       help=translations[language]["drinking_help"])
user_input['ALQ130'] = np.nan if drink_dontknow else drinking_frequency
user_choice[translations[language]["drinking_frequency"]] =  translations[language]["dont_know"] if drink_dontknow else drinking_frequency

# Disabilities (Radio)
st.sidebar.markdown("### " + "10. " + translations[language]["disabilities"])
dont_know_label_disab = translations[language]["dont_know"]
disab_dontknow = st.sidebar.checkbox(dont_know_label_disab, key="disab_dontknow", value=True)
disab_map = {
    translations[language]["Yes"]: 1,  # Yes -> 1
    translations[language]["No"]: 0    # No -> 0
}
disab_options = list(disab_map.keys())
disabilities = st.sidebar.radio(label='', options=disab_options, key="disabilities", disabled=disab_dontknow,
                                help=translations[language]["disab_help"])
user_input['disabilities'] = np.nan if disab_dontknow else disab_map[disabilities]
user_choice[translations[language]['disabilities']] =  translations[language]["dont_know"] if disab_dontknow else disabilities

# Mapping links for different languages
link_map = {
    "English": "https://www.washingtongroup-disability.com/fileadmin/uploads/wg/Documents/Questions/Washington_Group_Questionnaire__1_-_WG_Short_Set_on_Functioning__June_2022_.pdf",
    "Español": "https://www.washingtongroup-disability.com/fileadmin/uploads/wg/Documents/WG-Short-Set-Spanish-translation-v2020-June-23.pdf",
    "Português brasileiro": "https://www.washingtongroup-disability.com/fileadmin/uploads/wg/Documents/WG-Short-Set-Brazilian-Portuguese-translation-v2020-June-23.pdf"
}

# Get the correct link based on the selected language
link = link_map.get(language, link_map["English"])  # Default to English if the language is not found

# Display clickable word "More info" that links to the corresponding URL
st.sidebar.markdown(
    f"{translations[language]['more_info']} [{translations[language]['link']}]({link})", 
    unsafe_allow_html=True)

### MAIN CONTENT ###
# Streamlit app title
col1, col2 = st.columns([1, 6])  # Adjust the ratio as needed

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
    st.page_link("explanation.py", label=translations[language]['explain_page'], icon=":material/help:")
with col5:
    st.page_link("MoreInfo.py", label=translations[language]['more_info_page'], icon=":material/info:")
with col6:
    st.page_link("Survey.py", label=translations[language]['survey_page'], icon=":material/edit:")

st.info(translations[language]["default_info"], icon=":material/info:")
# Display collected data
st.write("## " + translations[language]["user_review"])
user_choice_table = pd.DataFrame(
        user_choice.items(),
        columns=[translations[language]["feature"], translations[language]["user_selection"]],
        index=np.arange(1, len(user_choice) + 1))  # Sets index starting from 1
st.table(user_choice_table)

# Convert user input into a DataFrame
input_df = preprocess_input(user_input, translations, language, scaler=scaler)

# Extract SHAP VALUES
shap_values = explainer(input_df)

if input_df.isna().any().any():  # Checks for NaN values across the entire DataFrame
    st.warning(translations[language]['check_table_fail'])
else:
    st.info(translations[language]['check_table_ok'])  # Show a warning

#st.write("### User Input Data for ML")
#st.write(input_df)

# Automatically update the prediction when inputs change
st.write('## ' + translations[language]["prediction_result"])

#generate_lime_explanation(input_df, app_model)

update_prediction(input_df, app_model, grouped_features, explainer=explainer, language=language)

st.info(translations[language]["shap_help"], icon=":material/info:")