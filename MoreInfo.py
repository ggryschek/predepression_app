import streamlit as st

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
        .stSidebar h4 { font-size: 1.1rem; color: #444444; }
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
        "more_info_page":"About the Model",
        "survey_page":"Survey",
        "sidebar_header": "Inform Patients Data",
        "page1_title": "Depression Prediction",
        "page2_title": "About the Model",
        "model_intro":'Light Gradient Boosting Machine (LGBM) - How it works',
        'tree_plot_title':'Simple Decision Tree',
        'before_lgbm':'Let’s understand a little better how decision trees work!',
        "lgbm_text1":"LGBM is a super-fast and powerful machine learning algorithm that helps predict things based on patterns in data and it is based on decision tree algorithms",
        "lgbm_text2":"A ",
        "lgbm_text3":"LGBM is a type of gradient boosting algorithm, which means:",
        "lgbm_text4": "1. It builds multiple decision trees step by step.",
        "lgbm_text5": "2. Each new tree corrects mistakes made by the previous trees.",
        "lgbm_text6": "3. It keeps improving until it makes the best possible predictions.",
        "lgbm_text7":"4. With the trained model, the result of this process, all new data will pass through these decision trees to reach the final classification.",
        "decision_tree_text1":"A decision tree for classification works like a flowchart that helps decide the class of an input step by step.",
        "decision_tree_text2a":"Start at the root node",
        "decision_tree_text2b":"The tree begins with a question based on a feature (e.g., 'Is the education level (DMDEDUC2) less or equal to 3.5 (number applied to a category in this feature)?').",
        "decision_tree_text3a":"Make a split",
        "decision_tree_text3b":"The data is divided based on the answer (e.g., 'True' or 'False').",
        "decision_tree_text4a":"Repeat the process",
        "decision_tree_text4b":"Each split leads to more questions until the data is classified into a final category (leaf node).",
        "decision_tree_text5a":"Final decision",
        "decision_tree_text5b":"Once you reach a leaf node, that becomes the predicted class.",
        'how_works':"How it works?",
        "dont_know":"Don't know",
        "No": "No",
        "Yes": "Yes",
        "non_depression": "Non Depression",
        "depression": "Depression",
        'male':'Male',
        'female':'Female',
        'married':'Married',
        'divorced':'Divorced',
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
        'link_intro':'About the training Data',
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
        'metric':'Metric',
        'meaning': 'Meaning',
        'math_representation': 'Mathematical Representation',
        'confusion_matrix_terminology':'Confusion Matrix Terminology',
        'matrix_explain':'To analyze how these metrics work, it is necessary to understand the terminology of the Confusion Matrix, which we use to check the errors and successes of predictions/classifications.',
        'tp': 'True Positive (TP)',
        'tn': 'True Negative (TN)',
        'fp': 'False Positive (FP)',
        'fn': 'False Negative (FN)',
        'tp_table': 'TP',
        'tn_table': 'TN',
        'fp_table': 'FP',
        'fn_table': 'FN',
        'tp_desc': 'Model predicted **positive**, and the actual value was **positive**',
        'tn_desc': 'Model predicted **negative**, and the actual value was **negative**',
        'fp_desc': 'Model predicted **positive**, but the actual value was **negative** (**Type I Error**)',
        'fn_desc': 'Model predicted **negative**, but the actual value was **positive** (**Type II Error**)',
        'explanation_title_1':'Understanding Model Performance Metrics',
        'explanation_intro':'When looking at a machine learning model’s performance, you may see terms like Accuracy, Precision, Recall, F1 Score, and AUC Score. These metrics help us understand how well the model is making predictions',
        'accuracy_title':'Accuracy – “How often is the model right?”',
        'accuracy_explain':'Accuracy tells us the percentage of total predictions that were correct. If the model has 90% accuracy, it means that out of 100 cases, 90 were correctly classified',
        'how_interpret':'How to interpret it',
        'what_mean':'What it means',
        'metrics_summary':'Metrics Summary',
        'high_accuracy':'High accuracy (close to 100%) → The model is generally making correct predictions',
        'low_accuracy':'Low accuracy (closer to 50% or lower) → The model is making a lot of mistakes',
        'precision_title':'Precision – “When the model predicts positive, how often is it correct?”',
        'precision_explain':'Precision is about avoiding false positives (wrong positive predictions). If a model predicts someone has a disease, precision tells us how often that prediction is actually correct',
        'high_precision':'High precision (close to 100%) → The model rarely predicts false positives, meaning when it predicts something as positive, it is almost always correct.',
        'low_precision':'Low precision → The model makes many false positive predictions, meaning it often wrongly identifies negative cases as positive.',
        'recall_title':'Recall - "When the condition is actually positive, how often does the model correctly identify it?"',
        'recall_explain':'Recall is about avoiding false negatives (wrong negative predictions). If a model predicts someone does not have a disease, recall tells us how often that prediction is actually correct, focusing on capturing all true positives.',
        'high_recall':'High recall (close to 100%) → The model correctly identifies almost all positive cases, missing very few actual positives.',
        'low_recall':'Low recall → The model misses many actual positive cases, falsely classifying them as negative.',
        'f1_title':'F1 Score - "How well does the model balance correctly identifying positive cases and avoiding false positives?"',
        'f1_explain':'F1 score combines precision and recall into a single metric, providing a balance between the two. It is particularly useful when you need to prioritize both avoiding false positives and capturing all true positives.',
        'high_f1':'High F1 score → The model has a good balance between precision and recall, meaning it’s both accurate when predicting positives and not missing too many actual positives.',
        'low_f1':'Low F1 score → The model performs poorly, either due to too many false positives (low precision) or missing many true positives (low recall).',
        'AUC_title':'Area Under the Curve - "How well can the model distinguish between positive and negative cases across all possible thresholds?"',
        'AUC_explain':'AUC measures the overall ability of the model to distinguish between positive and negative cases. It reflects the model’s capability to rank predictions from most likely to least likely to be positive.',
        'high_AUC':'High AUC (close to 1) → The model is very good at distinguishing between positive and negative cases across different thresholds.',
        'low_AUC':'Low AUC (close to 0.5 or lower) → The model is poor at distinguishing between positive and negative cases, behaving almost like random guessing.',

    },
    "Español": {
        "app_name": "DepreScan",
        "app_title": "Detección de depresión en atención primaria",
        "home_page":"Início",
        "depre_page":"DepreScan",
        "explain_page":"Explicaciones",
        "more_info_page":"Sobre el Modelo",
        "survey_page":"Investigación",
        "sidebar_header": "Introducir datos del paciente",
        "page1_title": "Predicción de la depresión",
        "page2_title": "Sobre el Modelo",
        "model_intro":'Light Gradient Boosting Machine (LGBM) - Cómo funciona',
        'tree_plot_title':'Árbol de decisión simple',
        'before_lgbm':'¡Entendamos un poco mejor cómo funcionan los árboles de decisión!',
        "lgbm_text1":"LGBM es un algoritmo de aprendizaje automático súper rápido y poderoso que ayuda a predecir cosas basándose en patrones en datos y se basa en algoritmos de árboles de decisión.",
        "lgbm_text2":"A ",
        "lgbm_text3":"LGBM es un tipo de algoritmo de potenciación de gradiente, lo que significa: ",
        "lgbm_text4": "1. Construye múltiples árboles de decisión paso a paso.",
        "lgbm_text5": "2. Cada nuevo árbol corrige los errores de los árboles anteriores.",
        "lgbm_text6": "3. Continúa mejorando hasta obtener las mejores predicciones posibles.",
        "lgbm_text7":"4. Con el modelo entrenado, resultado de este proceso, todos los datos nuevos pasarán por estos árboles de decisión para llegar a la clasificación final.",
        "decision_tree_text1":"Un árbol de decisión para la clasificación funciona como un diagrama de flujo que ayuda a decidir la clase de una entrada paso a paso.",
        "decision_tree_text2a":"Comenzar en el nodo raíz",
        "decision_tree_text2b":"El árbol comienza con una pregunta basada en una característica (p. ej., '¿El nivel educativo (DMDEDUC2) es menor o igual a 3.5 (número aplicado a una categoría en esta característica)?').",
        "decision_tree_text3a":"Divide",
        "decision_tree_text3b":"Los datos se dividen según la respuesta (p. ej., 'Verdad' o 'Falso').",
        "decision_tree_text4a":"Repite el proceso",
        "decision_tree_text4b":"Cada división genera más preguntas hasta que los datos se clasifican en una categoría final (nodo hoja).",
        "decision_tree_text5a":"Decisión final",
        "decision_tree_text5b":"Una vez que se llega a un nodo hoja, este se convierte en la clase predicha.",
        'how_works':"Cómo funciona?",
        "dont_know":"No sé",
        "No": "No",
        "Yes": "Sin",
        "non_depression": "Non Depression",
        "depression": "Depression",
        'male':'Male',
        'female':'Female',
        'married':'Married',
        'divorced':'Divorced',
        'gender':'Biological Sex',
        'marital_status':'Marital Status',
        'age':'Edad',
        'education_level':'Nivel de Estudos',
        'household_size':'Household Size',
        'medication_use':'Medication Use',
        'sleep_hours':'Sleep Hours',
        'drinking_frequency':'Alcohol Use',
        'disabilities':'Disabilities',
        'sedentarism':'Rest Time',
        'link_intro':'Acerca de los datos de entrenamiento',
        "link_1":'Consulte la fuente del conjunto de datos para entrenar este modelo en NATIONAL CENTER FOR HEALTH STATISTICS/Center for Control and Prevention of Diseases (CDC): ',
        'link_2':'National Health and Nutrition Examination Survey (NHANES 21-23)',
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
        'term': 'Termo',
        'metric':'Métrica',
        'meaning': 'Significado',
        'math_representation': 'Representación matemática',
        'confusion_matrix_terminology':'Terminología de la matriz de confusión',
        'matrix_explain':'Para analizar cómo funcionan esas métricas es preciso comprender la terminología de la matriz de confusión, que usamos para comprobar los errores y acertar las predicciones/clasificaciones.',
        'tp': 'Verdadero Positivo (VP)',
        'tn': 'Verdedero Negativo (VN)',
        'fp': 'Falso Positivo (FP)',
        'fn': 'Falso Negativo (FN)',
        'tp_table': 'VP',
        'tn_table': 'VN',
        'fp_table': 'FP',
        'fn_table': 'FN',
        'tp_desc': 'Modelo predijo **positivo**, y el valor real fue **positivo**',
        'tn_desc': 'Modelo predijo **negativo** y el valor real fue **negativo**.',
        'fp_desc': 'Modelo predijo **positivo**, pero el valor real fue **negativo** (**Error de tipo I**)',
        'fn_desc': 'Modelo predijo **negativo**, pero el valor real fue **positivo** (**Error de tipo II**)',
        'how_interpret':'Cómo interpretarlo',
        'metrics_summary':'Resumen de métricas',
        'what_mean':'Qué significa',
        'explanation_title_1':'Comprensión de las métricas de rendimiento del modelo',
        'explanation_intro':'Al observar el rendimiento de un modelo de aprendizaje automático, es posible que vea términos como exactitud, precisión, recuperación, puntuación F1 y puntuación AUC. Estas métricas nos ayudan a comprender qué tan bien realiza predicciones el modelo',
        'accuracy_title':'Acurácia - “¿Con qué frecuencia el modelo es correcto?”',
        'accuracy_explain':'La acurácia nos indica el porcentaje del total de predicciones correctas. Si el modelo tiene una precisión del 90 %, significa que de 100 casos, 90 se clasificaron correctamente.',
        'high_accuracy':'Acurácia alta (cercana al 100 %) → El modelo generalmente realiza predicciones correctas.',
        'low_accuracy':'Acurácia baja (cercana al 50 % o inferior) → El modelo comete muchos errores.',
        'precision_title':'Precisión: «Cuando el modelo predice un resultado positivo, ¿con qué frecuencia acierta?»',
        'precision_explain':'La precisión consiste en evitar falsos positivos (predicciones positivas erróneas). Si un modelo predice que alguien tiene una enfermedad, la precisión nos dice con qué frecuencia esa predicción es realmente correcta.',
        'high_precision':'Alta precisión (cercana al 100%) → El modelo rara vez predice falsos positivos, lo que significa que cuando predice algo como positivo, casi siempre acierta.',
        'low_precision':'Baja precisión → El modelo realiza muchas predicciones de falsos positivos, lo que significa que a menudo identifica erróneamente los casos negativos como positivos.',
        'recall_title':'Recall - "Cuando la condición es realmente positiva, ¿con qué frecuencia la identifica correctamente el modelo?"',
        'recall_explain':'Recall se trata de evitar falsos negativos (predicciones negativas erróneas). Si un modelo predice que alguien no tiene una enfermedad, la recall nos indica con qué frecuencia esa predicción es realmente correcta, centrándose en capturar todos los verdaderos positivos.',
        'high_recall':'Alta recall (cercana al 100%) → El modelo identifica correctamente casi todos los casos positivos, pasando por alto muy pocos positivos reales.',
        'low_recall':'Baja recall → El modelo pasa por alto muchos casos positivos reales, clasificándolos erróneamente como negativos.',
        'f1_title':'F1 Score - "¿Qué tan bien equilibra el modelo la correcta identificación de casos positivos y la evitación de falsos positivos?"',
        'f1_explain':'La puntuación F1 combina precisión y recall en una sola métrica, proporcionando un equilibrio entre ambas. Es particularmente útil cuando se necesita priorizar tanto la prevención de falsos positivos como la captura de todos los verdaderos positivos.',
        'high_f1':'Alta F1 score → El modelo tiene un buen equilibrio entre precisión y recall, lo que significa que es preciso al predecir positivos y no pasa por alto demasiados positivos reales.',
        'low_f1':'Baja F1 score → El modelo tiene un rendimiento deficiente, ya sea por demasiados falsos positivos (baja precisión) o por la omisión de muchos verdaderos positivos (baja recall).',
        'AUC_title':'Area Under the Curve - "¿Qué tan bien puede el modelo distinguir entre casos positivos y negativos en todos los umbrales posibles?"',
        'AUC_explain':'AUC mide la capacidad general del modelo para distinguir entre casos positivos y negativos. Refleja la capacidad del modelo para clasificar las predicciones de mayor a menor probabilidad de ser positivas.',
        'high_AUC':'Alta AUC (close to 1) → El modelo es muy bueno para distinguir entre casos positivos y negativos en diferentes umbrales.',
        'low_AUC':'Baja AUC (close to 0.5 or lower) → El modelo es deficiente para distinguir entre casos positivos y negativos, comportándose casi como una suposición aleatoria.',
    },
    "Português Br": {
        "app_name": "DepreScan",
        "app_title": "Rastreio de depressão na Atenção Primária",
        "home_page":"Início",
        "depre_page":"DepreScan",
        "explain_page":"Explicações",
        "more_info_page":"Sobre o Modelo",
        "survey_page":"Pesquisa",
        "sidebar_header": "Informe os dados do paciente",
        "page1_title": "Predição de Depressão",
        "page2_title": "Sobre o Modelo",
        "model_intro":'Light Gradient Boosting Machine (LGBM) - Como funciona',
        'tree_plot_title':'Árvore de Decisão Simples',
        'before_lgbm':'Vamos entender um pouco melhor como as árvores de decisão funcionam!',
        "lgbm_text1":"LGBM é um algoritmo de machine learning super rápido e poderoso que ajuda a prever coisas analisando os padrões de dados e é baseado em algoritmos de árvore de decisão",
        "lgbm_text2":"A ",
        "lgbm_text3":"LGBM é um tipo de algoritmo de aumento de gradiente, o que significa: ",
        "lgbm_text4": "1. Ele constrói várias árvores de decisão passo a passo.",
        "lgbm_text5": "2. Cada nova árvore corrige erros cometidos pelas árvores anteriores.",
        "lgbm_text6": "3. Ele continua melhorando até fazer as melhores previsões possíveis.",
        "lgbm_text7":"4. Com o modelo treinado, resultado desse processo, todo novo dado passará por essas árvores de decisão para chegar na classificação final",
        "decision_tree_text1":"Uma árvore de decisão para classificação funciona como um fluxograma que ajuda a decidir a classe de um novo conjunto de dados passo a passo.",
        "decision_tree_text2a":"Comece no nó raiz",
        "decision_tree_text2b":"A árvore começa com uma pergunta baseada em uma característica (por exemplo, 'O nível de educação (DMDEDUC2) é menor ou igual a 3,5 (número que representa uma categoria desta variável)?'",
        "decision_tree_text3a":"Faça uma divisão",
        "decision_tree_text3b":"Os dados são divididos com base na resposta (por exemplo, 'Verdadeiro' ou 'Falso').",
        "decision_tree_text4a":"Repita o processo",
        "decision_tree_text4b":"Cada divisão leva a mais perguntas até que os dados sejam classificados em uma categoria final (nó folha).",
        "decision_tree_text5a":"Decisão final",
        "decision_tree_text5b":"Quando você alcança um nó folha (final), ele se torna a classe prevista.",
        'how_works':"Como funciona?",
        "dont_know":"Não sei",
        "No": "Não",
        "Yes": "Sim",
        "non_depression": "Não Depressão",
        "depression": "Depressão",
        'male':'Masculino',
        'female':'Feminino',
        'married':'Casado',
        'divorced':'Divorciado',
        'gender':'Sexo Biológico',
        'marital_status':'Estado Civil',
        'age':'Idade',
        'education_level':'Education Level',
        'household_size':'Household Size',
        'medication_use':'Medication Use',
        'sleep_hours':'Sleep Hours',
        'drinking_frequency':'Alcohol Use',
        'disabilities':'Disabilities',
        'sedentarism':'Rest Time',
        'link_intro':'Sobre os Dados de treinamento',
        "link_1":'Consulte a fonte do conjunto de datos usados para treinar este modelo de Machine Learning em NATIONAL CENTER FOR HEALTH STATISTICS/Center for Control and Prevention of Diseases (CDC): ',
        'link_2':'National Health and Nutrition Examination Survey (NHANES 21-23)',
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
        'auc':'AUC - Área sob a Curva',
        'average_AUC_CV':'AUC (Treino)',
        'auc_table_explain': 'Varia de acordo com método de cálculo',
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
        'metric':'Métrica',
        'meaning': 'Significado',
        'math_representation': 'Representação Matemática',
        'confusion_matrix_terminology':'Terminologia da Matriz de Confusão',
        'matrix_explain':'Para analisar como essas métricas funcionam é preciso compreender a terminologia da Matriz de Confusão, que usamos para checar os erros e acertos das previsões/classificações',
        'tp': 'Verdadeiro Positivo (VP)',
        'tn': 'Verdadeiro Negativo (VN)',
        'fp': 'Falso Positivo (FP)',
        'fn': 'Falso Negativo (FN)',
        'tp_table': 'VP',
        'tn_table': 'VN',
        'fp_table': 'FP',
        'fn_table': 'FN',
        'tp_desc': 'Modelo previu **positivo**, e o valor de fato era **positivo**',
        'tn_desc': 'Modelo previu **negativo**, e o valor de fato era **negativo**',
        'fp_desc': 'Modelo previu **positivo**, mas o valor de fato era **negativo** (**Erro Tipo I**)',
        'fn_desc': 'Modelo previu **negativo**, mas o valor de fato era **positivo** (**Erro Tipo II**)',
        'metrics_summary':'Resumo das Medidas de Performance',
        'what_mean':'O que significa',
        'explanation_title_1':'Entendendo as métricas de performance dos modelos de Machine Learning',
        'explanation_intro':'Ao observar o desempenho de um modelo de machine learning, você pode ver termos como Acurácia, Precisão, Recall, F1 Score e AUC Score. Essas métricas nos ajudam a entender o quão bem o modelo está fazendo previsões',
        'accuracy_title':'Acurácia - “Com que frequência o modelo está certo?”',
        'accuracy_explain':'A acurácia nos diz a porcentagem do total de previsões que estavam corretas. Se o modelo tem 90% de precisão, significa que de 100 casos, 90 foram classificados corretamente',
        'high_accuracy':'Alta acurácia (próximo a 100%) → O modelo geralmente está fazendo previsões corretas',
        'low_accuracy':'Baixa acurácia (próximo a 50% ou menos) → O modelo está cometendo muitos erros',
        'precision_title':'Precisão – "Quando o modelo prevê positivo, com que frequência ele está correto?"',
        'precision_explain':'A precisão é sobre evitar falsos positivos (previsões positivas erradas). Se um modelo prevê que alguém tem uma doença, a precisão nos diz com que frequência essa previsão está realmente correta',
        'high_precision':'Alta precisão (próximo a 100%) → O modelo raramente prevê falsos positivos, o que significa que quando ele prevê algo como positivo, ele quase sempre está correto.',
        'low_precision':'Baixa precisão → O modelo faz muitas previsões de falsos positivos, o que significa que ele frequentemente identifica erroneamente casos negativos como positivos.',
        'recall_title':'Recall - "Quando a condição é realmente positiva, com que frequência o modelo a identifica corretamente?"',
        'recall_explain':'Recall é sobre evitar falsos negativos (previsões negativas erradas). Se um modelo prevê que alguém não tem uma doença, a recordação nos diz com que frequência essa previsão está realmente correta, focando em capturar todos os verdadeiros positivos.',
        'high_recall':'Alto recall (perto de 100%) → O modelo identifica corretamente quase todos os casos positivos, perdendo muito poucos positivos reais.',
        'low_recall':'Baixo recall → O modelo perde muitos casos positivos reais, classificando-os falsamente como negativos.',
        'f1_title':'F1 Score - "Quão bem o modelo equilibra corretamente a identificação de casos positivos e a prevenção de falsos positivos?"',
        'f1_explain':'A pontuação F1 combina precisão e recordação em uma única métrica, fornecendo um equilíbrio entre as duas. É particularmente útil quando você precisa priorizar evitar falsos positivos e capturar todos os verdadeiros positivos.',
        'high_f1':'Alto F1 score → O modelo tem um bom equilíbrio entre precisão e recordação, o que significa que é preciso ao prever positivos e não perde muitos positivos reais.',
        'low_f1':'Baixo F1 score → O modelo tem um desempenho ruim, seja devido a muitos falsos positivos (baixa precisão) ou à falta de muitos verdadeiros positivos (baixa recuperação).',
        'AUC_title':'Area Under the Curve - "Quão bem o modelo consegue distinguir entre casos positivos e negativos em todos os limites possíveis?"',
        'AUC_explain':'AUC mede a capacidade geral do modelo de distinguir entre casos positivos e negativos. Ele reflete a capacidade do modelo de classificar as previsões da mais provável para a menos provável de serem positivas.',
        'high_AUC':'Alta AUC (close to 1) → O modelo é muito bom em distinguir entre casos positivos e negativos em diferentes limites.',
        'low_AUC':'Baixa AUC (close to 0.5 or lower) → O modelo é ruim em distinguir entre casos positivos e negativos, comportando-se quase como um palpite aleatório.',
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

def plot_custom_decision_tree(model, language, translations):
    """
    Function to plot a customized decision tree in Streamlit.

    Parameters:
    - model: Trained decision tree model (e.g., DecisionTreeClassifier).
    - language: Language key for translation.
    - translations: Dictionary containing translated feature names.
    """
    
    # Original feature names
    feature_names = [
        'RIDAGEYR', 'RIAGENDR_Female', 'RIAGENDR_Male', 'DMDMARTZ_2.0',
        'DMDMARTZ_3.0', 'DMDEDUC2', 'DMDHHSIZ', 'RXQ050', 'SLD012', 
        'ALQ130', 'FNDADI', 'PAD680'
    ]

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Class names translation
    class_names = [translations[language]['non_depression'], translations[language]['depression']]
    
    # Plot the decision tree
    plot_tree(
        model,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=class_names,
        fontsize=12,
        precision=2,
        label='all',
        proportion=True,
        node_ids=True,
        impurity=False
    )

    # Set title with custom size
    plt.title(translations[language]['tree_plot_title'], fontsize=16, fontweight="bold")

    # Improve layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# Create a SHAP explainer object from the model
explainer = shap.Explainer(app_model)

features_names = ['RIDAGEYR', 'RIAGENDR_Female', 'RIAGENDR_Male', 'DMDMARTZ_2.0',
       'DMDMARTZ_3.0', 'DMDEDUC2', 'DMDHHSIZ', 'RXQ050', 'SLD012', 'ALQ130',
       'FNDADI', 'PAD680']

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
    st.page_link("explanation.py", label=translations[language]['explain_page'], icon=":material/help:")
with col6:
    st.page_link("Survey.py", label=translations[language]['survey_page'], icon=":material/edit:")

st.write("# " + translations[language]['more_info_page'])

st.write("## " + translations[language]['model_intro'])

st.info(translations[language]['lgbm_text1'], icon=":material/info:")

st.write("### " + translations[language]['before_lgbm'])

st.write(translations[language]['decision_tree_text1'])

col7, col8 = st.columns([1,4])
with col7:
    st.write("#### " + translations[language]['decision_tree_text2a'])
with col8:
    st.info(translations[language]['decision_tree_text2b'])

col9, col10 = st.columns([1,4])
with col9:
    st.write("#### " + translations[language]['decision_tree_text3a'])
with col10:
    st.info(translations[language]['decision_tree_text3b'])

col11, col12 = st.columns([1,4])
with col11:
    st.write("#### " + translations[language]['decision_tree_text4a'])
with col12:
    st.info(translations[language]['decision_tree_text4b'])

col13, col14 = st.columns([1,4])
with col13:
    st.write("#### " + translations[language]['decision_tree_text5a'])
with col14:
    st.info(translations[language]['decision_tree_text5b'])

plot_custom_decision_tree(dt_model, language=language, translations=translations)

st.write("#### " + translations[language]['lgbm_text3'])

col15, col16 = st.columns([1,8])
with col15:
     st.write(' ')
with col16:     
    st.info(translations[language]['lgbm_text4'])

    st.info(translations[language]['lgbm_text5'])

    st.info(translations[language]['lgbm_text6'])

    st.info(translations[language]['lgbm_text7'])

st.write("## " + translations[language]['explanation_title_1'])

st.write(translations[language]['explanation_intro'])

plot_model_results(metrics_df, language)

# Display in Streamlit
st.write(translations[language]['matrix_explain'])
st.write("#### " + translations[language]['confusion_matrix_terminology'])
st.dataframe(confusion_matrix_data.style.hide(axis="index"))

## Accuracy Section ##
st.markdown(f"<h3>{translations[language]['accuracy_title']}</h3>", unsafe_allow_html=True)

col17, col18 = st.columns([1,4])
with col17:
    st.write(translations[language]['what_mean'] + ':')
with col18:
    st.info(translations[language]['accuracy_explain'])

col19, col20, col21 = st.columns([4, 2, 4])  # Adjust the ratio as needed
with col19:
    st.write("Formula:")
with col20:
    st.write("#### " + translations[language]['test_accuracy'])
with col21:
    # Alternative using Markdown
    accuracy_formula = (
        rf"$ \frac{{{translations[language]['tp_table']} + {translations[language]['tn_table']}}}{{{translations[language]['tp_table']} + {translations[language]['tn_table']} + {translations[language]['fp_table']} + {translations[language]['fn_table']}}} $"
    )
    # Render with markdown
    st.markdown("#### " + "= " + accuracy_formula, unsafe_allow_html=True)  

col22, col23 = st.columns([1, 4])  # Adjust the ratio as needed

# Place the image in the first column
with col22:
    st.write(translations[language]['how_interpret'] + ': ')

# Place the title in the second column
with col23:
    st.info(translations[language]['high_accuracy'])

    st.info(translations[language]['low_accuracy'])

## Precision Section ##
st.markdown(f"<h3>{translations[language]['precision_title']}</h3>", unsafe_allow_html=True)

col24, col25 = st.columns([1,4])
with col24:
    st.write(translations[language]['what_mean'] + ':')
with col25:
    st.info(translations[language]['precision_explain'])

col26, col27, col28 = st.columns([4, 2, 4])  # Adjust the ratio as needed
with col26:
    st.write("Formula:")
with col27:
    st.write("#### " + translations[language]['test_precision'])
with col28:
    # Alternative using Markdown
    precision_formula = (
        rf"$ \frac{{{translations[language]['tp_table']}}}{{{translations[language]['tp_table']} + {translations[language]['fp_table']}}} $"
    )
    # Render with markdown
    st.markdown("#### " + "= " + precision_formula, unsafe_allow_html=True)  

col29, col30 = st.columns([1, 4])  # Adjust the ratio as needed

# Place the image in the first column
with col29:
    st.write(translations[language]['how_interpret'] + ': ')

# Place the title in the second column
with col30:
    st.info(translations[language]['high_precision'])

    st.info(translations[language]['low_precision'])

## Recall Section ##
st.markdown(f"<h3>{translations[language]['recall_title']}</h3>", unsafe_allow_html=True)

col31, col32 = st.columns([1,4])
with col31:
    st.write(translations[language]['what_mean'] + ':')
with col32:
    st.info(translations[language]['recall_explain'])

col33, col34, col35 = st.columns([4, 2, 4])  # Adjust the ratio as needed
with col33:
    st.write("Formula:")
with col34:
    st.write("#### " + translations[language]['test_recall'])
with col35:
    # Alternative using Markdown
    recall_formula = (
        rf"$ \frac{{{translations[language]['tp_table']}}}{{{translations[language]['tp_table']} + {translations[language]['fn_table']}}} $"
    )
    # Render with markdown
    st.markdown("#### " + "= " + recall_formula, unsafe_allow_html=True)  

col36, col37 = st.columns([1, 4])  # Adjust the ratio as needed

# Place the image in the first column
with col36:
    st.write(translations[language]['how_interpret'] + ': ')

# Place the title in the second column
with col37:
    st.info(translations[language]['high_recall'])

    st.info(translations[language]['low_recall'])

## F1 Score Section ##
st.markdown(f"<h3>{translations[language]['f1_title']}</h3>", unsafe_allow_html=True)

col38, col39 = st.columns([1,4])
with col38:
    st.write(translations[language]['what_mean'] + ':')
with col39:
    st.info(translations[language]['f1_explain'])

col40, col41, col42 = st.columns([4, 2, 4])  # Adjust the ratio as needed
with col40:
    st.write("Formula:")
with col41:
    st.write("#### " + translations[language]['test_f1'])
with col42:
    # Alternative using Markdown
    f1_formula = (
        rf"$ 2 \times \frac{{{translations[language]['test_precision']} \times {translations[language]['test_recall']}}}{{{translations[language]['test_precision']} + {translations[language]['test_recall']}}} $"
    )
    # Render with markdown
    st.markdown("#### " + "= " + f1_formula, unsafe_allow_html=True)  

col43, col44 = st.columns([1, 4])  # Adjust the ratio as needed

# Place the image in the first column
with col43:
    st.write(translations[language]['how_interpret'] + ': ')

# Place the title in the second column
with col44:
    st.info(translations[language]['high_f1'])

    st.info(translations[language]['low_f1'])

## AUC Section ##
st.markdown(f"<h3>{translations[language]['AUC_title']}</h3>", unsafe_allow_html=True)

col45, col46 = st.columns([1,4])
with col45:
    st.write(translations[language]['what_mean'] + ':')
with col46:
    st.info(translations[language]['AUC_explain'])

col47, col48 = st.columns([1,4])
with col47:    
    st.write(" ")
with col48:
    st.image("ROC_plot.png", width=550) 

col50, col51 = st.columns([1, 4])  # Adjust the ratio as needed

# Place the image in the first column
with col50:
    st.write(translations[language]['how_interpret'] + ': ')

# Place the title in the second column
with col51:
    st.info(translations[language]['high_AUC'])

    st.info(translations[language]['low_AUC'])

st.markdown(f"<h3>{translations[language]['metrics_summary']}</h3>", unsafe_allow_html=True)

# Example DataFrame
data = [
    [translations[language]['test_accuracy'], rf"$ \frac{{{translations[language]['tp_table']} + {translations[language]['tn_table']}}}{{{translations[language]['tp_table']} + {translations[language]['tn_table']} + {translations[language]['fp_table']} + {translations[language]['fn_table']}}} $"],
    [translations[language]['test_precision'], rf"$ \frac{{{translations[language]['tp_table']}}}{{{translations[language]['tp_table']} + {translations[language]['fp_table']}}} $"],
    [translations[language]['test_recall'], rf"$ \frac{{{translations[language]['tp_table']}}}{{{translations[language]['tp_table']} + {translations[language]['fn_table']}}} $"],
    [translations[language]['test_f1'], rf"$ 2 \times \frac{{{translations[language]['test_precision']} \times {translations[language]['test_recall']}}}{{{translations[language]['test_precision']} + {translations[language]['test_recall']}}} $"],
    [translations[language]['auc'], translations[language]['auc_table_explain']]
]

# Create DataFrame without an index
df = pd.DataFrame(data, columns=[translations[language]['metric'], "Formula"])

table_md = f"| {translations[language]['metric']} | Formula |\n|--------|---------|\n"
for _, row in df.iterrows():
    table_md += f"| {row[0]} | {row[1]} |\n"

col52, col53 = st.columns([1,4])
with col52:
    st.write(' ')
with col53:
    st.markdown(table_md)

st.write("## " + translations[language]['link_intro'])
# Adding links for more information
link ="https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023"
st.markdown(f"{translations[language]['link_1']}")
st.markdown(f"[{translations[language]['link_2']}]({link})")