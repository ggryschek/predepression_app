import streamlit as st

### MULTIPAGE NAVIGATION ###

st.set_page_config(page_title="Streamlit DepreScan", page_icon=":material/medical_services:")

home_page = st.Page("Home.py", title="Home / Inicio", icon=":material/home:")
pred_page = st.Page("DepressionPrediction.py", title="DepreScan", icon=":material/psychology:")
explain_page = st.Page("explanation.py", title="Explanations / Explicações", icon=":material/help:")
info_page = st.Page("MoreInfo.py", title="About the Model / Sobre o Modelo", icon=":material/info:")
survey_page = st.Page("Survey.py", title="Survey / Pesquisa", icon=":material/edit:")

pg = st.navigation([home_page, pred_page, explain_page, info_page, survey_page])

# Initialize session state keys if not present
if "my_key" not in st.session_state:
    st.session_state["my_key"] = "English"  # Default language

if "_my_key" not in st.session_state:
    st.session_state["_my_key"] = st.session_state["my_key"]

def store_value():
    # Copy the value to the permanent key
    st.session_state["my_key"] = st.session_state["_my_key"]

# Dictionary of flag paths
flag_path = {
    "English": "us_flag.png",
    "Español": "es_flag.png",
    "Português Br": "br_flag.png"
}

#Initialize session state for language if it's not already set
if "language" not in st.session_state:
        st.session_state["language"] = "English"  # Default language

col1, col2 = st.sidebar.columns([1,4])
with col1:
    st.write(" ")
    st.image("logo_app.jpg", width=50)
with col2:
    st.markdown('# ' + '**DepreScan**')
#with col3:
#    st.image("KI_logo.png")
#    st.image("SU_logo.jpg")

col4, col5 = st.columns([1,3])
width_head = 100
with col4:
    st.write(" ")
    st.image("logo_app.jpg", width=width_head)
with col5:
     st.write(" ")
     st.markdown('# ' + '**DepreScan**')

## Language Menu ##
col6, col7, col8 = st.columns([5,2,1])  # Wider space for selectbox, smaller for flag

with col6:
    # Language selection widget (for the first page or the initialization page)
    # Apply styling to the selectbox label using markdown with HTML
    # Apply styling to the language selection prompt with font and centralized text
    st.write(" ")
    st.write(" ")
    st.markdown(
        """
        <style>
        .language-selection {
            color: black;
            background-color: lightgray;  /* Example of a background color change */
            padding: 5px;
            border-radius: 4px;
            font-size: 13px;  /* Change font size */
            font-family: 'Inter', sans-serif;  /* Change font to Arial */
            text-align: center;  /* Centralize the text */
            font-weight: bold;  /* Make the text bold */
            width: 100%;  /* Make the div take up the full width */
        }
        </style>
        <div class="language-selection">
            Choose your language / Elige tu idioma / Escolha seu idioma
        </div>
        """, unsafe_allow_html=True
    )
with col7:
    language = st.selectbox('',
        #'Choose your language / Elige tu idioma / Escolha seu idioma',
        ["English", "Español", "Português Br"],
        index=["English", "Español", "Português Br"].index(st.session_state["language"]),
        key="language_select"
    )

# Update session state with the selected language
st.session_state["language"] = language

with col8:
    st.write(" ")
    st.write(" ")
    st.image(flag_path[language], width=40)  # Resize the flag

pg.run()