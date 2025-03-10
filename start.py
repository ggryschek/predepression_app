import streamlit as st

### MULTIPAGE NAVIGATION ###

st.set_page_config(page_title="DepreScan", page_icon=":material/medical_services:")

home_page = st.Page("Home.py", title="Home / Inicio", icon=":material/home:")
pred_page = st.Page("DepressionPrediction.py", title="DepreScan", icon=":material/psychology:")
explain_page = st.Page("explanation.py", title="Explanations / Explicações", icon=":material/help:")
info_page = st.Page("MoreInfo.py", title="More Information / Mais Informações", icon=":material/info:")
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
    "Português brasileiro": "br_flag.png"
}

#Initialize session state for language if it's not already set
if "language" not in st.session_state:
        st.session_state["language"] = "English"  # Default language

# Create columns within the sidebar
col3, col4 = st.sidebar.columns([4, 1])  # Wider space for selectbox, smaller for flag

with col3:
    # Language selection widget (for the first page or the initialization page)
    language = st.selectbox(
        'Choose your language / Elige tu idioma / Escolha seu idioma',
        ["English", "Español", "Português brasileiro"],
        index=["English", "Español", "Português brasileiro"].index(st.session_state["language"]),
        key="language_select"
    )

# Update session state with the selected language
st.session_state["language"] = language

with col4:
    st.image(flag_path[language], width=40)  # Resize the flag

pg.run()