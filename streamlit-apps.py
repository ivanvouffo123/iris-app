import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Iris Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# -------------------------
# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:
    st.title('Iris Classification')

    # Page Button Navigation
    st.subheader("Pages")

    # Define button logic to activate based on page selection
    about_button = st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',))
    dataset_button = st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',))
    eda_button = st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',))
    data_cleaning_button = st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',))
    machine_learning_button = st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',))
    prediction_button = st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',))
    conclusion_button = st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',))

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")

# -------------------------

# Load data
df = pd.read_csv('iris.csv', delimiter=',')

# Set page title
st.title('ISJM BI - Exploration des données des Iris')

st.header('Pré-analyse visuelles données données des Iris TP1')  # On définit l'en-tête d'une section

# Afficher les premières lignes des données chargées
#st.write(df.head())
    
st.subheader('Description des données')  # Sets a subheader for a subsection

# Show Dataset
if st.checkbox("Boutons de prévisualisation du DataFrame"):
    if st.button("Head"):
        st.write(df.head(2))
    if st.button("Tail"):
        st.write(df.tail())
    if st.button("Infos"):
        st.write(df.info())
    if st.button("Shape"):
        st.write(df.shape)
else:
    st.write(df.head(2))

# Create chart
chart = alt.Chart(df).mark_point().encode(
    x='petal_length',
    y='petal_width',
    color="species"
)

# Display chart
st.write(chart)

# Interactive design representation 
chart2 = alt.Chart(df).mark_circle(size=60).encode(
    x='sepal_length',
    y='sepal_width',
    color='species',
    tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
).interactive()

st.write(chart2)

# Page specific content
if st.session_state.page_selection == 'about':
    st.subheader("About App")
    st.text("App d'exploration des données des Iris")
    st.text("Construite avec Streamlit")
    st.text("Thanks to the Streamlit Team Amazing Work")

if st.session_state.page_selection == 'dataset':
    st.subheader("Dataset")
    st.write(df)

if st.session_state.page_selection == 'eda':
    st.subheader("Exploration des Données")
    st.write("Visualisation des relations entre les différentes variables")

if st.session_state.page_selection == 'data_cleaning':
    st.subheader("Data Cleaning / Pre-processing")
    st.write("Prétraitement des données")

if st.session_state.page_selection == 'machine_learning':
    st.subheader("Machine Learning")
    st.write("Entraînement des modèles de classification")

if st.session_state.page_selection == 'prediction':
    st.subheader("Prediction")
    st.write("Faire des prédictions")

if st.session_state.page_selection == 'conclusion':
    st.subheader("Conclusion")
    st.write("Résumé et prochaines étapes")

# About the author
if st.button("About Author"):
    st.text("Stéphane C. K. Tékouabou")
    st.text("ctekouaboukoumetio@gmail.com")
