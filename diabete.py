import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
import base64 
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Diabetes Analysis App",
    page_icon="🏥",
    layout="wide"
)

# Fonction pour charger le csv
@st.cache_data  # Mise en cache des données
def load_data(dataset):
    if isinstance(dataset, str):
        df = pd.read_csv(dataset)
    else:
        df = pd.read_csv(dataset)
    return df

# Fonction pour télécharger le csv
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="diabetes_predictions.csv">Download CSV File</a>'
    return href

# Chargement des images et données
try:
    st.sidebar.image('diabets.jpg', width=200)
except FileNotFoundError:
    st.sidebar.warning("Image 'diabets.jpg' non trouvée")

try:
    data = load_data('diabetes.csv')
except FileNotFoundError:
    st.error("Le fichier 'diabetes.csv' n'a pas été trouvé. Veuillez vérifier le chemin du fichier.")
    data = None

def main():
    # Titre de l'application
    st.markdown(
        "<h1 style='text-align:center;color:#8B4513;'>Application d'Analyse du Diabète</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<h2 style='text-align:center;color:#A0522D;'>Étude du diabète au Cameroun</h2>",
        unsafe_allow_html=True
    )

    # Menu sur le sidebar
    menu = ['Accueil', 'Analyse', 'Visualisation', 'Prédiction']
    choice = st.sidebar.selectbox('Sélectionnez une section', menu)

    if choice == 'Accueil':
        left, middle, right = st.columns(3)
        with middle:
            try:
                st.image('vaccination.jpg', width=500)
            except FileNotFoundError:
                st.warning("Image 'vaccination.jpg' non trouvée")

        st.write("Cette application analyse les données sur le diabète avec des outils Python pour optimiser les décisions")
        st.subheader('Informations sur le diabète')
        st.write("""
        Au Cameroun, la prévalence du diabète chez les adultes en zones urbaines est actuellement estimée 
        entre 6 et 8%, avec environ 80% des personnes vivant avec le diabète qui ne sont pas diagnostiquées. 
        Selon les données de 2002, seul un quart des personnes diabétiques connues avaient un contrôle 
        adéquat de leur glycémie. Le fardeau du diabète au Cameroun est non seulement élevé mais augmente 
        rapidement, avec une multiplication par 10 de la prévalence sur 10 ans (1994-2004).
        """)

    elif choice == 'Analyse' and data is not None:
        st.subheader('Données sur le diabète')
        st.write(data.head())
        
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox('Afficher le résumé statistique'):
                st.write(data.describe())
        
        with col2:
            if st.checkbox('Afficher la matrice de corrélation'):
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

    elif choice == 'Visualisation' and data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox('Distribution par âge'):
                fig = plt.figure(figsize=(10, 6))
                sns.histplot(data=data, x='Age', bins=30)
                plt.title('Distribution des âges')
                st.pyplot(fig)
        
        with col2:
            if st.checkbox('Glucose vs Âge'):
                fig = plt.figure(figsize=(10, 6))
                sns.scatterplot(data=data, x='Glucose', y='Age', hue='Outcome')
                plt.title('Relation entre glucose et âge')
                st.pyplot(fig)

    elif choice == 'Prédiction':
        st.subheader('Prédiction du diabète')
        
        uploaded_file = st.file_uploader('Téléchargez votre fichier CSV', type=['csv'])

        if uploaded_file:
            df = load_data(uploaded_file)
            st.write("Données chargées :", df.head())

            try:
                with open('model_reg.pkl', 'rb') as file:
                    model = pickle.load(file)
                
                prediction = model.predict(df)
                pp = pd.DataFrame(prediction, columns=['Prédiction'])
                ndf = pd.concat([df, pp], axis=1)
                ndf['Prédiction'] = ndf['Prédiction'].map({0: 'Non diabétique', 1: 'Diabétique'})
                
                st.write("Résultats de la prédiction :")
                st.write(ndf)

                if st.button('Télécharger les résultats'):
                    st.markdown(filedownload(ndf), unsafe_allow_html=True)

            except FileNotFoundError:
                st.error("Le modèle n'a pas été trouvé. Veuillez vérifier que 'model_reg.pkl' est présent.")
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de la prédiction : {str(e)}")

if __name__ == '__main__':
    main()
