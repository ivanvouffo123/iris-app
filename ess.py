import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns


# Page 
st.write("bonjour tout le monde")
st.set_page_config(
    page_title="iris Classification", 
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

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
        st.write ("hello world")

    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'
        df = pd.read_csv('C:/Users/GBK STORE/Desktop/Notebooks/iris.csv', delimiter=";")

        # Afficher les premières lignes du jeu de données
        print(df.head())
        print(df.describe())

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"
        sns.countplot(x='Species',data=df)
        plt.title('Distribution des espèces d\'Iris')
        plt.show()

        plt.bar( effectif.index,effectif.values,color='skyblue')
        plt.title("Graphiques en Barres")
        plt.xlabel("Catégories")
        plt.ylabel("Valeurs")
        plt.show()

        plt.pie(effectif.values, autopct='%1.1f%%', startangle=140)
        plt.title("Graphiques en Secteurs")
        plt.show()

        plt.bar( df.PetalLength.index,df.PetalLength.values,color='skyblue')
        plt.title("Graphiques en Barres")
        plt.xlabel("Catégories")
        plt.ylabel("Valeurs")
        plt.show()

        plt.bar( df.PetalWidth.index,df.PetalWidth.values,color='gold')
        plt.title("Graphiques en Barres")
        plt.xlabel("Catégories")
        plt.ylabel("Valeurs")
        plt.show()

        plt.bar( df.SepalLength.index,df.SepalLength.values,color='brown')
        plt.title("Graphiques en Barres")
        plt.xlabel("Catégories")
        plt.ylabel("Valeurs")
        plt.show()

        plt.bar( df.SepalWidth.index,df.SepalWidth.values,color='pink')
        plt.title("Graphiques en Barres")
        plt.xlabel("Catégories")
        plt.ylabel("Valeurs")
        plt.show()


    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

        plt.figure(figsize=(8,8))

        plt.subplot(221)
        sns.scatterplot(data=df,x="SepalLength",y="PetalWidth",hue="Species",palette=['orchid','green',"#77BFE2"])

        plt.subplot(222)
        sns.scatterplot(data=df,x="PetalLength",y="PetalWidth",hue="Species",palette=['orchid','green',"#77BFE2"])

        plt.subplot(223)
        sns.scatterplot(data=df,x="SepalLength",y="SepalWidth",hue="Species",palette=['orchid','green',"#77BFE2"])

        plt.subplot(224)
        sns.scatterplot(data=df,x="SepalLength",y="SepalWidth",hue="Species",palette=['orchid','green',"#77BFE2"])

        plt.show()

        sns.pairplot(df,hue="Species")
        plt.show()

        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        # Calculer la matrice de corrélation
        corr = numeric_df.corr()
        # Afficher la heatmap
        plt.figure(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Heatmap de la corrélation entre les variables numériques")
        plt.show()

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

        from sklearn.model_selection import train_test_split
        # Séparer les caractéristiques et la cible
        X = df.drop('Species', axis=1)
        y = df['Species'] 
        Y=df.Species

        # Diviser les données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
        random_state=42)

        from sklearn.preprocessing import StandardScaler
        # Normaliser les caractéristiques
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        st.write("phase d'apprentissage")


    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

        # Prédire les classes de l'ensemble de test
        y_pred = knn.predict(X_test)

        # Afficher la matrice de confusion
        from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
        xticklabels=df['Species'].unique(), yticklabels=df['Species'].unique())
        plt.title('Matrice de confusion')
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies classes')
        plt.show()

        # Calculer l'exactitude
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Exactitude du modèle : {accuracy * 100:.2f}%")
        # Afficher le rapport de classification
        print("Rapport de classification :\n", classification_report(y_test,y_pred))

        from sklearn.model_selection import GridSearchCV

        param_grid = {'n_neighbors': range(1, 11)}  # Corrected parameter name
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid.fit(X_train, y_train)
        print("Meilleur k :", grid.best_params_)

        import pickle
        with open('model_knn.pkl', 'wb') as file:
            pickle.dump(knn, file)

        from flask import Flask, request, jsonify
        app = Flask(__name__)

        @app.route('/predict', methods=['POST'])
        def predict():
            data = request.json
            features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
            prediction = knn.predict([features])
            return jsonify({'prediction': prediction[0]})

        if __name__ == '_main_':
             app.run(debug=True)




    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

        import streamlit as st
        import requests

        st.title("Prédiction des espèces d'Iris")
        sepal_length = st.number_input("SepalLength")
        sepal_width = st.number_input("SepalWidth")
        petal_length = st.number_input("PetalLength")
        petal_width = st.number_input("PetalWidth")

        if st.button("Prédire"):
            response = requests.post('http://127.0.0.1:5000/predict', json={
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
             })
            st.write("Espèce prédite :", response.json()['prediction'])

        

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
# Afficher les premières lignes des données chargées data
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
#Interactive design representation 
chart2 = alt.Chart(df).mark_circle(size=60).encode(
    x='sepal_length',
    y='sepal_width',
    color='species',
    tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
).interactive()
st.write(chart2)
# About
if st.button("About App"):
	st.subheader("App d'exploration des données des Iris")
	st.text("Contruite avec Streamlit")
	st.text("Thanks to the Streamlit Team Amazing Work")
if st.checkbox("By"):
	st.text("Stéphane C. K. Tékouabou")
	st.text("ctekouaboukoumetio@gmail.com")
