import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta, datetime

# Set page config
st.set_page_config(page_title="Banque Additionnale", layout="wide")

# Helper functions
@st.cache_data

def load_data():
    df = pd.read_csv("bank-additional-full.csv", delimiter=";")
    df["year"]="2022"
    df["day_of_week"]=df["day_of_week"].astype(str)
    df["last_contact_date"]=df["year"]+"-"+df["month"]+"-"+df["day_of_week"]
    df["last_contact_date"]= pd.to_datetime(df["last_contact_date"],format="%Y-%b-%a")
    df.drop(columns=["day_of_week","year","month"],inplace=True)
   # Nettoyage des colonnnes

    df["job"]=df["job"].str.replace(".","-")
    df["education"]=df["education"].str.replace(".","-")
    df["education"]=df["education"].replace("unknown",np.nan)
    df["marital"]=df["marital"].replace("unknown",np.nan)


    for col in ["default","housing","loan"]:
        df[col]=df[col].map({"yes":1,"no":0,"unknown":0})
        df[col]=df[col].astype(bool)
    
    df["poutcome"]=df["poutcome"].map({"success":1,"failure":0,"nonexistent":0})
    df["poutcome"]=df["poutcome"].astype(bool)
    x=LabelEncoder()

    df['marital']=x.fit_transform(df['marital'])
    df['job']=x.fit_transform(df['job'])
    df['contact']=x.fit_transform(df['contact'])
    df['education']=x.fit_transform(df['education'])

    return df
def create_metric_chart(df, column, color, chart_type, height=150):
    chart_data = df[[column]].copy()
    if chart_type=='Bar':
        st.bar_chart(chart_data, y=column, color=color, height=height)
    if chart_type=='Area':
        st.area_chart(chart_data, y=column, color=color, height=height)
# Load data
df = load_data()

# Set up input widgets
st.logo(image="images/china.jpg", 
        icon_image="images/flower.jpg")

with st.sidebar:
    st.title("YouTube Channel Dashboard")
    st.header("⚙️ Settings")
    
    max_date = df['last_contact_date'].max().date()
    default_start_date = max_date - timedelta(days=365)  # Show a year by default
    default_end_date = max_date
    start_date = st.date_input("Start date", default_start_date, min_value=df['last_contact_date'].min().date(), max_value=df['last_contact_date'])
    end_date = st.date_input("End date", default_end_date, min_value=df['last_contact_date'].min().date(), max_value=df['last_contact_date'])
    time_frame = st.selectbox("Select time frame", ("Daily", "Weekly", "Monthly", "Quarterly"))
    chart_selection = st.selectbox("Select a chart type",
                                   ("Bar", "Area"))
