import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import base64
import io 
import pickle

@st.cache_data
def loadCSV():
    df = pd.read_csv('data/corporate_rating_train.csv')
    return df

@st.cache_data
def loadnewCSV(nome):
    df = pd.read_csv(nome)
    return df

def clean_df(df):
    
    # Para verificar se há algum valor nulo em todo o dataset
    if df.isnull().sum().any():
        colunas_com_valores_nulos = df.columns[df.isnull().any()]
        print("Colunas com valores nulos removdos:")
        df.dropna(axis=1, how='all', inplace=True)
        for coluna in colunas_com_valores_nulos:
            print(f" - {coluna}")
    else:
        print("Não há valores nulos no dataset.")
    
    df.drop_duplicates(inplace=True, ignore_index=True)

    return df

def pickle_model(model):
        """Pickle the model inside bytes. In our case, it is the "same" as 
        storing a file, but in RAM.
        """
        f = io.BytesIO()
        pickle.dump(model, f)
        return f

# def to_excel(df):
#             output = BytesIO()
#             writer = pd.ExcelWriter(output)
#             df.to_excel(writer, sheet_name='Sheet1')
#             writer.save()
#             processed_data = output.getvalue()
#             return processed_data

# def get_table_download_link(df):
    
#     """Generates a link allowing the data in a given panda dataframe to be downloaded
#     in:  dataframe
#     out: href string
#     """
#     val = to_excel(df)
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download xlsx file</a>' # decode b'abc' => abc