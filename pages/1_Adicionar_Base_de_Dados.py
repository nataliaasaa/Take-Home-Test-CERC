import streamlit as st
import pandas as pd
from utils.utils import loadCSV, clean_df

st.set_page_config(page_title="Base de dados", layout="wide")


st.markdown(''' 
# Base de dados 

A base de dados inicial utilizada é o aquivo corporate_rating_train.csv.

Novas bases de dados podem ser adicionadas à base original, 
basta especificar o nome do arquivo csv no campo disponível. 

Obs: Se o arquivo não estiver no diretório em que o programa está
sendo executado, basta escrever o caminho completo do arqivo.
''')

df = loadCSV()

if st.checkbox("Adicionar novas bases de dados."): 

    filename_new = st.text_input('Digite o nome do novo arquivo de dados a ser adicionado:')

    if st.button('Adicionar!'):
        # Base das empresas noteiras com coluna de classe = 1
        df_new = pd.read_csv(filename_new)
        # Concatenando as duas mais a base original
        df = pd.concat([df, df_new], ignore_index=True)

df_final = clean_df(df)
df_final.to_csv('new_data.csv', index=False)
st.dataframe(df_final)