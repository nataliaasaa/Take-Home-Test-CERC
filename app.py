import streamlit as st

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="📊",
    layout="wide"
)

st.title("Credit Risk Assessment App")

st.markdown("""
Bem-vindo ao sistema de avaliação de risco de crédito corporativo.

### Navegação:
- **Exploração**: análise dos dados e métricas financeiras
- **Treinamento**: comparação e seleção de modelos de ML
- **Treinamento com Regras Financeiras**: criacao de regras financeiras + ML            
- **Predição**: avaliação de risco usando ML + regras financeiras
""")

st.info("Use o menu à esquerda para navegar entre as páginas.")
