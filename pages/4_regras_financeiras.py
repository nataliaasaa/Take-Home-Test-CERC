import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utils.utils import loadCSV, pickle_model
import datetime
import io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

st.set_page_config(page_title="Predição de Risco de Crédito", layout="wide")

st.title("📊 Predição de Risco de Crédito Corporativo")
st.markdown("---")

# ==========================
# Texto explicativo
# ==========================
with st.expander("📘 Metodologia de Avaliação de Risco", expanded=True):
    st.markdown("""
    Esta abordagem combina **análise financeira clássica** com **Machine Learning**.

    **Etapas principais:**
    1. Construção de scores financeiros
    2. Normalização dos indicadores
    3. Classificação binária de risco
    4. Modelo XGBoost
    5. Regras financeiras adicionais
    6. Score final híbrido
    """)

# ==========================
# Carregar dados
# ==========================
df_credit = loadCSV()

st.subheader("📥 Dataset carregado")
st.dataframe(df_credit.head())

# ==========================
# Rating Mapping
# ==========================
rating_dict = {
    'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3,
    'BB': 4, 'B': 5, 'CCC': 6, 'CC': 7, 'C': 8, 'D': 9
}

df_credit["Rating_id"] = df_credit["Rating"].map(rating_dict)

# ============================
# Financial scores
# ============================

st.subheader("📐 Construção dos Scores Financeiros")

st.markdown(r"""
### 🔹 1. Score de Liquidez
**Mede a capacidade da empresa de honrar obrigações de curto prazo**

$Liquidity = 0.4 \times CurrentRatio + 0.3 \times QuickRatio + 0.3 \times CashRatio$
""")

df_credit["liquidity_score"] = (
    0.4 * df_credit["currentRatio"] +
    0.3 * df_credit["quickRatio"] +
    0.3 * df_credit["cashRatio"]
)

st.markdown(r"""
### 🔹 2. Score de Rentabilidade
**Avalia eficiência operacional e geração de lucro**

$Profitability = 0.25 \times GrossMargin + 0.25 \times OperatingMargin + 0.25 \times NetMargin + 0.25 \times ROA$
""")

df_credit["profitability_score"] = (
    0.25 * df_credit["grossProfitMargin"] +
    0.25 * df_credit["operatingProfitMargin"] +
    0.25 * df_credit["netProfitMargin"] +
    0.25 * df_credit["returnOnAssets"]
)

st.markdown(r"""
### 🔹 3. Score de Endividamento
**Risco financeiro associado à alavancagem**

$Leverage = 0.6 \times DebtRatio + 0.4 \times DebtEquityRati$
""")

df_credit["leverage_score"] = (
    0.6 * df_credit["debtRatio"] +
    0.4 * df_credit["debtEquityRatio"]
)

st.markdown(r"""
### 🔹 4. Score de Fluxo de Caixa
**Capacidade de geração de caixa operacional**

$CashFlow = 0.5 \times \frac{OCF}{Share} + 0.5 \times \frac{FCF}{Share}$
""")

df_credit["cashflow_score"] = (
    0.5 * df_credit["operatingCashFlowPerShare"] +
    0.5 * df_credit["freeCashFlowPerShare"]
)

# ============================
# Normalization
# ============================

score_cols = [
    "liquidity_score",
    "profitability_score",
    "leverage_score",
    "cashflow_score"
]

scaler = StandardScaler()
df_credit[score_cols] = scaler.fit_transform(df_credit[score_cols])

st.subheader("📊 Score Financeiro Final")

st.markdown(r"""
O **Financial Health Score** combina todos os blocos financeiros:

$FinalScore = 0.3 \times Liquidity + 0.3 \times Profitability - 0.2 \times Leverage + 0.2 \times CashFlow$
            
""")

df_credit["financial_health_score"] = (
    0.3 * df_credit["liquidity_score"] +
    0.3 * df_credit["profitability_score"] -
    0.2 * df_credit["leverage_score"] +
    0.2 * df_credit["cashflow_score"]
)

# ==========================
# Target binário
# ==========================
df_credit["high_risk"] = (df_credit["Rating_id"] >= 5).astype(int)

features = score_cols + ["financial_health_score"]
X = df_credit[features]
y = df_credit["high_risk"]

st.subheader("📥 Novo dataset com regras financeiras e classificação binária de risco")
st.dataframe(df_credit.head(5))
# ==========================
# Treinamento
# ==========================

if st.button("Treinar XGBClassifier"):
    with st.spinner("Treinando modelo..."):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = XGBClassifier(
            n_estimators=100,
            booster="gbtree",
            objective="binary:logistic",
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)

        st.session_state.model = {"model": True}
        st.session_state.trained = True
        # ==========================
        # Avaliação
        # ==========================
        y_pred = model.predict(X_test)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Métricas do Modelo")
            st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
            st.text(classification_report(y_test, y_pred))

        with col2:
            st.subheader("🔲 Matriz de Confusão")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            st.pyplot(fig)

        # ==========================
        # Regras financeiras
        # ==========================
        def rule_based_risk(row):
            flags = 0
            if row["currentRatio"] < 1:
                flags += 1
            if row["debtRatio"] > 0.6:
                flags += 1
            if row["returnOnAssets"] < 0:
                flags += 1
            if row["operatingCashFlowPerShare"] < 0:
                flags += 1
            return flags

        st.markdown(r"""
                ### 1️⃣ Probabilidade de Risco (Machine Learning)

                O modelo **XGBoost** estima a **probabilidade de uma empresa ser de alto risco**, 
                com base exclusivamente em **indicadores financeiros agregados**.

                $P(Risco = Alto) = \text{Modelo}_{ML}(X)$

                Onde:
                - \(X\) inclui liquidez, rentabilidade, alavancagem, fluxo de caixa e score financeiro final
                - O resultado é um valor contínuo entre **0 e 1**

                📌 **Interpretação**  
                - Valores próximos de **1** → alta chance de risco elevado  
                - Valores próximos de **0** → empresa financeiramente saudável
                """)

        df_credit["ml_risk_probability"] = model.predict_proba(X)[:, 1]
        
        st.markdown("""
        ### 2️⃣ Regras Financeiras (Rule-Based Flags)

        Além do modelo estatístico, aplicamos **regras financeiras clássicas** 
        utilizadas por analistas de crédito para identificar sinais de alerta.

        Cada regra violada adiciona **1 flag de risco**:
        """)

        st.markdown("""
        | Regra Financeira | Justificativa |
        |-----------------|---------------|
        | `Current Ratio < 1` | Risco de insolvência de curto prazo |
        | `Debt Ratio > 0.6` | Estrutura de capital excessivamente alavancada |
        | `ROA < 0` | Operação não gera retorno econômico |
        | `Operating Cash Flow < 0` | Incapacidade de gerar caixa operacional |
        """)
        
        df_credit["rule_flags"] = df_credit.apply(rule_based_risk, axis=1)

        st.markdown(r"""
        ### 3️⃣ Score Final de Risco

        O score final combina:
        - 📊 **Probabilidade estimada pelo modelo**
        - 📏 **Penalização baseada em regras financeiras**

        $FinalRiskScore = 0.7 \times P_{ML} + 0.3 \times \frac{Flags}{Flags_{max}}$

        📌 **Por que essa combinação?**
        - O modelo captura **padrões complexos nos dados**
        - As regras adicionam **robustez econômica e explicabilidade**
        """)


        df_credit["final_risk_score"] = (
            0.7 * df_credit["ml_risk_probability"] +
            0.3 * (df_credit["rule_flags"] / df_credit["rule_flags"].max())
        )

        st.markdown("""
        | Faixa | Intervalo do Score | Interpretação |
        |------|-------------------|---------------|
        | **Low Risk** | 0.00 – 0.33 | Empresa financeiramente saudável |
        | **Medium Risk** | 0.33 – 0.66 | Atenção / Monitoramento |
        | **High Risk** | 0.66 – 1.00 | Alto risco de inadimplência |
        """)
        
        df_credit["risk_bucket"] = pd.cut(
            df_credit["final_risk_score"],
            bins=[0, 0.33, 0.66, 1],
            labels=["Low", "Medium", "High"]
        )

        # ==========================
        # Tabela final
        # ==========================
        st.subheader("📋 Resultado Final de Risco")

        st.markdown(""" 
        Cada empresa recebe uma classificação clara e acionável, 
        adequada para **análise de crédito**, **rating interno** ou **suporte à decisão**.
        """)
        df_result = df_credit[[
            "Name", "Rating", "financial_health_score",
            "ml_risk_probability", "rule_flags",
            "final_risk_score", "risk_bucket"
        ]]

        st.dataframe(df_result)

        # ==========================
        # Salvar modelo
        # ==========================
        st.markdown("---")

    
    if st.session_state.trained:
        st.download_button("Download .pkl file", data=pickle_model(model), file_name="pickled-model-regras_financeiras.pkl")
