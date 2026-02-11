import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import loadCSV, pickle_model

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
    Esta abordagem combina análise financeira clássica com Machine Learning.

    **Etapas principais:**
    1. Construção de scores financeiros e Regras financeiras adicionais
    2. Normalização dos indicadores
    3. Classificação binária de risco
    4. Modelo XGBoost
    5. Score final
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
Mede a capacidade da empresa de honrar obrigações de curto prazo

$Liquidity = 0.33 \times CurrentRatio + 0.33 \times QuickRatio + 0.33 \times CashRatio$
""")

df_credit["liquidity_score"] = (
    0.33 * df_credit["currentRatio"] +
    0.33 * df_credit["quickRatio"] +
    0.33 * df_credit["cashRatio"]
)

st.markdown(r"""
### 🔹 2. Score de Rentabilidade
Avalia eficiência operacional e geração de lucro

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
Risco financeiro associado à alavancagem

$Leverage = 0.5 \times DebtRatio + 0.5 \times DebtEquityRati$
""")

df_credit["leverage_score"] = (
    0.5 * df_credit["debtRatio"] +
    0.5 * df_credit["debtEquityRatio"]
)

st.markdown(r"""
### 🔹 4. Score de Fluxo de Caixa
Capacidade de geração de caixa operacional

$CashFlow = 0.5 \times operating Cash Flow Per Share + 0.5 \times free Cash Flow Per Share$
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

# scaler = StandardScaler()
# df_credit[score_cols] = scaler.fit_transform(df_credit[score_cols])

st.subheader("📊 Score Financeiro Final")

st.markdown(r"""
O Financial Health Score combina todos os blocos financeiros:

$FinalScore = 0.25 \times Liquidity + 0.25 \times Profitability - 0.25 \times Leverage + 0.25 \times CashFlow$
            
""")

df_credit["financial_health_score"] = (
    0.25 * df_credit["liquidity_score"] +
    0.25 * df_credit["profitability_score"] -
    0.25 * df_credit["leverage_score"] +
    0.25 * df_credit["cashflow_score"]
)

st.markdown("""
        ### 2️⃣ Regras Financeiras (Rule-Based Flags)

        Aplicamos regras financeiras para identificar sinais de alerta.

        Cada regra violada adiciona 1 flag de risco, e o valor é adicionado como feature no dataset:
        """)

st.markdown("""
        | Regra Financeira | Justificativa |
        |-----------------|---------------|
        | `liquidity score < 1` | Razão de Liquidez deve ser maior que 1  |
        | `profitability score < 0.2` | Buscamos uma margem de lucro maior que 20% |
        | `leverage score > 2` | Razão de endividamento elevada |
        | `cashflow score < 0.2` | Incapacidade de gerar caixa operacional |
        """)

def rule_based_risk(row):
    flags = 0

    if row["liquidity_score"] < 1:
    # Ideally the liquidity ratios will be greater than 1
        flags += 1
    if row["profitability_score"] < 0.2:
    # We want a profitability ratio greater than 20%
        flags += 1
    if row["leverage_score"] > 2:
    # Don't want a leverage ratio greater than 2
        flags += 1
    if row["cashflow_score"] < 0.2:
    # minimun operatingCashFlowPerShare of 20%
        flags += 1

    return flags

df_credit["rule_flags"] = df_credit.apply(rule_based_risk, axis=1)

# ==========================
# Target binário
# ==========================
df_credit["high_risk"] = (df_credit["Rating_id"] >= 5).astype(int)

features = score_cols + ["financial_health_score"] + ["rule_flags"]
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
      
        st.markdown(r"""
                ### 1️⃣ Probabilidade de Risco (Machine Learning)

                O modelo XGBoost estima a probabilidade de uma empresa ser de alto risco, 
                com base exclusivamente em indicadores financeiros agregados.

                $P(Risco = Alto) = \text{Modelo}_{ML}(X)$

                Onde:
                - \(X\) inclui liquidez, rentabilidade, alavancagem, fluxo de caixa, score financeiro final e flags
                - O resultado é um valor contínuo entre 0 e 1

                Interpretação
                - Valores próximos de 1 → alta chance de risco elevado  
                - Valores próximos de 0 → empresa financeiramente saudável
                """)
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Métricas do Modelo")
            st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
            st.text(classification_report(y_test, y_pred))

        with col2:
            st.subheader("Matriz de Confusão")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            st.pyplot(fig)

        df_credit["ml_risk_probability"] = model.predict_proba(X)[:, 1]
        
        df_credit["final_risk_score"] = (df_credit["ml_risk_probability"])

        st.markdown(r"""
        ### 3️⃣ Score Final de Risco

        O score final é dado pela Probabilidade estimada pelo modelo """)

        st.markdown("""
        | Faixa | Intervalo do Score | Interpretação |
        |------|-------------------|---------------|
        | Low Risk | 0.00 – 0.33 | Empresa financeiramente saudável |
        | Medium Risk | 0.33 – 0.66 | Atenção / Monitoramento |
        | High Risk | 0.66 – 1.00 | Alto risco de inadimplência |
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

        df_result = pd.DataFrame({
                        "Name": df_credit["Name"],
                        "Rating": df_credit["Rating"],
                        "Financial Health Score": df_credit["financial_health_score"],
                        "Rule Flags": df_credit["rule_flags"],
                        "ML Risk Probability": df_credit["ml_risk_probability"],
                        "Risk Bucket": df_credit["risk_bucket"]
                    })

        st.dataframe(df_result)

        # ==========================
        # Salvar modelo
        # ==========================
        st.markdown("---")

    
    if st.session_state.trained:
        st.download_button("Download .pkl file", data=pickle_model(model), file_name="pickled-model-regras_financeiras.pkl")
