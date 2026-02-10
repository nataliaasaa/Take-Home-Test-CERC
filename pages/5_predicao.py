import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import load_model, predict_model

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="Predição de Risco de Crédito", layout="wide")

st.title("📊 Predição de Risco de Crédito Corporativo")

st.markdown("""
Esta página realiza a **avaliação de risco em dois estágios**:

1. **Modelo de Machine Learning supervisionado**
2. **Modelo binário + regras financeiras clássicas**

O segundo estágio tem como objetivo **reduzir falsos positivos de alto risco**.
""")

# ============================================================
# 1️⃣ Upload da base do usuário
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Faça upload da base de dados da empresa (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Envie um arquivo CSV para iniciar a predição.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("📄 Preview da base enviada")
st.dataframe(df.head())

# ============================================================
# 2️⃣ Carregar modelos treinados
# ============================================================

@st.cache_resource
def load_models():

    ml_model = load_model("models/best_model_ml")

    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("models/pickled-model-regras_financeiras.pkl", "rb") as f:
        binary_model = pickle.load(f)

    return ml_model, label_encoder, binary_model

ml_model, label_encoder, binary_model = load_models()

# ============================================================
# 3️⃣ Predição — Estágio 1 (ML supervisionado)
# ============================================================

st.markdown("## 🧠 Estágio 1 — Predição com Machine Learning")

features_ml = df.drop(columns=["Unnamed: 0", "Name", "Symbol", "Rating", "Rating Agency Name", "Date"], errors="ignore")

predictions_ml = predict_model(ml_model, data=features_ml)

df["ml_predicted_class"] = predictions_ml['prediction_label']

df["ml_predict_label"] = label_encoder.inverse_transform(df["ml_predicted_class"])

st.success("Predição inicial realizada com sucesso.")

# ============================================================
# 4️⃣ Engenharia financeira (explicada)
# ============================================================

st.markdown("""
## 📐 Estágio 2 — Avaliação Financeira + Regras

### Scores financeiros utilizados:
- **Liquidez**
- **Rentabilidade**
- **Endividamento**
- **Fluxo de Caixa**

Os scores são normalizados e combinados em um **Financial Health Score**.
""")

df["liquidity_score"] = (
    0.4 * df["currentRatio"] +
    0.3 * df["quickRatio"] +
    0.3 * df["cashRatio"]
)

df["profitability_score"] = (
    0.25 * df["grossProfitMargin"] +
    0.25 * df["operatingProfitMargin"] +
    0.25 * df["netProfitMargin"] +
    0.25 * df["returnOnAssets"]
)

df["leverage_score"] = (
    0.6 * df["debtRatio"] +
    0.4 * df["debtEquityRatio"]
)

df["cashflow_score"] = (
    0.5 * df["operatingCashFlowPerShare"] +
    0.5 * df["freeCashFlowPerShare"]
)

score_cols = [
    "liquidity_score",
    "profitability_score",
    "leverage_score",
    "cashflow_score"
]

scaler = StandardScaler()
df[score_cols] = scaler.fit_transform(df[score_cols])

df["financial_health_score"] = (
    0.3 * df["liquidity_score"] +
    0.3 * df["profitability_score"] -
    0.2 * df["leverage_score"] +
    0.2 * df["cashflow_score"]
)

# ============================================================
# 5️⃣ Modelo binário + regras financeiras
# ============================================================

st.markdown("## ⚖️ Redução de Falsos Positivos")

df["risk_probability"] = binary_model.predict_proba(
    df[score_cols + ["financial_health_score"]]
)[:, 1]

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

df["rule_flags"] = df.apply(rule_based_risk, axis=1)

df["final_risk_score"] = (
    0.7 * df["risk_probability"] +
    0.3 * (df["rule_flags"] / df["rule_flags"].max())
)

df["risk_bucket"] = pd.cut(
    df["final_risk_score"],
    bins=[0, 0.33, 0.66, 1],
    labels=["Low", "Medium", "High"]
)

# ============================================================
# 6️⃣ Resultado final
# ============================================================

st.markdown("## ✅ Resultado Final da Avaliação")

ml_to_bucket = {
    "Lowest Risk": "Low",
    "Low Risk": "Low",
    "Medium Risk": "Medium",
    "High Risk": "High",
    "Highest Risk": "High"
}

df["ml_risk_bucket"] = df["ml_predict_label"].map(ml_to_bucket)

df["risk_disagreement"] = df["ml_risk_bucket"] != df["risk_bucket"]

# Armazenar resultados na sessão para uso posterior
st.session_state["final_results_df"] = df

final_cols = [
    "Name",
    "ml_predict_label",
    "risk_probability",
    "rule_flags",
    "final_risk_score",
    "risk_bucket",
    "risk_disagreement"
]

final_cols = [c for c in final_cols if c in df.columns]


def highlight_disagreement(row):
    if row["risk_disagreement"]:
        return ["background-color: #ffcccc"] * len(row)
    else:
        return [""] * len(row)

styled_df = df[final_cols].style.apply(highlight_disagreement, axis=1)

st.dataframe(styled_df, use_container_width=True)

agreement_summary = (df["risk_disagreement"].map({True: "Discordam", False: "Concordam"}).value_counts())

st.subheader("📊 Concordância entre os modelos")

col1, col2, col3 = st.columns(3)

col1.metric(
    label="Total de casos",
    value=len(df)
)

col2.metric(
    label="Casos que concordam",
    value=int(agreement_summary.get("Concordam", 0))
)

col3.metric(
    label="Casos que discordam",
    value=int(agreement_summary.get("Discordam", 0))
)


# ============================================================
# 7️⃣ Download
# ============================================================

st.download_button(
    "⬇️ Baixar resultados",
    data=df.to_csv(index=False),
    file_name="credit_risk_prediction.csv",
    mime="text/csv"
)
