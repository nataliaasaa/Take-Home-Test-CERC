import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils import loadCSV, clean_df

st.set_page_config(
    page_title="Exploração de Dados",
    layout="wide"
)

st.title("📊 Exploração de Dados – Corporate Credit Rating")

# =========================
# Load data
# =========================

df_credit = loadCSV()

# =========================
# Dataset overview
# =========================
st.header("📁 Visão Geral do Dataset")

with st.expander("Visualizar dados"):
    st.dataframe(df_credit.head(10))

with st.expander("Informações gerais"):
    #st.dataframe(df_credit.info())
    buffer = io.StringIO()
    df_credit.info(buf=buffer)
    st.text(buffer.getvalue())

with st.expander("Estatísticas descritivas"):
    st.dataframe(df_credit.describe())


# =========================
# Rating distribution
# =========================
st.header("🏷️ Distribuição dos Ratings")

rating_order = [
    "AAA", "AA", "A",
    "BBB", "BB", "B",
    "CCC", "CC", "C", "D"
]

fig_rating = px.histogram(
    df_credit,
    x="Rating",
    category_orders={"Rating": rating_order}
)

st.plotly_chart(fig_rating, use_container_width=True)

# =========================
# Boxplots – global distributions
# =========================
st.header("📦 Distribuição Global de Features Financeiras")

fig_box = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Current Ratio",
        "Quick Ratio",
        "Net Profit Margin",
        "Gross Profit Margin"
    ]
)

fig_box.add_trace(
    go.Box(y=df_credit["currentRatio"], name="Current Ratio"),
    row=1, col=1
)

fig_box.add_trace(
    go.Box(y=df_credit["quickRatio"], name="Quick Ratio"),
    row=1, col=2
)

fig_box.add_trace(
    go.Box(y=df_credit["netProfitMargin"], name="Net Profit Margin"),
    row=2, col=1
)

fig_box.add_trace(
    go.Box(y=df_credit["grossProfitMargin"], name="Gross Profit Margin"),
    row=2, col=2
)

fig_box.update_layout(showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

st.info(
    "A presença significativa de outliers sugere a necessidade de uma análise especializada "
    "para avaliar se representam erros de entrada ou valores financeiros legítimos."
)

# =========================
# Feature by Rating
# =========================
st.header("📊 Distribuição de Features por Rating")

feature = st.selectbox(
    "Selecione a feature:",
    [
        "netProfitMargin",
        "grossProfitMargin",
        "currentRatio",
        "quickRatio"
    ]
)

fig_feat_rating = px.box(
    df_credit,
    x="Rating",
    y=feature,
    category_orders={"Rating": rating_order},
    title=f"Distribuição de {feature} por Rating"
)

st.plotly_chart(fig_feat_rating, use_container_width=True)

# =========================
# Temporal analysis
# =========================
st.header("📈 Análise Temporal por Empresa")

company = st.selectbox(
    "Selecione a empresa:",
    df_credit["Name"].value_counts().index.tolist()
)

df_company = df_credit[df_credit["Name"] == company]

fig_time = make_subplots(
    rows=2, cols=1,
    subplot_titles=[
        "Credit Rating",
        "Gross Profit Margin"
    ]
)

fig_time.add_trace(
    go.Scatter(
        x=df_company["Date"],
        y=df_company["Rating"],
        mode="markers",
        name="Rating"
    ),
    row=1, col=1
)

fig_time.add_trace(
    go.Scatter(
        x=df_company["Date"],
        y=df_company["grossProfitMargin"],
        mode="markers",
        name="Gross Profit Margin"
    ),
    row=2, col=1
)

fig_time.update_layout(showlegend=False)
st.plotly_chart(fig_time, use_container_width=True)

# =========================
# Correlation analysis
# =========================
st.header("🔗 Análise de Correlação")

rating_dict = {
    'AAA': 0, 'AA': 1, 'A': 2,
    'BBB': 3, 'BB': 4, 'B': 5,
    'CCC': 6, 'CC': 7, 'C': 8, 'D': 9
}

df_corr = df_credit.copy()
df_corr["Rating_id"] = df_corr["Rating"].map(rating_dict)

num_df = df_corr.select_dtypes(include=["int64", "float64"])
corr_spearman = num_df.corr(method="spearman")

fig_corr = px.imshow(
    corr_spearman,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="Matriz de Correlação (Spearman)"
)

fig_corr.update_layout(
    width=900,
    height=900
)

st.plotly_chart(fig_corr, use_container_width=True)

st.info(
    "Nenhuma variável apresenta correlação forte (>0.5) com o Rating_id, "
    "indicando que a predição depende de relações multivariadas."
)
