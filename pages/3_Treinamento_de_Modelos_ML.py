from os import mkdir
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.utils import loadCSV, pickle_model


st.set_page_config(
    page_title="Treinamento do Modelo",
    layout="wide"
)

st.title("🧠 Treinamento e Avaliação de Modelos")

# =========================
# Load data
# =========================
df_credit = loadCSV()

# =========================
# Risk mapping
# =========================
risk_dict = {
    'AAA': 'Lowest Risk',
    'AA': 'Low Risk',
    'A': 'Low Risk',
    'BBB': 'Medium Risk',
    'BB': 'High Risk',
    'B': 'High Risk',
    'CCC': 'Highest Risk',
    'CC': 'Highest Risk',
    'C': 'Highest Risk',
    'D': 'Highest Risk'
}

df_credit["Rating_group"] = df_credit["Rating"].map(risk_dict)

le = LabelEncoder()
df_credit["Rating_encoded"] = le.fit_transform(df_credit["Rating_group"])

# create models directory if not existent
if not os.path.exists("models"):
    mkdir("models")

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
# =========================
# Feature selection
# =========================
st.header("📌 Preparação dos Dados")

df_model = df_credit.drop(
    columns=[
        "Name",
        "Symbol",
        "Rating",
        "Rating_group",
        "Rating Agency Name",
        "Date"
    ]
)

categorical_features = df_model.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_features = df_model.select_dtypes(include=["int64", "float64"]).columns.tolist()

st.write("**Features numéricas:**", numeric_features)
st.write("**Features categóricas:**", categorical_features)

# =========================
# Train-test split
# =========================
X = df_model.drop(columns=["Rating_encoded"])
y = df_model["Rating_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Pipeline
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    class_weight="balanced"
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# =========================
# Training
# =========================
st.header("Treinamento do Modelo")

if st.button("Treinar Random Forest"):
    with st.spinner("Treinando modelo..."):
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Acurácia do modelo: **{acc:.2f}**")

    # =========================
    # Classification report
    # =========================
    # st.subheader("📄 Classification Report")

    # report = classification_report(
    #     y_test,
    #     y_pred,
    #     target_names=le.classes_,
    #     output_dict=True
    # )

    # st.dataframe(pd.DataFrame(report).transpose())

    # =========================
    # Confusion Matrix
    # =========================
    # st.subheader("🔁 Matriz de Confusão")


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Métricas do Modelo")
        st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
        st.text(classification_report(y_test, y_pred,))

    with col2:
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Reds",
            linewidths=0.5,
            linecolor="white",
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

    # =========================
    # Feature Importance
    # =========================
    st.subheader("📊 Feature Importance (Agregada)")

    # Numeric features
    num_features = numeric_features

    # One-hot encoded categorical features
    cat_features = pipe.named_steps["preprocess"] \
        .named_transformers_["cat"] \
        .get_feature_names_out(categorical_features)

    all_features = np.concatenate([num_features, cat_features])
    importances = pipe.named_steps["model"].feature_importances_

    fi_raw = pd.Series(importances, index=all_features)

    def aggregate_feature_importance(fi_raw, numeric_features, categorical_features):
        aggregated = {}

        for col in numeric_features:
            aggregated[col] = fi_raw.get(col, 0)

        for col in categorical_features:
            mask = fi_raw.index.str.startswith(col + "_")
            aggregated[col] = fi_raw[mask].sum()

        return pd.Series(aggregated).sort_values(ascending=False)

    fi_agg = aggregate_feature_importance(
        fi_raw,
        numeric_features,
        categorical_features
    )

    fig_fi = px.bar(
        fi_agg.head(15)[::-1],
        orientation="h",
        title="Top Features – Random Forest"
    )

    fig_fi.update_layout(
        xaxis_title="Importância",
        yaxis_title="Feature"
    )

    st.plotly_chart(fig_fi, use_container_width=True)

    # =========================
    # Model discussion
    # =========================
    st.info(
        "O primeiro modelo treinado obteve uma acurácia de 0.68. " \
        "No entanto, investgando o resultado do feature importance plot pode-se " \
        "notar que a coluna de maior importancia era Symbol -> o simbolo (nome) da empresa. "
        "A coluna fo removida e modelo retreinado, a acurácia abaixou para 0.63."
    )

    st.download_button("Download .pkl file", data=pickle_model(model), file_name="decision_tree_ml.pkl")

# =========================
# PyCaret – Sanity Check
# =========================
st.header("🧪 Benchmarking com PyCaret")

st.warning(
    "PyCaret é utilizado como sanity check. "
    "O treinamento pode levar alguns minutos."
)

if st.button("Executar PyCaret (Benchmarking)"):
    with st.spinner("Rodando PyCaret..."):

        from pycaret.classification import (
            setup,
            pull,
            compare_models,
            automl,
            save_model,
            predict_model
        )

        # Setup
        clf_setup = setup(
            data=X_train.copy(),
            target=y_train.copy(),
            session_id=42,
            fold=5,
        )

        # Compare models
        best_model = compare_models()

        best_model_results = pull()
        st.dataframe(best_model_results)

        st.success(f"Melhor modelo segundo PyCaret: **{best_model.__class__.__name__}**")

        # st.subheader("📄 Otimizar modelo utilizando a metrica Precisao: reduz falsos positivos.")
        # # Optimize for precision (minimizing false positives)
        # tuned_model = tune_model(best_model, optimize="Precision")

        # tuned_model_results = pull()
        # st.dataframe(tuned_model_results)

        # Predictions
        preds = predict_model(best_model, data=X_test)

        # =========================
        # Metrics
        # =========================
        # st.subheader("📄 Classification Report (PyCaret)")

        # report_pc = classification_report(
        #     y_test,
        #     preds["prediction_label"],
        #     target_names=le.classes_,
        #     output_dict=True
        # )

        # st.dataframe(pd.DataFrame(report_pc).transpose())

        # =========================
        # Confusion Matrix
        # =========================
        # st.subheader("🔁 Matriz de Confusão (PyCaret)")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Métricas do Modelo")
            report_pc = classification_report(
                        y_test,
                        preds["prediction_label"])
            st.text(report_pc)

        with col2:
            st.subheader("Matriz de Confusão")
            cm_pc = confusion_matrix(
            y_test,
            preds["prediction_label"])

            cm_pc_df = pd.DataFrame(
                cm_pc,
                index=le.classes_,
                columns=le.classes_)

            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(
                cm_pc_df,
                annot=True,
                fmt="d",
                cmap="Reds",
                linewidths=0.5,
                linecolor="white",
                ax=ax
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix – PyCaret")

            st.pyplot(fig)

        

        st.info(
            "Os resultados do PyCaret corroboram que modelos baseados em árvores "
            "(Random Forest / XGBoost) são mais adequados para este dataset."
        )


        st.download_button("Download .pkl file", data=pickle_model(best_model), file_name="best_model_ml.pkl")
