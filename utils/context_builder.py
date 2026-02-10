# utils/context_builder.py

import pandas as pd
from typing import Optional

from typing import Optional
import pandas as pd


def build_risk_context(
    df: pd.DataFrame,
    company_name: Optional[str] = None,
    top_n: int = 5
) -> str:
    """
    Constrói um contexto textual estruturado para análise de risco de crédito,
    adequado para uso com LLMs (Gemini, GPT, etc).

    Args:
        df: DataFrame com resultados finais de risco
        company_name: Empresa selecionada para análise detalhada (None ou "Todas" = portfólio)
        top_n: Número de empresas de maior risco a listar

    Returns:
        Contexto textual formatado
    """

    # -----------------------------
    # 1. Validações
    # -----------------------------
    required_cols = [
        "Name",
        "Rating",
        "final_risk_score",
        "risk_probability",
        "risk_bucket"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    if df.empty:
        return "⚠️ Nenhum dado de risco disponível para análise."

    df = df.copy()

    # Garantir tipos
    df["final_risk_score"] = pd.to_numeric(df["final_risk_score"], errors="coerce")
    df["risk_probability"] = pd.to_numeric(df["risk_probability"], errors="coerce")

    df = df.dropna(subset=["final_risk_score", "risk_probability", "risk_bucket"])

    if df.empty:
        return "⚠️ Dados insuficientes após limpeza."

    total = len(df)

    # -----------------------------
    # 2. Estatísticas do portfólio
    # -----------------------------
    risk_dist = (
        df["risk_bucket"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    avg_score = df["final_risk_score"].mean()
    avg_prob = df["risk_probability"].mean() * 100

    disagreements = None
    if "ml_risk_bucket" in df.columns:
        disagreements = (df["ml_risk_bucket"] != df["risk_bucket"]).sum()

    # -----------------------------
    # 3. Empresa selecionada
    # -----------------------------
    selected_company = None
    if company_name and company_name != "Todas":
        selected_company = df[df["Name"] == company_name]
        if selected_company.empty:
            selected_company = None
        else:
            selected_company = selected_company.iloc[0]

    # -----------------------------
    # 4. Top-N empresas por risco
    # -----------------------------
    top_companies = (
        df.sort_values("final_risk_score", ascending=False)
          .head(top_n)
    )

    # -----------------------------
    # 5. Construção do contexto
    # -----------------------------
    lines = []

    # Header
    lines.append("📊 ANÁLISE DE RISCO DE CRÉDITO CORPORATIVO")
    lines.append("=" * 80)
    lines.append(f"📅 Total de empresas analisadas: {total}")
    lines.append("")

    # Portfólio
    lines.append("📈 VISÃO GERAL DO PORTFÓLIO")
    lines.append("-" * 80)
    lines.append("Distribuição de risco:")
    for bucket, count in risk_dist.items():
        pct = count / total * 100
        lines.append(f" • {bucket}: {count} empresas ({pct:.1f}%)")

    lines.append("")
    lines.append(f"Score médio de risco: {avg_score:.2f}/100")
    lines.append(f"Probabilidade média de default (12m): {avg_prob:.2f}%")

    if disagreements is not None:
        lines.append(
            f"Divergências ML vs Regras: {disagreements} "
            f"({disagreements / total * 100:.1f}%)"
        )

    # Empresa selecionada
    if selected_company is not None:
        lines.append("")
        lines.append("🏢 EMPRESA SELECIONADA – ANÁLISE DETALHADA")
        lines.append("-" * 80)
        lines.append(f"Empresa: {selected_company['Name']}")
        lines.append(f"Rating: {selected_company.get('Rating', 'N/A')}")
        lines.append(
            f"Score de risco final: "
            f"{selected_company['final_risk_score']:.2f}/100"
        )
        lines.append(
            f"Probabilidade de default (12m): "
            f"{selected_company['risk_probability'] * 100:.2f}%"
        )
        lines.append(f"Classificação de risco: {selected_company['risk_bucket']}")

        if "ml_risk_bucket" in selected_company:
            lines.append(f"Classificação ML: {selected_company['ml_risk_bucket']}")
            lines.append(f"Classificação Regras: {selected_company['risk_bucket']}")

            if selected_company["ml_risk_bucket"] != selected_company["risk_bucket"]:
                lines.append("⚠️ Divergência entre ML e Regras – requer análise manual")

    # Top-N
    lines.append("")
    lines.append(f"🏢 TOP {top_n} EMPRESAS COM MAIOR RISCO")
    lines.append("-" * 80)

    for i, (_, row) in enumerate(top_companies.iterrows(), 1):
        lines.append(f"{i}. {row['Name']}")
        lines.append(f"   Rating: {row.get('Rating', 'N/A')}")
        lines.append(f"   Score: {row['final_risk_score']:.2f}/100")
        lines.append(f"   Prob. Default: {row['risk_probability'] * 100:.2f}%")
        lines.append(f"   Classificação: {row['risk_bucket']}")

        if "ml_risk_bucket" in row and row["ml_risk_bucket"] != row["risk_bucket"]:
            lines.append(
                f"   ⚠️ Divergência: "
                f"ML={row['ml_risk_bucket']} vs Regras={row['risk_bucket']}"
            )

    # Notas finais
    lines.append("")
    lines.append("=" * 80)
    lines.append("ℹ️ NOTAS METODOLÓGICAS")
    lines.append(" • Score de risco: escala 0–100 (quanto maior, maior o risco)")
    lines.append(" • Probabilidade: estimativa de default em 12 meses")
    lines.append(" • Classificação: Low (<33), Medium (33–66), High (>66)")
    lines.append(" • Divergências indicam necessidade de análise por comitê")

    return "\n".join(lines)
