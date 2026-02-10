# utils/context_builder.py

import pandas as pd
from typing import Optional

def build_risk_context(df: pd.DataFrame, company_name: Optional[str] = None) -> str:
    """
    Constrói um contexto textual rico para análise de risco de crédito.
    
    Args:
        df: DataFrame com resultados da predição (obrigatório: colunas específicas)
        company_name: Nome da empresa para análise individual (None = portfólio completo)
    
    Returns:
        str: Contexto formatado para o modelo de IA
    """
    # Validação crítica
    required_cols = ["Name", "Rating", "final_risk_score", "risk_probability", "risk_bucket"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
    
    if df.empty:
        return "⚠️ Nenhum dado de risco disponível para análise."
    
    # Filtro por empresa (se aplicável)
    if company_name and company_name != "Todas":
        filtered_df = df[df["Name"] == company_name].copy()
        if filtered_df.empty:
            return f"⚠️ Empresa '{company_name}' não encontrada nos resultados."
        scope = f"Empresa individual: {company_name}"
    else:
        filtered_df = df.copy()
        scope = "Portfólio completo"
    
    # Garantir tipos numéricos
    filtered_df["final_risk_score"] = pd.to_numeric(filtered_df["final_risk_score"], errors="coerce")
    filtered_df["risk_probability"] = pd.to_numeric(filtered_df["risk_probability"], errors="coerce")
    
    # Remover NaNs críticos
    filtered_df = filtered_df.dropna(subset=["final_risk_score", "risk_probability", "risk_bucket"])
    
    if filtered_df.empty:
        return "⚠️ Dados insuficientes após filtragem (scores ou probabilidades ausentes)."
    
    # ========== ESTATÍSTICAS AGREGADAS ==========
    total = len(filtered_df)
    
    # Distribuição de risco (trata coluna como string/category)
    risk_dist = filtered_df["risk_bucket"].value_counts().to_dict()
    risk_dist_pct = {
        k: f"{v} ({v/total*100:.1f}%)"
        for k, v in sorted(risk_dist.items(), key=lambda x: ["Baixo", "Médio", "Alto"].index(x[0]) if x[0] in ["Baixo", "Médio", "Alto"] else 999)
    }
    
    # Divergências ML vs Regras (se coluna existir)
    disagreements = 0
    if "ml_risk_bucket" in filtered_df.columns:
        disagreements = (filtered_df["ml_risk_bucket"] != filtered_df["risk_bucket"]).sum()
        disagreement_pct = f"{disagreements} ({disagreements/total*100:.1f}%)"
    else:
        disagreement_pct = "Não aplicável (coluna ml_risk_bucket ausente)"
    
    # Métricas numéricas
    avg_score = filtered_df["final_risk_score"].mean()
    avg_prob = filtered_df["risk_probability"].mean() * 100  # Converter para %
    max_risk = filtered_df.loc[filtered_df["final_risk_score"].idxmax()] if not filtered_df.empty else None
    
    # ========== FORMATAÇÃO DA STRING DE CONTEXTO ==========
    context_lines = []
    
    context_lines.append("📊 ANÁLISE DE RISCO DE CRÉDITO")
    context_lines.append("=" * 70)
    context_lines.append(f"📌 Escopo: {scope}")
    context_lines.append(f"📅 Total de empresas analisadas: {total}")
    context_lines.append("")
    
    # Distribuição de risco
    context_lines.append("📈 Distribuição de Risco:")
    for bucket, count_pct in risk_dist_pct.items():
        context_lines.append(f"   • {bucket}: {count_pct}")
    context_lines.append("")
    
    # Divergências
    context_lines.append(f"⚠️  Divergências ML vs Regras: {disagreement_pct}")
    context_lines.append("")
    
    # Métricas agregadas
    context_lines.append("📉 Métricas Agregadas:")
    context_lines.append(f"   • Score médio de risco: {avg_score:.2f}/100")
    context_lines.append(f"   • Probabilidade média de default: {avg_prob:.2f}%")
    if max_risk is not None:
        context_lines.append(f"   • Maior risco identificado: {max_risk['Name']} (Score: {max_risk['final_risk_score']:.2f})")
    context_lines.append("")
    
    # ========== EMPRESAS DETALHADAS (Top 5 ou única) ==========
    if company_name and company_name != "Todas":
        # Modo empresa única
        company = filtered_df.iloc[0]
        context_lines.append(f"🏢 DADOS DA EMPRESA: {company['Name']}")
        context_lines.append("-" * 70)
        context_lines.append(f"   • Rating: {company.get('Rating', 'N/A')}")
        context_lines.append(f"   • Score de Risco Final: {company['final_risk_score']:.2f}/100")
        context_lines.append(f"   • Probabilidade de Default (12m): {company['risk_probability']*100:.2f}%")
        context_lines.append(f"   • Classificação de Risco: {company['risk_bucket']}")
        if "ml_risk_bucket" in company:
            context_lines.append(f"   • Bucket ML: {company['ml_risk_bucket']}")
            context_lines.append(f"   • Bucket Regras: {company['risk_bucket']}")
            if company['ml_risk_bucket'] != company['risk_bucket']:
                context_lines.append("   ⚠️  ALERTA: Divergência entre modelos!")
    else:
        # Modo portfólio (Top 5)
        context_lines.append("🏢 TOP 5 EMPRESAS POR RISCO (Score Descendente):")
        context_lines.append("-" * 70)
        
        top5 = filtered_df.nlargest(5, "final_risk_score")
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            context_lines.append(f"{idx}. {row['Name']}")
            context_lines.append(f"   • Rating: {row.get('Rating', 'N/A')}")
            context_lines.append(f"   • Score: {row['final_risk_score']:.2f}/100")
            context_lines.append(f"   • Prob. Default: {row['risk_probability']*100:.2f}%")
            context_lines.append(f"   • Classificação: {row['risk_bucket']}")
            if "ml_risk_bucket" in row and row["ml_risk_bucket"] != row["risk_bucket"]:
                context_lines.append(f"   ⚠️  Divergência: ML={row['ml_risk_bucket']} vs Regras={row['risk_bucket']}")
            context_lines.append("")
    
    # ========== NOTAS IMPORTANTES ==========
    context_lines.append("=" * 70)
    context_lines.append("ℹ️  NOTAS PARA ANÁLISE:")
    context_lines.append("   • Score de risco: 0-100 (quanto maior, maior o risco de default)")
    context_lines.append("   • Probabilidade: estimativa de inadimplência em 12 meses")
    context_lines.append("   • Classificações: Baixo (<30), Médio (30-70), Alto (>70) - ajuste conforme sua modelagem")
    context_lines.append("   • Divergências indicam casos que requerem análise manual por comitê de crédito")
    
    return "\n".join(context_lines)