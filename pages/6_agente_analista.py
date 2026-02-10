import streamlit as st
import google.generativeai as genai
import os
import hashlib
from utils.context_builder import build_risk_context  # Mantém sua função existente

st.title("🤖 Risk Analyst AI (Google Gemini)")


# # =============== DEBUG OPCIONAL (remova em produção) ===============
# if st.checkbox("🔍 Mostrar debug de secrets", key="debug"):
#     st.write("st.secrets existe?", hasattr(st, 'secrets'))
#     if hasattr(st, 'secrets'):
#         st.write("Chaves em st.secrets:", list(st.secrets.keys()))
#         st.write("google em st.secrets?", "google" in st.secrets)
#         if "google" in st.secrets:
#             st.write("API_KEY presente?", "GOOGLE_API_KEY" in st.secrets["google"])

# =============== VALIDAÇÃO DE DADOS ===============
df = st.session_state.get("final_results_df")
if df is None:
    st.warning("⚠️ Execute a predição na página 'Predição' antes de acessar esta análise.")
    st.page_link("pages/5_Predicao.py", label="Ir para página de Predição", icon="➡️")
    st.stop()

# =============== SELEÇÃO DE EMPRESA ===============
company = st.selectbox(
    "🔍 Selecione uma empresa para análise detalhada",
    ["Todas"] + sorted(df["Name"].dropna().unique().tolist()),
    help="Filtre os resultados para focar em uma empresa específica"
)

# =============== CONFIGURAÇÃO DA API ===============
try:
    # Prioriza Streamlit Secrets (recomendado para produção)
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    elif os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        st.error("🔑 Chave API não encontrada! Configure em:")
        st.code("Secrets do Streamlit: GOOGLE_API_KEY\nOU variável de ambiente: GOOGLE_API_KEY")
        st.stop()
except Exception as e:
    st.error(f"❌ Erro na configuração da API: {str(e)}")
    st.stop()

# =============== CONSTRUÇÃO DE CONTEXTO ===============
try:
    context = build_risk_context(
        df,
        None if company == "Todas" else company
    )
    
    # Validação crítica do contexto
    if not context or len(context.strip()) < 20:
        st.error("⚠️ Contexto de risco vazio ou inválido. Verifique os dados de predição.")
        st.stop()
except Exception as e:
    st.error(f"❌ Erro ao construir contexto: {str(e)}")
    st.stop()

# =============== RESET DE HISTÓRICO AO MUDAR CONTEXTO ===============
context_hash = hashlib.md5(context.encode()).hexdigest()
if "context_hash" not in st.session_state or st.session_state.context_hash != context_hash:
    st.session_state.context_hash = context_hash
    st.session_state.messages = []
    st.session_state.chat_session = None  # Força nova sessão

# =============== INSTRUÇÃO DO SISTEMA (OTIMIZADA) ===============
SYSTEM_INSTRUCTION = f"""Você é um analista sênior de risco de crédito corporativo do Banco Central.

REGRAS ESTRITAS:
1. USE EXCLUSIVAMENTE os dados do contexto abaixo. NUNCA invente números, nomes ou métricas.
2. Se informação não existir no contexto: "⚠️ Dado não disponível na análise realizada."
3. Responda com clareza para comitê de crédito: destaque risco alto/médio/baixo, principais drivers e recomendações objetivas.
4. Use prioritariamente a seção "EMPRESA SELECIONADA – ANÁLISE DETALHADA" quando o usuário perguntar sobre uma empresa específica.
5. Formate respostas com:
   - 📌 Resumo executivo (1 linha)
   - 🔍 Análise detalhada (tópicos)
   - 💡 Recomendação prática
6. Mantenha linguagem técnica mas acessível (evite jargões excessivos).

CONTEXTO DOS DADOS (ATUALIZADO PARA: {company if company != 'Todas' else 'TODAS AS EMPRESAS'}):
{context}"""

# =============== INICIALIZAÇÃO DO MODELO ===============
try:
    if "gemini_model" not in st.session_state:

        st.session_state.gemini_model = genai.GenerativeModel(
            model_name="gemini-flash-latest",
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40
            )
        )
    
    # Cria nova sessão de chat se necessário
    if st.session_state.chat_session is None:
        st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=[])
except Exception as e:
    st.error(f"❌ Falha ao inicializar modelo: {str(e)}")
    st.stop()

# =============== INTERFACE DE CHAT ===============
st.subheader("💬 Conversa com Analista de Risco")
st.caption(f"Contexto carregado: {company} | Modelo: Gemini Flash")

# Exibe histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.write(msg["content"])

# Input do usuário
if prompt := st.chat_input("Ex: 'Qual o principal risco desta empresa?', 'Compare com a média do setor', 'Justifique a classificação'"):
    # Adiciona mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
    
    # Processa resposta
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analisando dados de risco..."):
            try:
                response = st.session_state.chat_session.send_message(prompt)
                if not response.text.strip():
                    raise ValueError("Resposta vazia do modelo")
                
                st.write(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
                # Feedback sutil de uso
                st.caption(f"✅ Resposta gerada com {len(response.text)} caracteres")
                
            except Exception as e:
                error_msg = (
                    "⚠️ Limite de tokens excedido. Tente perguntas mais objetivas." if "429" in str(e) or "token" in str(e).lower() else
                    "⚠️ Erro temporário na API. Tente novamente em 10 segundos." if "503" in str(e) else
                    f"❌ Erro inesperado: {str(e)}"
                )
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# =============== DICA CONTEXTUAL ===============
with st.expander("💡 Dicas para melhores respostas"):
    st.markdown("""
    - **Seja específico**: "Qual o risco de inadimplência para a empresa X?"  
    - **Peça comparações**: "Como esta empresa se compara à média do setor?"  
    - **Solicite ações**: "Quais documentos complementares você recomenda analisar?"  
    - **Evite perguntas genéricas**: O modelo foca APENAS nos dados do contexto carregado.
    """)
