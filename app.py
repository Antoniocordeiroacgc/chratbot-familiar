"""
Chatbot de Apoio Familiar - versão web (Streamlit)
----------------------------------------------------
Assistente de orientação (NÃO substitui psicólogo, terapeuta,
assistente social ou profissional de saúde) para:
  - Finanças familiares
  - Crise conjugal
  - Dificuldades com filhos
  - Jogo compulsivo (ludopatia)
  - Detecção de risco de suicídio, com encaminhamento imediato
"""

import os
import re
import streamlit as st
import google.generativeai as genai

# ============================================================
# CONFIGURAÇÃO DA CHAVE DE API
# ============================================================
# Em produção (Streamlit Community Cloud), a chave vem de st.secrets,
# configurada no painel do site (nunca fica escrita no código).
# Em desenvolvimento local, pode vir de uma variável de ambiente ou de
# .streamlit/secrets.toml. O try/except evita erro quando não existe
# nenhum secrets.toml localmente (comportamento do Streamlit).
try:
    API_KEY = st.secrets.get("GOOGLE_API_KEY")
except Exception:
    API_KEY = None

if not API_KEY:
    API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error(
        "Chave de API não configurada. Defina GOOGLE_API_KEY nos 'Secrets' "
        "do Streamlit Cloud (ou como variável de ambiente localmente)."
    )
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-3.6-flash")

# ============================================================
# PERSONA / PROMPT DO SISTEMA
# ============================================================
SYSTEM_INSTRUCTION = """
Você é um assistente virtual de apoio ao bem-estar familiar, projetado para ajudar
pessoas e famílias que enfrentam dificuldades em quatro áreas principais:

1. FINANÇAS FAMILIARES: dívidas, orçamento doméstico, brigas por dinheiro,
   dificuldade em pagar contas, planejamento financeiro básico.
2. RELACIONAMENTO/CASAMENTO: conflitos conjugais, comunicação, crises no
   relacionamento, possibilidade de separação.
3. FILHOS: dificuldades na criação, conflitos entre pais e filhos, comportamento
   infantil/adolescente, dinâmica familiar.
4. JOGO COMPULSIVO (LUDOPATIA): impacto de apostas/jogos de azar na família e nas
   finanças, sinais de dependência.

DIRETRIZES OBRIGATÓRIAS:
1. Empatia e acolhimento: linguagem calorosa, sem julgamento, escuta ativa.
2. Orientação, NUNCA diagnóstico ou tratamento: dê informações gerais e passos
   práticos, mas nunca diagnostique, nunca dê aconselhamento jurídico ou
   financeiro formal, e nunca substitua um profissional licenciado.
3. Encaminhamento profissional sempre que perceber sofrimento significativo,
   dependência, conflito familiar grave, violência doméstica ou risco a alguém.
4. Linguagem simples, sem jargões.
5. Foco em passos práticos e resiliência.
6. Privacidade: você não guarda dados entre sessões.
7. Se o assunto sair do escopo de apoio familiar/financeiro/emocional,
   redirecione com gentileza.
8. Nunca incentive apostas, jogos de azar ou empréstimos informais.

Sempre que a conversa começar, apresente-se brevemente e deixe claro que você é
um apoio inicial, e que em situações mais sérias buscar um profissional é o
caminho mais seguro.
"""

# ============================================================
# DETECÇÃO DE CRISE POR CATEGORIA
# ============================================================
CATEGORIAS_CRISE = {
    "suicidio": {
        "palavras": [
            "quero me matar", "não aguento mais viver", "pensando em suicídio",
            "tirar minha vida", "acabar com tudo", "não vejo saída",
            "melhor eu não existir", "vou me matar", "suicida", "depressão grave",
            "sem vontade de viver",
        ],
        "mensagem": """
**Estou muito preocupado(a) com o que você disse.** Você não precisa passar por
isso sozinho(a). Por favor, procure ajuda agora:

- **CVV** (Centro de Valorização da Vida): ligue **188** (gratuito, 24h) ou acesse cvv.org.br
- **SAMU**: ligue **192**
- Vá a um pronto-socorro ou hospital mais próximo

Buscar um psicólogo ou psiquiatra o quanto antes é fundamental. Sua vida é muito importante.
""",
    },
    "financeira": {
        "palavras": [
            "não consigo pagar as contas", "estou endividado", "dívida",
            "vou perder minha casa", "sem dinheiro para comida", "cheque especial",
            "nome sujo", "não tenho dinheiro", "empréstimo", "desempregado",
            "crise financeira", "sem condições de pagar",
        ],
        "mensagem": """
Percebo que a situação financeira está pesando bastante. Posso ajudar a organizar
ideias sobre orçamento e prioridades. Para uma solução mais completa, considere
também:
- Um **psicólogo**, se o peso emocional estiver forte
- **Procon** ou **Defensoria Pública**, para negociação de dívidas
- Serviços de **assistência social** do seu município
""",
    },
    "conjugal": {
        "palavras": [
            "crise no casamento", "vou me separar", "não aguento mais meu marido",
            "não aguento mais minha esposa", "traição", "quero me divorciar",
            "brigamos muito", "relacionamento abusivo", "meu casamento está acabando",
        ],
        "mensagem": """
Relacionamentos passam por momentos difíceis, e o que você sente é válido. Em
crises mais profundas, um **terapeuta de casais** ou **psicólogo** pode ajudar
muito. Se houver qualquer sinal de violência doméstica, procure ajuda pelo
**180** (Central de Atendimento à Mulher) ou **190** (Polícia).
""",
    },
    "filhos": {
        "palavras": [
            "problema com meu filho", "problema com minha filha", "meu filho não me obedece",
            "briga com os filhos", "meu filho se droga", "meu filho fugiu de casa",
            "não sei como lidar com meu filho", "comportamento agressivo do meu filho",
        ],
        "mensagem": """
Cuidar de filhos em momentos difíceis é desafiador. Um **psicólogo infantil ou
familiar** pode oferecer acompanhamento mais próximo para a situação específica.
""",
    },
    "jogos": {
        "palavras": [
            "não consigo parar de apostar", "aposta", "vício em jogo",
            "gastei tudo apostando", "jogo do tigrinho", "casa de apostas",
            "ludopatia", "perdi dinheiro jogando", "bet",
        ],
        "mensagem": """
O que você descreveu tem características de **jogo compulsivo (ludopatia)**, um
transtorno reconhecido que afeta finanças e família. Você não está sozinho(a).
Considere buscar:
- Um **psicólogo especializado em dependências**
- Grupos de apoio como **Jogadores Anônimos** (jogadoresanonimos.org.br)
""",
    },
}


def detectar_crise(pergunta: str):
    ordem_prioridade = ["suicidio", "financeira", "conjugal", "filhos", "jogos"]
    for categoria in ordem_prioridade:
        for palavra in CATEGORIAS_CRISE[categoria]["palavras"]:
            if palavra in pergunta:
                return categoria, CATEGORIAS_CRISE[categoria]["mensagem"]
    return None, None


def limpar_texto_markdown(texto: str) -> str:
    texto = re.sub(r"\s+\n", "\n", texto)
    return texto.strip()


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.set_page_config(page_title="Apoio Familiar", page_icon="💬", layout="centered")

with st.sidebar:
    st.markdown("### 🆘 Precisa de ajuda urgente?")
    st.markdown(
        "- **CVV**: ligue **188** (24h, gratuito)\n"
        "- **SAMU**: ligue **192**\n"
        "- **Central da Mulher**: ligue **180**\n"
        "- **Polícia**: ligue **190**"
    )
    st.divider()
    st.markdown("### Sobre")
    st.caption(
        "Este assistente oferece apoio inicial em finanças familiares, "
        "relacionamento, filhos e jogo compulsivo. Não substitui "
        "acompanhamento psicológico profissional."
    )

st.title("💬 Chatbot de Apoio Familiar")
st.caption(
    "Um apoio inicial para finanças familiares, relacionamento, filhos e jogo "
    "compulsivo. Não substitui acompanhamento profissional."
)

AVATAR_USUARIO = "🧑"
AVATAR_ASSISTENTE = "💬"

# Histórico de conversa por sessão (some ao fechar a aba/navegador)
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(
        history=[
            {"role": "user", "parts": [SYSTEM_INSTRUCTION]},
            {
                "role": "model",
                "parts": [
                    "Entendido. Estou pronto para apoiar em finanças familiares, "
                    "relacionamento, filhos e jogo compulsivo, sempre recomendando "
                    "ajuda profissional quando necessário."
                ],
            },
        ]
    )
    st.session_state.mensagens_exibidas = []

# Reexibe o histórico já mostrado nesta sessão
for autor, texto in st.session_state.mensagens_exibidas:
    avatar = AVATAR_USUARIO if autor == "user" else AVATAR_ASSISTENTE
    with st.chat_message(autor, avatar=avatar):
        st.markdown(texto)

pergunta_usuario = st.chat_input("Digite sua mensagem...")

if pergunta_usuario:
    with st.chat_message("user", avatar=AVATAR_USUARIO):
        st.markdown(pergunta_usuario)
    st.session_state.mensagens_exibidas.append(("user", pergunta_usuario))

    categoria, mensagem_crise = detectar_crise(pergunta_usuario.lower())

    if categoria:
        with st.chat_message("assistant", avatar=AVATAR_ASSISTENTE):
            st.markdown(mensagem_crise)
        st.session_state.mensagens_exibidas.append(("assistant", mensagem_crise))
        if categoria == "suicidio":
            st.warning(
                "Se você está em risco imediato, ligue agora para o CVV (188) ou SAMU (192)."
            )
    else:
        try:
            with st.spinner("Pensando..."):
                resposta = st.session_state.chat.send_message(pergunta_usuario)
                # Verifica se a resposta veio vazia/bloqueada antes de tentar ler .text
                if not resposta.candidates or not resposta.candidates[0].content.parts:
                    raise ValueError("Resposta vazia ou bloqueada pelo modelo")
                resposta_limpa = limpar_texto_markdown(resposta.text)
            with st.chat_message("assistant", avatar=AVATAR_ASSISTENTE):
                st.markdown(resposta_limpa)
            st.session_state.mensagens_exibidas.append(("assistant", resposta_limpa))
        except Exception as e:
            # Fallback acolhedor: como o tema é sensível (tristeza, saúde mental,
            # família), nunca mostramos um erro técnico frio para o usuário.
            fallback = """
Percebo que você está passando por um momento difícil. No momento não consegui
processar sua mensagem, mas isso não diminui o que você está sentindo.

Se quiser, tente reformular a pergunta de um jeito um pouco diferente. E lembre-se:
conversar com um psicólogo pode fazer muita diferença para entender o que você
está sentindo - isso não é algo que um chatbot consegue diagnosticar.
"""
            with st.chat_message("assistant", avatar=AVATAR_ASSISTENTE):
                st.markdown(fallback)
            st.session_state.mensagens_exibidas.append(("assistant", fallback))

# ============================================================
# RODAPÉ FIXO
# ============================================================
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        [data-testid="stStatusWidget"] {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        footer {visibility: hidden;}
        .rodape-fixo {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #F0F4F2;
            color: #2C3E3A;
            text-align: center;
            padding: 6px 0;
            font-size: 12px;
            z-index: 999;
            border-top: 1px solid #d9e0dd;
        }
    </style>
    <div class="rodape-fixo">
        CRIAR.IA TECNOLOGIA - Todos os direitos reservados
    </div>
    """,
    unsafe_allow_html=True,
)
