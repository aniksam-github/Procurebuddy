import streamlit as st
import os
import re
import json
from dotenv import load_dotenv

from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="C.B.R.I ProcureBuddy", page_icon="🤖")
st.title("🤖 C.B.R.I Purchase Assistant")
st.caption("powered by Groq (Llama 3) & GFR Rules")

# ------------------ HELPERS ------------------

def extract_amount(text):
    match = re.search(r'(\d{3,})', text.replace(",", ""))
    if match:
        return int(match.group(1))
    return None

def is_purchase_query(text):
    keywords = ["purchase", "buy", "procure", "item", "rs", "₹", "worth", "amount", "price"]
    text = text.lower()
    return any(k in text for k in keywords)

# ------------------ CHAT HISTORY -----------------------
HISTORY_FILE = "chat_history.json"

def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ----------------- SIDEBAR -----------------
with st.sidebar:
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.session_state.pending_input = None
        st.session_state.busy = False
        st.rerun()

    if st.button("🕘 Load Old Chats"):
        st.session_state.messages = load_history()
        st.rerun()

    if st.button("🗑️ Clear Chats"):
        st.session_state.messages = []
        save_history([])
        st.rerun()

# ------------------ SESSION STATE INIT ------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "busy" not in st.session_state:
    st.session_state.busy = False

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# ------------------ DB & MODEL ------------------
@st.cache_resource
def get_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists("./chroma_db"):
        import ingest
        ingest.create_vector_db()

    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return retriever, client

retriever, client = get_resources()

# ------------------ SHOW CHAT HISTORY ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ INPUT ------------------
user_input = st.chat_input(
    "Ask about purchase rules (GFR 2017)...",
    disabled=st.session_state.busy
)

if user_input and not st.session_state.busy:
    st.session_state.busy = True
    st.session_state.pending_input = user_input
    st.rerun()

# ------------------ PROCESS QUEUED MESSAGE ------------------
if st.session_state.pending_input and retriever and client:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("📘 Analyzing GFR Rules..."):

            amount = extract_amount(user_input)
            purchase_intent = is_purchase_query(user_input)

            if amount is None or not purchase_intent:
                # Not a purchase query
                answer = (
                    "👋 Hi! Main CBRI Purchase Rules (GFR 2017) ke hisaab se help karta hoon.\n\n"
                    "👉 Aise poochho:\n"
                    "- I want to purchase an item worth ₹25000\n"
                    "- ₹35000 ka item lena hai, process kya hoga?\n"
                )
                st.markdown(answer)

            else:
                # Valid purchase query → RAG + LLM
                docs = retriever.get_relevant_documents(user_input)
                context = "\n\n".join(d.page_content for d in docs)

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": """
You are ProcureBuddy, an expert procurement assistant for CBRI (CSIR), strictly based on GFR 2017.

STRICT RULES (MANDATORY):

1. Use ONLY the provided context from GFR 2017.
   Do NOT use outside knowledge, assumptions, or personal judgment.

2. First, identify the EXACT purchase value slab from the given amount.
   - NEVER round, approximate, split, or reinterpret the amount.
   - Any amount greater than ₹25,000, even by ₹0.01, falls in the ₹25,001 slab.

3. Purchase value slabs (STRICT):
   - Up to ₹25,000 → NO committee required
   - ₹25,001 to ₹2,50,000 → Local Purchase Committee (LPC) is MANDATORY

4. Committee rules:
   - If amount is ₹25,000 or below:
     • Do NOT mention LPC or Purchase Committee
     • Clearly state: "NO committee is required"
   - If amount is above ₹25,000:
     • Clearly mention that LPC is mandatory
     • Briefly state LPC’s role (market survey, reasonableness of price)

5. Item type (laptop, emergency, service, single vendor, urgency, etc.)
   does NOT change committee requirements.

6. Artificial splitting of purchase to avoid rules is NOT allowed.

7. Mention GeM portal ONLY if it is present in the provided context.
   Do NOT assume GeM applicability.

ANSWER STYLE:

- Clear, natural Hinglish (simple Hindi + English)
- Short, bulleted, practical
- Clearly mention:
  • Purchase value
  • Applicable slab
  • Whether committee is required (Yes / No)

IF INFORMATION IS MISSING:

- If the answer is NOT clearly present in the provided context, reply EXACTLY:
  "This information is not found in GFR 2017."
"""
                        },
                        {
                            "role": "user",
                            "content": f"""
Context:
{context}

Question:
{user_input}
"""
                        }
                    ],
                    temperature=0.3
                )

                answer = response.choices[0].message.content
                st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_history(st.session_state.messages)

    # Unlock input
    st.session_state.busy = False
    st.rerun()
