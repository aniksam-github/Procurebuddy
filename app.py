import streamlit as st
import pandas as pd
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
    text = text.replace(",", "").lower()

    # Handle ₹, rs, inr, etc.
    patterns = [
        r'₹\s*([\d]+)',
        r'rs\.?\s*([\d]+)',
        r'inr\s*([\d]+)',
        r'worth\s*([\d]+)',
        r'amount\s*([\d]+)',
        r'([\d]{4,})'  # fallback: any big number
    ]

    for p in patterns:
        match = re.search(p, text)
        if match:
            return int(match.group(1))

    return None


def is_purchase_query(text):
    keywords = [
        "purchase", "buy", "procure", "lena", "khareed", "item",
        "worth", "amount", "price", "rs", "₹", "rupaye"
    ]
    text = text.lower()
    return any(k in text for k in keywords)

def is_table_query(text):
    keywords = [
        "table", "slab", "slab wise", "slab-wise", "cost wise", "cost-wise",
        "procedure", "process table", "overview", "chart"
    ]
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


# ------------------ EXTRA FUNCTIONALITIES ------------------
import pandas as pd

def show_process_table():
    data = [
        ["Up to ₹2,00,000", "Direct Purchase", "No", "No", "-", "Indent + Certificate", "Indent → Approval → Purchase"],
        ["₹2,00,001 – ₹10,00,000", "LPC", "No (Market survey)", "Yes", "LPC", "Indent + LPC Certificate", "Indent → LPC → Approval → Purchase"],
        ["₹10,00,001 – ₹25,00,000", "LTE", "Yes (Limited)", "Yes", "T&PC", "Indent + NIT + Eval Report", "Indent → Tender → T&PC → PO"],
        ["Above ₹25,00,000", "Open / Global Tender", "Yes (Open)", "Yes", "T&PC + BOC", "Indent + NIT + Bid Minutes", "Indent → Tender → Committees → PO"],
    ]

    df = pd.DataFrame(data, columns=[
        "Cost Slab (₹)", "Procurement Mode", "Quotation / Tender",
        "Committee Required", "Which Committee", "Key Documents", "Short Process"
    ])

    st.table(df)




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
            table_intent = is_table_query(user_input)

            # Case 0: User wants an overview table
            if table_intent:
                st.markdown("### 📊 CBRI / CSIR Purchase Process – Cost Slab Wise")
                show_process_table()
                answer = "📊 CBRI / CSIR Purchase Process – Cost Slab Wise table shown."

            # Case 1: User is asking about purchase but amount is missing
            elif purchase_intent and amount is None:
                answer = (
                    "🙂 Samajh aa raha hai aap purchase ke baare me pooch rahe ho, "
                    "lekin amount mention nahi hua.\n\n"
                    "👉 Please bata do: item ki estimated cost kitni hai? "
                    "(jaise ₹25000, ₹3 lakh, ₹10,00,000)"
                )
                st.markdown(answer)

            # Case 2: Not a purchase-related query at all
            elif not purchase_intent:
                answer = (
                    "👋 Hi! Main CBRI / CSIR Purchase Rules (GFR 2017 + CSIR Manual) ke hisaab se help karta hoon.\n\n"
                    "📝 Tum kaise bhi pooch sakte ho, jaise:\n"
                    "• I want to purchase an item worth ₹25000\n"
                    "• ₹35000 ka item lena hai, process kya hoga?\n"
                    "• Show me a table of procurement process as per cost slabs in CBRI\n\n"
                    "👉 Bas amount mention kar do, ya bolo 'table dikha do' 🙂"
                )
                st.markdown(answer)

            # Case 3: Valid purchase query → continue to RAG + LLM
            else:
                st.markdown("### 📊 CBRI / CSIR Purchase Process (Cost-wise)")
                show_process_table()

                docs = retriever.get_relevant_documents(user_input)
                context = "\n\n".join(d.page_content for d in docs)

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": """
You are ProcureBuddy, an expert procurement assistant for CBRI (CSIR), strictly based on:

- General Financial Rules (GFR) 2017 (updated till 31 July 2025)
- CSIR Manual on Procurement of Goods 2019 (MPG 2019)
- Special provisions for Scientific Departments (DoE OM)
- The user may ask in Hindi, English, or Hinglish.
- You must infer intent and amount even if the question is informal or incomplete, and ask a clarification only if amount is missing.


STRICT RULES (MANDATORY):

1. Use ONLY the provided context from:
   - GFR 2017
   - CSIR Manual on Procurement of Goods 2019
   - Official OMs provided in the knowledge base
   Do NOT use outside knowledge or assumptions.

2. Always first extract the EXACT purchase value from the user query.
   - Never round, approximate, split, or reinterpret the amount.

3. Apply the CORRECT CSIR / GFR based procurement logic:

   - Up to ₹2,00,000 (for scientific departments):
     → Direct Purchase
     → NO committee required

   - ₹2,00,001 to ₹10,00,000:
     → Local Purchase Committee (LPC)
     → Committee IS required

   - Above ₹10,00,000 up to ₹25,00,000:
     → Limited Tender Enquiry (LTE)
     → Technical & Purchase Committee (T&PC) IS required

   - Above ₹25,00,000:
     → Open / Global Tender
     → Technical & Purchase Committee (T&PC) IS required

4. You MUST consider CSIR Manual on Procurement of Goods 2019 for:
   - Committee requirements
   - Tender modes
   - Procurement procedures

5. Do NOT say “This information is not found in GFR 2017” if the procedure is defined
   in CSIR Manual on Procurement of Goods 2019.

6. Clearly mention in the answer:
   - Purchase value
   - Applicable procurement mode
   - Whether committee is required (Yes/No)
   - Which committee (if applicable)

7. Item type (laptop, equipment, consumable, emergency, single vendor, etc.)
   does NOT change the basic committee requirement unless explicitly stated in the rules.

8. Artificial splitting of purchase to bypass rules is NOT allowed.

ANSWER STYLE:

- Simple Hinglish (easy Hindi + English)
- Short, clear, audit-friendly
- Structured output, for example:
  • Purchase value
  • Applicable procurement mode
  • Committee required or not
  • Brief explanation

If the required procedure is genuinely not present in the provided CSIR/GFR context,
then and only then say:
"This information is not found in the provided rules."


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
                st.markdown("---")
                st.markdown("### 🧾 Aapke case ka summary")
                st.markdown(answer)


    # Save assistant message
if "answer" in locals() and answer.strip():
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_history(st.session_state.messages)

    # Unlock input
    st.session_state.busy = False
    st.rerun()
