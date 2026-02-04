import streamlit as st
import os
from dotenv import load_dotenv

from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Sahi Import Paths ye hain:
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# --------------------------------------

load_dotenv()

st.set_page_config(page_title="C.B.R.I ProcureBuddy", page_icon="🤖")
st.title("🤖 C.B.R.I Purchase Assistant")
st.caption("powered by Groq (Llama 3) & GFR Rules")

#2. Database & Model Setup

@st.cache_resource
def get_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists("./chroma_db"):
        st.error("Databases does not found!!! Please run ingest.py first...")
        return None,None

    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return retriever, client


retriever, client = get_resources()

# ---------------------------------------------- CHAT USER INTERFACE --------------------------------------- #
if retriever and client:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask about CBRI purchase rules (GFR 2017)...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("📘 Analyzing GFR Rules..."):

                # -------- RETRIEVE CONTEXT --------
                docs = retriever.get_relevant_documents(user_input)
                context = "\n\n".join(d.page_content for d in docs)

                # -------- GROQ CALL --------
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
- NOT word-by-word translation
- Short, bulleted, practical
- Clearly mention:
  • Purchase value
  • Applicable slab
  • Whether committee is required (Yes / No)
- No jokes, no fillers, no repetition

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

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )