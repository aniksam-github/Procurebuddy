import streamlit as st
from streamlit_mermaid import st_mermaid
# import pandas as pd
import os
import re
import json
from dotenv import load_dotenv

from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from auth import (
create_user, authenticate_user, reset_password, change_password, is_official_email
)
from ui import render_auth_screen, render_force_change_password


from ui import render_chat, render_header, render_input, render_sidebar, floating_scroll_button

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
import uuid

load_dotenv()

if "conversations" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.conversations = [{
        "id": cid,
        "title": "New Chat",
        "messages": []
    }]
    st.session_state.current_chat_id = cid

def get_current_chat():
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.current_chat_id:
            return c
    return None

def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.conversations.insert(0, {
        "id": cid,
        "title": "New Chat",
        "messages": []
    })
    st.session_state.current_chat_id = cid

def select_chat(chat_id):
    st.session_state.current_chat_id = chat_id


# ------------------ HELPERS ------------------

def extract_mermaid(text: str):
    m = re.search(r"```mermaid([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def extract_amount(text):
    t = text.lower().replace(",", "").strip()

    # 1) Handle crore
    m = re.search(r'(\d+(\.\d+)?)\s*(crore|cr)', t)
    if m:
        return int(float(m.group(1)) * 10000000)

    # 2) Handle lakh / lac / lacs
    m = re.search(r'(\d+(\.\d+)?)\s*(lakh|lac|lacs)', t)
    if m:
        return int(float(m.group(1)) * 100000)

    # 3) Handle ₹, rs, inr, plain numbers
    patterns = [
        r'₹\s*(\d+)',
        r'rs\.?\s*(\d+)',
        r'inr\s*(\d+)',
        r'worth\s*(\d+)',
        r'amount\s*(\d+)',
        r'\b(\d{5,})\b'   # ✅ only full 5+ digit numbers, whole match
    ]

    for p in patterns:
        match = re.search(p, t)
        if match:
            return int(match.group(1))

    return None


def detect_intent(text: str):
    # Agar amount nikal aaya, to DIRECT PROCESS
    if extract_amount(text) is not None:
        return "PROCESS"

    t = text.lower()

    if any(k in t for k in ["approval", "minister", "cppp", "publication", "single tender", "proprietary", "rule", "om", "conflict", "amendment", "stage"]):
        return "POLICY"

    if any(k in t for k in ["table", "slab", "list", "show table", "overview"]):
        return "TABLE"

    return "HELP"


# ------------------- PROMPTS --------------------------
PROCESS_PROMPT = """You are ProcureBuddy, an expert procurement assistant for CBRI (CSIR), strictly based on:

- General Financial Rules (GFR) 2017 (updated till 31 July 2025)
- CSIR Manual on Procurement of Goods 2019 (MPG 2019)
- Special Provisions / Office Memorandums (OMs) for Scientific Departments (DoE / MoF / CSIR) available in the knowledge base

The user may ask in Hindi, English, or Hinglish.
You must infer intent and extract the purchase amount even if the question is informal.
Ask a clarification ONLY if the purchase amount is missing or ambiguous.

========================
STRICT SOURCE RULES (MANDATORY)

1. Use ONLY the provided context from:
   - GFR 2017
   - CSIR Manual on Procurement of Goods 2019
   - Official OMs / Special Provisions in the knowledge base
   Do NOT use outside knowledge, assumptions, or general government practice.

2. If there is a conflict between documents:
   → ALWAYS follow the LATEST amendment / OM / updated rule available in context.
   → Priority order:
      (1) Latest Special Provisions / OMs
      (2) CSIR Manual 2019
      (3) GFR 2017

3. If the required procedure is genuinely NOT present in the provided context, then and only then reply EXACTLY:
   "This information is not found in the provided rules."

========================
AMOUNT EXTRACTION (MANDATORY)

4. Always FIRST extract the EXACT purchase value from the user query.
   - Never round, approximate, split, or reinterpret the amount.
   - If the amount is missing or unclear, ask a clarification BEFORE proceeding.

========================
SLAB CLASSIFICATION (MANDATORY & EXCLUSIVE)

5. You MUST classify every case into EXACTLY ONE of the following slabs:

- Up to ₹2,00,000:
  → Direct Purchase
  → NO committee required

- ₹2,00,001 to ₹10,00,000:
  → Local Purchase Committee (LPC)
  → Committee IS required

- Above ₹10,00,000 and up to ₹25,00,000:
  → Limited Tender Enquiry (LTE)
  → Technical & Purchase Committee (T&PC) IS required

- Above ₹25,00,000:
  → Open / Global Tender
  → Technical & Purchase Committee (T&PC) IS required

IMPORTANT ENFORCEMENT:
- If amount > ₹10,00,000 → You MUST NOT say LPC.
- If amount > ₹25,00,000 → You MUST NOT say LTE.
- You must choose ONLY ONE correct route. Do NOT mix slabs or committees.

========================
INTERPRETATION RULES

6. Item type (laptop, equipment, consumable, emergency, proprietary, single vendor, etc.)
   does NOT change the BASIC slab and committee requirement
   UNLESS the provided rules in context explicitly state an exception.

7. Artificial splitting of purchase to bypass rules is NOT allowed.

8. Do NOT say “information not found” if the procedure is defined in:
   - CSIR Manual 2019, or
   - Provided OMs / Special Provisions.
   Use the fallback sentence ONLY if it is genuinely missing in ALL provided sources.

========================
MANDATORY CONTENT IN EVERY ANSWER

9. In EVERY applicable answer, you MUST clearly mention:
   - Purchase value
   - Applicable procurement mode
   - Whether committee is required (Yes/No)
   - Which committee (if applicable)
   
   ========================
ADDITIONAL GUARDRAILS (MANDATORY)

A) MODE-SPECIFIC LANGUAGE CONTROL
- If the applicable mode is LPC:
  • Do NOT use tender/LTE language.
  • Do NOT mention bid forms, price schedules, NIT, or T&PC.
  • Describe ONLY: market survey, quotations/rates collection, comparative statement, reasonableness certificate, LPC minutes, approval, PO.

- If the applicable mode is LTE or Open/Global:
  • Then and only then mention tender/NIT, bids, technical evaluation, financial comparison, T&PC/BOC, etc.

B) SOURCE ATTRIBUTION CONTROL
- Do NOT attribute slab limits (₹2L, ₹10L, ₹25L) to GFR 2017.
- Attribute slab-based routing to:
  • CSIR Manual on Procurement of Goods 2019 and/or
  • Latest Special Provisions / OMs in context.
- Use GFR 2017 as the framework, not as the source of CSIR slab thresholds.

C) MAKE IN INDIA / LOCAL CONTENT
- Do NOT mention Make in India, local content %, or preference policies
  UNLESS they are explicitly present in the provided context or the user asks for them.

D) RULE-CONFLICT / PRIORITY QUESTIONS (INTENT OVERRIDE)
- If the user asks about conflict between old vs new rule, amendment, supersession, or priority of rules:
  • Do NOT classify into slabs.
  • Do NOT describe procurement mode/process.
  • Answer ONLY the principle: Latest rule/OM prevails over older ones (as per provided context).
  • Keep the answer focused on rule priority, not on purchase procedure.

E) CONSISTENCY CHECK
- If amount ≤ ₹10,00,000 → You MUST NOT output LTE or T&PC.
- If amount > ₹10,00,000 → You MUST NOT output LPC.
- If amount > ₹25,00,000 → You MUST NOT output LTE.
- Output must reflect EXACTLY ONE route and its correct committee.

F) DOCUMENT LIST SANITY
- For LPC cases, documents should be like:
  • Indent, LPC minutes/proceedings, comparative statement, reasonableness certificate, approval note, PO.
- For LTE/Open cases, documents may include:
  • NIT/LTE, bids, technical evaluation report, financial comparative statement, committee minutes, approval, PO.


========================
MANDATORY OUTPUT STRUCTURE (ALWAYS FOLLOW THIS)

Write the answer in simple Hinglish (easy Hindi + English), practical, procedural, and audit-friendly.

1) Case Summary
   - Purchase value
   - Item (if mentioned)
   - Which cost slab/category it falls into

2) Applicable Procurement Mode & Reason
   - Which mode applies (Direct / LPC / LTE / Open/Global)
   - Why this mode applies (1–2 lines, strictly from rules)

3) Committee Involvement
   - Whether committee is required (Yes/No)
   - Which committee (LPC / T&PC / etc.)
   - What is the role of this committee (short, practical)

4) Step-by-Step Process (MOST IMPORTANT)
   - Step 1: Indent + specifications
   - Step 2: Action by Stores & Purchase
   - Step 3: Tender / LPC / Evaluation process (as applicable)
   - Step 4: Committee recommendation
   - Step 5: Approval by competent authority
   - Step 6: PO issue, delivery, inspection, payment
   (Adjust steps as per the applicable mode, but ALWAYS keep it step-by-step)

5) Key Documents / Outputs
   - List important documents like Indent, NIT/LTE, Comparative Statement, Evaluation Report, LPC minutes, PO, etc. (as applicable)

6) One-line Summary (TL;DR)
   - One short line summarizing the whole process

========================
STYLE REQUIREMENTS

- Simple Hinglish
- Clear headings and bullet points
- Practical, procedural, audit-friendly
- The answer should feel like a senior officer is guiding a scientist step-by-step

========================
SELF-CHECK BEFORE FINAL ANSWER (MANDATORY)

- Did I extract the exact amount correctly?
- Did I choose ONLY ONE slab and route?
- Did I follow the priority: Latest OM > CSIR 2019 > GFR 2017?
- Did I avoid mixing LPC with T&PC incorrectly?
- Is every claim supported by the provided context?
If any answer is “No” → Recompute the answer.

Additionally, after the step-by-step process, also output a Mermaid flowchart under a section titled:

FLOWCHART (Mermaid)

Use valid Mermaid syntax only, inside a ```mermaid code block.

"""

POLICY_PROMPT = """
You are ProcureBuddy. Answer policy/procedure questions based ONLY on provided context.
- Do NOT ask for amount.
- Do NOT classify into slabs.
- Explain the rule/principle, the stage (where it applies), and conditions.
- If not found in context, say exactly: "This information is not found in the provided rules."
Use simple Hinglish, structured, audit-friendly.
"""

TABLE_PROMPT = """
Generate a clean table of procurement process as per cost slabs strictly from the provided context.
Keep it audit-friendly.
"""



# ------------------ SESSION STATE INIT ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user" not in st.session_state:
    st.session_state.user = None

if "force_change_pw" not in st.session_state:
    st.session_state.force_change_pw = False

if "pending_email" not in st.session_state:
    st.session_state.pending_email = None

# /*
# Flow -> if user clicks create account -> call start_createacount(email) -> set pending_email
#         if pending_email exists -> show OTP screen -> call verify_otp_and_create_user(...)
#
# */


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

# retriever, client = get_resources()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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


auth_handlers = {
    "login": authenticate_user,
    "create": create_user,
    "reset":reset_password
}

# ------------------ USER INTERFACE ---------------------------

if st.session_state.user is None:
    result = render_auth_screen(auth_handlers)

    if result.get("action") == "login_success":
        st.session_state.user = result["user"]
        st.session_state.force_change_pw = result["user_record"].get("must_change", False)
        st.rerun()

    st.stop()

if st.session_state.force_change_pw:
    res = render_force_change_password()
    if res.get("action") == "change_password":
        ok, msg = change_password(st.session_state.user, res["new_password"])
        if ok:
            st.success("Password changed successfully. Please login again.")
            st.session_state.user = None
            st.session_state.force_change_pw = False
            st.rerun()
        else:
            st.error(msg)
    st.stop()



render_header()

render_sidebar(
    st.session_state.conversations,
    st.session_state.current_chat_id,
    on_new_chat=new_chat,
    on_select_chat=select_chat
)

current_chat = get_current_chat()
render_chat(current_chat["messages"], show_process_table)
user_input = render_input(False)


current_chat = get_current_chat()

if user_input:
    # title set if first user message
    if current_chat["title"] == "New Chat":
        current_chat["title"] = user_input[:30]

    current_chat["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("📘 Analyzing rules..."):
            intent = detect_intent(user_input)
            amount = extract_amount(user_input)

            if intent == "TABLE":
                st.markdown("### 📊 CBRI / CSIR Purchase Process – Cost Slab Wise")
                show_process_table()
                answer = "__TABLE_SHOWN__"

            elif intent == "POLICY":
                context = "TEST CONTEXT"
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": POLICY_PROMPT},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"}
                    ],
                    temperature=0.3
                )
                answer = response.choices[0].message.content

                st.markdown(answer)

            elif intent == "PROCESS":
                if amount is None:
                    answer = "🙂 Purchase process batane ke liye exact amount bata do (jaise ₹8,00,000 / 8 lakh)."
                    st.markdown(answer)

                else:
                    # ---- HARD SLAB DECISION IN PYTHON (FINAL AUTHORITY) ----
                    if amount <= 200000:
                        slab = "Up to ₹2,00,000"
                        mode = "Direct Purchase"
                        committee = "No committee"
                        committee_required = "No"

                    elif 200000 < amount <= 1000000:
                        slab = "₹2,00,001 to ₹10,00,000"
                        mode = "LPC"
                        committee = "Local Purchase Committee (LPC)"
                        committee_required = "Yes"

                    elif 1000000 < amount <= 2500000:
                        slab = "₹10,00,001 to ₹25,00,000"
                        mode = "LTE"
                        committee = "Technical & Purchase Committee (T&PC)"
                        committee_required = "Yes"

                    else:
                        slab = "Above ₹25,00,000"
                        mode = "Open / Global Tender"
                        committee = "Technical & Purchase Committee (T&PC)"
                        committee_required = "Yes"

                # --------------------------------------------------------

                # LLM ko sirf STEPS / DOCUMENTS likhne bolo, numbers/mode change na kare

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": PROCESS_PROMPT},
                        {"role": "user", "content": f"""
                           The exact purchase amount is: {amount}
                           The slab is: {slab}
                           The applicable mode is: {mode}
                           The committee is: {committee}

                           IMPORTANT:
                           - DO NOT write Case Summary.
                           - DO NOT write Applicable Procurement Mode.
                           - DO NOT write Committee Involvement.
                           - ONLY write these sections:

                           Step-by-Step Process
                           Key Documents / Outputs
                           One-line Summary (TL;DR)
                           FLOWCHART (Mermaid)

                           Write in simple Hinglish.
                           Do NOT repeat headings already given.

                           User question:
                           {user_input}
                        """}
                    ],
                    temperature=0.3
                )

                llm_text = response.choices[0].message.content
                # ---- CLEAN DUPLICATE HEADERS IF MODEL STILL PRINTS THEM ----
                for bad in [
                    "Case Summary",
                    "Applicable Procurement Mode",
                    "Applicable Procurement Mode & Reason",
                    "Committee Involvement",
                ]:
                    llm_text = llm_text.replace(bad, "")
                # -----------------------------------------------------------

                # FINAL ANSWER: Header Python se, details LLM se
                answer = f"""
                ## Case Summary
                - Purchase value: ₹{amount}
                - Item: Not mentioned
                - Which cost slab/category it falls into: {slab}

                ## Applicable Procurement Mode & Reason
                - Which mode applies: {mode}
                - Why this mode applies: As per CSIR Manual on Procurement of Goods 2019, this amount falls in this slab.

                ## Committee Involvement
                - Whether committee is required: {committee_required}
                - Which committee: {committee}

                {llm_text}
                

            """
                st.markdown(answer)

                diagram = extract_mermaid(answer)

                if diagram:
                    st.subheader("📊 Process Flowchart")

                    st_mermaid(diagram)

            else:
                answer = (
                    "👋 Examples:\n"
                    "• ₹8 lakh ka purchase process kya hoga?\n"
                    "• Minister approval kis stage par chahiye?\n"
                    "• Show table of procurement process as per cost slabs"
                )
                st.markdown(answer)

    if "answer" in locals() and answer.strip():
        current_chat["messages"].append({"role": "assistant", "content": answer})

    st.rerun()
