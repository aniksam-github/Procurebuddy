🤖 CBRI ProcureBuddy

Domain-Specific RAG Chatbot for Government Procurement (GFR 2017)

📌 Overview

CBRI ProcureBuddy is a Retrieval-Augmented Generation (RAG) based conversational assistant designed to help CBRI / CSIR scientists and officials understand and apply Government Financial Rules (GFR 2017) for procurement decisions.

The system answers procurement-related queries (e.g. ₹10,000, ₹30,000 purchase cases) using only official GFR documents, ensuring accuracy, audit-safety, and zero hallucination.

🎯 Key Objectives

1. Provide correct procurement procedure based on exact value slabs

2. Avoid hallucination and incorrect committee recommendations

3. Deliver clear, practical Hinglish answers

4. Ensure audit-safe and rule-compliant responses

5. Replace manual rule-checking with an intelligent assistant

🧠 System Architecture (High Level)

PDF Documents (GFR 2017)
        ↓
Text Chunking
        ↓
Embeddings (HuggingFace - Local)
        ↓
ChromaDB (Persistent Vector Store)
        ↓
Retriever (Top-k semantic search)
        ↓
Strict System Prompt (Rules)
        ↓
Groq LLM (LLaMA 3.1)
        ↓
Streamlit Chat UI

🧩 Tech Stack

🔹Frontend
* Streamlit – Interactive chat-based UI

🔹 Backend / AI
* LangChain (Classic v0.1.x) – RAG pipeline
* Groq API – LLM inference
* HuggingFace Sentence Transformers – Local embeddings
* ChromaDB – Vector database (persistent)

bot/
│
├── app.py              # Streamlit application
├── ingest.py           # PDF ingestion & vector DB creation
├── chroma_db/          # Persistent Chroma vector store
├── data/               # GFR 2017 PDFs
├── .env                # Environment variables (API keys)
├── requirements.txt
└── README.md


📦 Packages & Version Stability

⚠️ IMPORTANT:
This project intentionally uses LangChain 0.1.x (classic).
Newer LangChain versions (≥1.0) cause breaking changes and incompatibilities.

✅ Stable & Tested Package Versions

* python==3.11.x
* streamlit==1.31.0
* langchain==0.1.16
* langchain-core==0.1.53
* langchain-community==0.0.38
* langchain-text-splitters==0.0.2
* chromadb==0.4.24
* sentence-transformers==2.2.2
* transformers==4.37.2
* torch==2.1.2
* groq==0.37.1
* python-dotenv==1.0.1
* pypdf==4.0.1
* numpy==1.26.4

🔍 Version Stability Notes

| Package                  | Reason for Version Pin                     |
| ------------------------ | ------------------------------------------ |
| `langchain==0.1.16`      | Stable RAG APIs (`create_retrieval_chain`) |
| `langchain-core==0.1.53` | Compatible with classic LangChain          |
| `chromadb==0.4.24`       | Avoids SQLite schema conflicts             |
| `numpy==1.26.4`          | NumPy ≥2.0 breaks Chroma                   |
| `torch==2.1.2`           | Compatible with sentence-transformers      |
| `groq==0.37.1`           | Stable Groq client (pre-1.0)               |

🔐 Environment Variables

Create a .env file:
GROQ_API_KEY=your_groq_api_key_here

🛠️ Setup Instructions

1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add GFR PDFs

Place official GFR 2017 PDFs inside:

data/

4️⃣ Run Ingestion
python ingest.py


This will:
* Load PDFs
* Split text into chunks
* Create embeddings
* Store vectors in chroma_db/

5️⃣ Run Application
- streamlit run app.py


🧠 RAG Logic (Important)

🔹Retrieval
* Top-5 relevant chunks retrieved from ChromaDB
* Based on semantic similarity

🔹Prompt Discipline

Strict system rules:

* Use ONLY provided context
* Exact value slab detection
* No rounding of amounts
* No committee mention below ₹25,000
* Mandatory LPC for ₹25,001–₹2,50,000
* Fallback:
“This information is not found.”

🗣️ Response Style
* Hinglish (simple Hindi + English)
* Short, bulleted, practical
* Officer / audit friendly
* No unnecessary explanation

✅ Supported Value Slabs (GFR 2017)

| Purchase Value      | Procedure                                |
|---------------------|------------------------------------------|
| <= ₹25,000          | Direct purchase, no committee            |
| ₹25,001 – ₹2,50,000 | Local Purchase Committee (LPC) mandatory |
| > ₹2,50,000         | Outside current scope                    |

❌ Known Limitations

* Only covers GFR 2017 PDFs provided
* No internet browsing
* No financial approval workflow
* Not fine-tuned (pure RAG)

🚀 Why RAG (Not Fine-Tuning)?

* Government rules change → PDFs can be updated
* No retraining cost
* Answers are traceable to source
* Audit-safe & explainable

🧪 Testing Strategy

* Boundary value tests (₹25,000 / ₹25,001)
* Hallucination control queries
* Trick questions
* Language quality checks

📈 Future Enhancements

* Source citation display
* Decision-table UI
* Multi-rule support (Store, Works, Consultancy)
* Deployment on intranet server
* Role-based access (Scientist / Purchase Officer)

🏁 Conclusion

CBRI ProcureBuddy demonstrates a production-grade RAG system for government procurement use cases, combining:
* Strong prompt engineering
* Controlled LLM behavior
* Domain accuracy
* Practical UI

⚠️ Hard Rules

DO NOT:
- Upgrade langchain to >=1.0
- Upgrade numpy to >=2.0
- Mix pip installs without venv
- Install langchain-groq (not needed)

USE:
- Groq client directly (groq.Groq)
- LangChain only for RAG logic
