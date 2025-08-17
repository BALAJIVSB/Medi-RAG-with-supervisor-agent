

# Medical Support RAG Agent

Large Language Models (LLMs) like ChatGPT and Gemini perform well in understanding natural language. However, they can suffer from hallucinations in critical areas such as medical question answering. To fix this issue, a Supervisor-Agent framework with adaptive query routing is proposed. This system focuses on oncology, especially on queries related to osteosarcoma. It integrates a Chroma-based vector store from specific PDFs into a tailored Retrieval-Augmented Generation (RAG) module. The Supervisor directs domain queries to the RAG and non-domain queries to the LLM.

---

## ✨ Features

* 📄 Loads and splits PDF documents from the `data/` directory
* 📦 Embeds and stores document chunks in a persistent vector database
* 🤖 Uses OpenAI’s GPT-3.5-turbo for answering questions with context retrieval (RAG)
* 🔄 Falls back to LLM-only answers if context is insufficient
* 💻 Interactive command-line interface

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/med_sup_rag.git
cd med_sup_rag/Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Set your OpenAI API key as an environment variable:

```bash
set OPENAI_API_KEY=your-openai-api-key
```

Or update the code to pass your API key directly into
`OpenAIEmbeddings()` and `ChatOpenAI()` inside `main.py`.

### 4. Add PDF files

Place your medical PDF documents in the `data/` directory.

---

## ▶️ Usage

Run the agent:

```bash
python main.py
```

You will see:

```
RAG Agent Ready! Ask me anything. Type 'exit' to quit.
```

Type your medical question and get a concise answer based on your documents.

---


## 🔧 Customization

* ✏️ Update the prompt in `main.py` to specialize for other medical domains
* 📏 Adjust `chunk_size` and `chunk_overlap` for different document types
* 🎯 Modify the retrieval `k` value to return more or fewer context chunks

---

