# Project Spec: News Research Bot for Financial Analysis

> **For Claude Code:** Read this whole file before writing code. Build it phase by phase (§9) and check in with me after each phase. Keep the code simple and readable. Use current LangChain imports (the newer `langchain_community`, `langchain_openai` and `langchain_text_splitters` packages), not the old deprecated ones.

---

## 1. About the project

- **What:** a personal project in Agentic AI
- **Goal:** a chatbot for financial research. The user pastes links to news articles (for example from Moneycontrol, Economic Times, Reuters or Mint). The bot reads them, stores them in a vector database and answers questions about them, citing its sources. It can also pull out key insights such as company names, numbers, sentiment and main takeaways.

### What the CV says (the project must back all of this up)

> - Developed a **financial analysis chatbot**, capable of **extracting key insights from news articles** and **answering queries**
> - Utilized **LangChain** features like **FAISS, UnstructuredURLLoader & RecursiveTextSplitter** to create a **vector database**
> - Integrated **OpenAI API** to leverage **large language models** and developed an intuitive user interface using **Streamlit**

Note: "RecursiveTextSplitter" on the CV refers to LangChain's `RecursiveCharacterTextSplitter`.

---

## 2. How it works

```
News URLs ──► UnstructuredURLLoader ──► RecursiveCharacterTextSplitter ──► OpenAI Embeddings ──► FAISS index (saved to disk)
                                                                                                       │
User question ──► retrieve top matching chunks from FAISS ──► OpenAI chat model ──► answer + source URLs
```

1. **Load:** `UnstructuredURLLoader` downloads and extracts the text of each article.
2. **Split:** `RecursiveCharacterTextSplitter` breaks the text into chunks (start with about 1000 characters and 200 overlap; make this configurable).
3. **Embed and store:** OpenAI embeddings turn chunks into vectors. **FAISS** stores them. Save the index to `vectorstore/` so it doesn't need rebuilding on every run.
4. **Answer:** for a question, retrieve the top 4 most relevant chunks. The LLM answers **using only those chunks** and lists the source URLs. If the answer isn't in the articles, it says so instead of guessing.

---

## 3. Features

### 3.1 Q&A over articles (core)
- Ask any question about the loaded articles
- Every answer shows its **sources**: the URLs the information came from
- Keep a short chat history so follow-up questions work (for example: "What about its profit?")

### 3.2 Key insight extraction
A button, **"Get Key Insights"**, that runs for each article and produces:
- a 3–5 line summary
- companies and tickers mentioned
- important numbers (revenue, profit, % changes, price targets)
- sentiment (Positive / Negative / Neutral) with a one-line reason
- the main takeaway for an investor

Use a fixed prompt that returns structured output (JSON or a Pydantic model), so the app can show it neatly.

### 3.3 Simple agent (the "Agentic AI" part)
A LangChain agent using an OpenAI chat model with tool calling. It decides which tool to use for a question:
- `search_articles`: searches the FAISS index (the Q&A above)
- `extract_insights`: runs the insight extraction on one article
- `compare_articles`: compares what different articles say about the same company or topic

Keep it to these 3 tools. Simple questions should still go straight to the search tool.

---

## 4. Streamlit UI (`app.py`)

- **Sidebar:** boxes to paste up to 3–5 news URLs, and a **"Process URLs"** button that shows progress (Loading → Splitting → Building index → Done)
- **Main area:**
  - a chat box for questions, with answers and their sources below
  - a **"Key Insights"** tab showing one card per article (summary, companies, numbers, sentiment badge)
- Show clear error messages if a URL fails to load or the API key is missing
- Show a "Clear / start over" button that deletes the index and chat history

---

## 5. Settings and API key

- The OpenAI API key goes in a `.env` file (`OPENAI_API_KEY=...`), loaded with `python-dotenv`. Never put the key in code. Add `.env` to `.gitignore`.
- Put settings in `config.py`: chat model name, embedding model name, chunk size, chunk overlap, top-k, temperature (default 0 for factual answers).

### Models to use

| Job | Model | Why |
|---|---|---|
| Chat, Q&A, insights, agent | **`gpt-5.4-mini`** | Cheap, good at following instructions and at tool calling (needed for the agent) |
| Cheaper backup for chat | `gpt-5.4-nano` | Use it if credits run low. It's weaker at the agent and insights. |
| Embeddings (vector database) | **`text-embedding-3-small`** | Very cheap and good enough for searching news articles |

- Put the model names only in `config.py`, so switching is a one-line change.
- If OpenAI rejects a model name, check the current model list on OpenAI's docs and tell me before switching to a different model.
- **Don't change the embedding model after building the FAISS index.** Vectors from different embedding models don't match. If it changes, rebuild the index.
- Use a temperature of 0 for Q&A and insights, so answers stick to the facts.

---

## 6. Handling problems

- Some news sites block scraping or have paywalls. If a URL returns little or no text, warn the user and skip it.
- If `UnstructuredURLLoader` fails on a site, fall back to `WebBaseLoader`.
- Remove duplicate chunks.
- If the API call fails (no key, rate limit), show a friendly error instead of crashing.

---

## 7. Folder structure

```
news-research-bot/
├── NEWS_BOT_SPEC.md
├── README.md
├── requirements.txt
├── .env.example          # OPENAI_API_KEY=your-key-here
├── .gitignore
├── config.py
├── app.py                # Streamlit UI
├── src/
│   ├── loader.py         # URL loading + text splitting
│   ├── vectorstore.py    # embeddings + FAISS build/save/load
│   ├── qa_chain.py       # retrieval Q&A with sources
│   ├── insights.py       # key insight extraction
│   └── agent.py          # agent + 3 tools
├── vectorstore/          # saved FAISS index (gitignored)
└── tests/
    └── test_basic.py     # a few simple tests (splitting, index build/load)
```

**Tech:** Python 3.10+, langchain, langchain-community, langchain-openai, langchain-text-splitters, faiss-cpu, unstructured, openai, python-dotenv, pydantic, streamlit.

---

## 8. How to check it works

Test with 3 real financial news articles (I'll give the URLs, or you can pick recent ones I approve). Then check:
- 5 factual questions whose answers are in the articles: the answers are correct and the sources are right
- 2 questions whose answers are **not** in the articles: the bot says it doesn't know
- A follow-up question uses the chat history correctly
- Key insights for each article look accurate
- The agent picks the right tool for a search, an insights request and a comparison question

Write the results in `reports/test_results.md`.

---

## 9. Build phases

1. **Setup:** folders, requirements, `.env` loading, config
2. **Loading and splitting:** load URLs with `UnstructuredURLLoader`, split with `RecursiveCharacterTextSplitter`, print chunk counts
3. **Vector database:** OpenAI embeddings and FAISS, saved to and loaded from disk
4. **Q&A with sources:** the retrieval chain with citations and chat history
5. **Key insight extraction:** structured output for each article
6. **Agent:** 3 tools with tool calling
7. **Streamlit UI:** sidebar, chat and insights tab
8. **Testing:** the checks in §8
9. **README:** what the project does, how to install and run it, a screenshot, and an example Q&A

After each phase, tell me what you built, how to try it and what's next.
