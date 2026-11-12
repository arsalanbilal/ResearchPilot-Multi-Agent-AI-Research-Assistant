# 🤖 Supervisor Multi-AI-Agent System

A multi-agent AI application where a **Supervisor** coordinates a **Researcher → Analyst → Writer** pipeline to turn a single question into a structured executive report. The Researcher is connected to **live news and Wikipedia** through a custom **Model Context Protocol (MCP)** server, so it can answer questions about recent events, not just what the model memorised in training.

Built with **LangGraph**, **Groq (LLMs)**, **MCP**, **LangSmith** and **Streamlit**.

---

## 📌 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Key Features](#-key-features)
3. [Architecture](#-architecture)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Getting Started](#-getting-started)
7. [Usage](#-usage)
8. [How It Works](#-how-it-works)
9. [Design Decisions](#-design-decisions)
10. [Limitations & Future Work](#-limitations--future-work)
11. [What I Learned](#-what-i-learned)
12. [Author](#-author)

---

## 🎯 Problem Statement

A single LLM call is often not enough for a good research report: it mixes research, analysis and writing in one step, and it cannot answer questions about **recent events** because its knowledge is frozen at training time.

This project splits the work between specialised agents, each with one responsibility, managed by a supervisor. It also gives the research agent access to live data through MCP.

---

## ✨ Key Features

- **Supervisor pattern**: a supervisor agent decides which agent works next, orchestrated as a state graph with LangGraph.
- **Specialised agents**: Researcher (gathers information), Analyst (insights, risks, recommendations), Writer (final Markdown report).
- **Live information via MCP**: the Researcher calls tools from a custom MCP server for latest news and Wikipedia.
- **Guard-rails**: the supervisor's LLM decision is validated against the required order (Researcher → Analyst → Writer), so the workflow cannot loop or skip a step.
- **Graceful fallback**: if the MCP server is unreachable, the Researcher falls back to the model's own knowledge and tells the user.
- **Observability**: optional LangSmith tracing with region/workspace support and an in-app connection test and trace verification.
- **Friendly UI**: Streamlit app with live agent progress, model selection, temperature control and a one-click report download (`.md`).
- **Secure by default**: API keys are entered in the sidebar (or `.env`), used only for the session and never written to disk.

---

## 🏗 Architecture

```
                        ┌──────────────────────────┐
        User Task ────▶ │        SUPERVISOR        │ ◀──────────────┐
                        │ (decides next agent +    │                │
                        │  pipeline guard-rail)    │                │
                        └────────────┬─────────────┘                │
                                     │                              │
                 ┌───────────────────┼───────────────────┐          │
                 ▼                   ▼                   ▼          │
          ┌────────────┐      ┌────────────┐      ┌────────────┐    │
          │ RESEARCHER │      │  ANALYST   │      │   WRITER   │    │
          │ (MCP host) │      │            │      │            │────┘
          └─────┬──────┘      └────────────┘      └────────────┘
                │  MCP client (stdio)                    │
                ▼                                        ▼
        ┌───────────────────┐                   📄 Final Markdown Report
        │    MCP SERVER     │
        │ • get_latest_news │──▶ Google News RSS
        │ • wikipedia_search│──▶ Wikipedia API
        │ • wikipedia_      │
        │   current_events  │
        └───────────────────┘

        LangSmith traces every run (optional)
```

**MCP roles in this project**

| Role | Component | Responsibility |
|------|-----------|----------------|
| **Host** | `superviser.py` (Researcher agent) | Owns the LLM, decides when tools are needed |
| **Client** | `mcp_client.py` | Starts the server, lists tools, forwards tool calls |
| **Server** | `mcp_news_server.py` | Exposes news and Wikipedia tools |

---

## 🧰 Tech Stack

| Area | Technology |
|------|-----------|
| Language | Python 3.10+ |
| Agent orchestration | LangGraph, LangChain |
| LLM provider | Groq (Llama 3.3 70B, Llama 3.1 8B, GPT-OSS) |
| Tool protocol | Model Context Protocol (`mcp` Python SDK, FastMCP) |
| Observability | LangSmith |
| Frontend | Streamlit |
| Data sources | Google News RSS, Wikipedia REST/Action API (no API keys needed) |

---

## 📁 Project Structure

```
.
├── superviser.py          # Streamlit app + LangGraph supervisor workflow + agents
├── mcp_client.py          # MCP client: connects the Researcher to the MCP server
├── mcp_news_server.py     # MCP server: live news + Wikipedia tools
├── requirements.txt       # Python dependencies
├── .env                   # (optional) API keys, not committed
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com/keys)
- (Optional) A [LangSmith API key](https://smith.langchain.com) for tracing

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration (optional)

You can type keys in the app sidebar, or create a `.env` file:

```env
GROQ_API_KEY=gsk_your_key_here
LANGSMITH_API_KEY=lsv2_your_key_here
LANGSMITH_PROJECT=Supervisor-Multi-Agent
LANGSMITH_WORKSPACE_ID=            # only if your LangSmith key is org-scoped
```

### Run

```bash
streamlit run superviser.py
```

The app opens at `http://localhost:8501`.

---

## 💡 Usage

1. Enter your **Groq API key** in the sidebar (and optionally LangSmith details).
2. Pick a model and temperature.
3. Type a task, for example:
   - `What are the benefits of AI in Healthcare`
   - `Latest news on electric vehicles this week` *(triggers live news tools)*
4. Click **Run Multi-Agent Team** and watch each agent report its progress.
5. Read the final report and download it as a Markdown file.

---

## ⚙️ How It Works

1. **Supervisor** reads the shared state (has research? analysis? report?) and picks the next agent. The LLM's choice is checked against a guard-rail that enforces the correct order.
2. **Researcher** starts the MCP server over stdio, loads its tools, and lets the LLM call them in a short tool-calling loop (max 4 rounds). For recent-news questions it uses `get_latest_news`; for background it uses `wikipedia_search`.
3. **Analyst** turns the research into insights, risks, opportunities and recommendations.
4. **Writer** compiles a structured report: Executive Summary, Key Findings, Analysis & Insights, Recommendations, Conclusion.
5. The graph ends when the report is complete, and every step is traced in **LangSmith** if enabled.

---

## 🧠 Design Decisions

- **LLM advisory + deterministic guard-rail**: LLM routing can be unreliable, so the supervisor accepts the model's suggestion only if it matches the valid next step. This prevents infinite loops.
- **MCP instead of hard-coded API calls**: tools live in a separate server, so they can be reused by other agents or clients and extended without touching agent code.
- **MCP only for the Researcher**: only that agent needs external data; the Analyst and Writer work on the data they are given.
- **Keyless data sources**: Google News RSS and Wikipedia keep setup friction low.
- **LLM injected into the graph**: the user's API key is never stored globally.

---

## 🔭 Limitations & Future Work

- News is limited to headlines, sources and links; full article text is not fetched.
- Wikipedia "current events" may be empty early in the day.
- Small models may be less reliable at tool calling; larger models are recommended.
- Planned improvements:
  - Add more MCP tools (web search, finance data, article extraction)
  - Add source citations inside the final report
  - Let the Supervisor re-run the Researcher if the Analyst finds gaps
  - Persist past reports and add PDF export
  - Add unit tests and Docker deployment

---

## 📚 What I Learned

- Designing **multi-agent workflows** with shared state using LangGraph
- Implementing the **Model Context Protocol** end to end (server, client and host)
- **Tool calling** with LLMs and handling failures with fallbacks
- Adding **observability** to LLM apps with LangSmith
- Building and securing a small **Streamlit** app that handles user-provided API keys

---

## 👤 Author

**Arsalan Bilal**
📧 arsalanbilal11@gmail.com

---

⭐ If you found this project interesting, consider giving it a star!
