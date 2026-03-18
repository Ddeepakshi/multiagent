# Multi-Agent AI System

A multi-agent AI system built with **LangGraph** and **Groq (LLaMA 3.3 70B)** that uses a Planner-Worker-Reviewer architecture to generate high-quality, reviewed responses to user queries.

## Architecture

```
User Query
    |
    v
[Planner Agent] -- Creates an actionable plan
    |
    v
[Worker Agent] -- Generates a response based on the plan
    |
    v
[Reviewer Agent] -- Evaluates the response for quality
    |
   / \
  /   \
Approve  Revise (max 2 iterations)
  |        \
  v         --> back to Worker Agent
Final Response
```

### How It Works

1. **Planner Agent** - Analyzes the user query and creates a structured action plan for the worker
2. **Worker Agent** - Generates a detailed response following the plan, incorporating reviewer feedback if available
3. **Reviewer Agent** - Strictly evaluates the draft for concrete examples, implementation details, tradeoffs, clarity, and actionable recommendations
4. **Conditional Loop** - If the reviewer requests revisions, the worker refines its response (up to 2 iterations to prevent infinite loops)

## Tech Stack

| Layer     | Technology                     |
|-----------|--------------------------------|
| LLM       | LLaMA 3.3 70B via Groq API    |
| Orchestration | LangGraph (StateGraph)     |
| Backend   | FastAPI + Uvicorn              |
| Frontend  | HTML, CSS, JavaScript          |
| Language  | Python 3.12                    |

## Project Structure

```
multiagent/
├── backend/
│   ├── app.py              # CLI version (standalone)
│   ├── server.py           # FastAPI backend with /run endpoint
│   ├── requirements.txt    # Python dependencies
│   └── .env                # API keys (not tracked)
├── frontend/
│   └── index.html          # Web UI
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone https://github.com/Ddeepakshi/multiagent.git
cd multiagent/backend

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file inside `backend/`:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Run the CLI Version

```bash
cd backend
python app.py
```

### Run the Web App

**Start the backend:**

```bash
cd backend
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Open the frontend:**

Open `frontend/index.html` in your browser. Enter a query and click **Run**.

## API Reference

### `POST /run`

Runs the multi-agent pipeline on a user query.

**Request:**

```json
{
  "query": "Explain the pros and cons of microservices architecture"
}
```

**Response:**

```json
{
  "response": "Final refined answer from the worker agent...",
  "steps": [
    { "agent": "Planner", "output": "..." },
    { "agent": "Worker (attempt 1)", "output": "..." },
    { "agent": "Reviewer (round 1)", "output": "approve: ..." }
  ]
}
```

## Key Design Decisions

- **LangGraph StateGraph** for orchestrating agent flow with conditional edges, enabling a review loop without complex control flow
- **Groq API** for fast inference on LLaMA 3.3 70B — sub-second token generation
- **Max 2 review iterations** to balance response quality with latency and API usage
- **Step-by-step visibility** in the API response so the frontend can show the reasoning process of each agent
- **File logging** of every agent output for debugging and auditability

## Demo

1. Enter a query in the web UI
2. The system runs through the Planner → Worker → Reviewer pipeline
3. Each agent's output is displayed as a step card
4. The final reviewed response is highlighted at the bottom

## Future Improvements

- Streaming responses for real-time agent output
- Add more specialized agents (Research, Code Generator, Fact Checker)
- Persistent conversation history
- Deploy with Docker

## License

MIT
