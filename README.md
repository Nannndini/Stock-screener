# Stock Screener Agent

An AI agent built with **LangGraph** that screens stocks against live market
data. You give it a natural-language query — a ticker, a sector theme, a
P/E cutoff — and it decides which tools to call, pulls real data from
Yahoo Finance via `yfinance`, and reasons over the results before answering.

Runs as a CLI script or a FastAPI service, deployable on Render.

## How it works

```
        ┌────────────┐
        │   agent    │  The LLM decides: answer now, or call a tool?
        └─────┬──────┘
              │
       tool call needed?
         │           │
        yes          no
         │           │
   ┌─────▼─────┐     │
   │   tools   │     │
   │ (yfinance)│     │
   └─────┬─────┘     │
         │            │
         └──────► back to agent ──► END (final answer)
```

- **State** (`agent.py`): a running `messages` list, auto-appended via
  LangGraph's `add_messages` reducer.
- **Tools**: `get_stock_price`, `screen_stocks`, `get_price_history` — plain
  Python functions decorated with `@tool`, backed by `yfinance`.
- **Graph**: `agent` node calls the LLM (Llama 3.3 70B via Groq) with tools bound
  (`llm.bind_tools()`). `tools_condition` routes to the prebuilt `ToolNode`
  if the LLM requested a tool call, otherwise straight to `END`. Tool
  results loop back into `agent` so it can chain multiple calls
  (e.g. screen a list, then pull history on the winners) before answering.

## Project structure

```
.
├── agent.py         # LangGraph agent: state, tools, graph definition
├── main.py          # FastAPI wrapper exposing POST /screen
├── requirements.txt
├── render.yaml       # Render deploy config
└── .env.example
```

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env   # paste your real GROQ_API_KEY into .env
```

**As a CLI:**
```bash
python agent.py "Find undervalued tech stocks with P/E under 20"
```

**As an API:**
```bash
uvicorn main:app --reload
```
Then:
```bash
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d '{"query": "Screen AAPL, MSFT, GOOGL, NVDA, INTC for P/E under 30"}'
```

Response includes the final answer **and** a trace of every tool call
the agent made — what it called, with what arguments, and what came back —
so you can see the reasoning, not just the output.

## Deploy on Render

1. Push this repo to GitHub (done).
2. On [render.com](https://render.com), **New → Web Service** → connect
   this repo. Render will pick up `render.yaml` automatically.
3. Add the `GROQ_API_KEY` environment variable in the Render
   dashboard (it's marked `sync: false` in `render.yaml` so it's never
   read from the repo).
4. Deploy. First request after idle will cold-start (~30-50s on the free
   tier) — expected, not a bug.

## API reference

`POST /screen`

Request:
```json
{ "query": "string" }
```

Response:
```json
{
  "answer": "string",
  "tool_calls": [
    { "tool": "string", "input": {}, "output": {} }
  ]
}
```

`GET /` — health check.

## Notes

- `yfinance` hits Yahoo's unofficial, unauthenticated endpoint — it
  throttles under heavy/repeated calls. If you see empty results or
  errors under load, that's rate-limiting, not a bug in the graph.
- Swapping LLM providers: install the relevant `langchain-*` package,
  replace the `llm =` line in `agent.py` (e.g. `ChatAnthropic(...)` or
  `ChatOpenAI(...)`), and set the matching API key env var. Nothing else
  changes — that's the point of LangGraph's tool-binding abstraction.
