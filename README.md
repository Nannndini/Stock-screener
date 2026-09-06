# Stock Screener Agent (LangGraph)

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # then paste your real ANTHROPIC_API_KEY into .env
```

## Run
```bash
python agent.py "Find undervalued tech stocks with P/E under 20"
```
or with defaults:
```bash
python agent.py
```

## How it's wired
- **State**: `AgentState` holds a running `messages` list (auto-appended via
  `add_messages`).
- **Tools** (`get_stock_price`, `screen_stocks`, `get_price_history`): plain
  Python functions wrapped with `@tool`, backed by `yfinance`.
- **agent node**: calls Claude with the tools bound; Claude decides whether
  to call a tool or answer directly.
- **tools_condition / ToolNode**: LangGraph's prebuilt routing — if the last
  message has tool calls, go to `tools`; otherwise `END`.
- **Loop**: `tools -> agent` so results feed back in until Claude is
  satisfied and returns a plain answer.

## Swap the LLM
Using OpenAI instead? `pip install langchain-openai`, then in `agent.py`:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```
Everything else (tools, graph, routing) stays the same — that's the point
of LangGraph's tool-binding abstraction.

## Extend it
- Add a `rank_stocks` tool that sorts screened results by a metric.
- Add a Streamlit/Gradio front end that calls `graph.invoke()`.
- Swap `yfinance` for a paid data API if you hit rate limits (Yahoo's
  unofficial endpoint throttles hard under load).
