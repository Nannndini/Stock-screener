"""
Stock Screener Agent — LangGraph
---------------------------------
A minimal agentic loop: LLM decides which tools to call, tools hit yfinance,
results get fed back to the LLM, LLM decides whether to keep going or answer.

Set ANTHROPIC_API_KEY in your environment (or a .env file) before running.
"""

import os
from typing import Annotated, TypedDict

import yfinance as yf
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ---------- Tools ----------

@tool
def get_stock_price(ticker: str) -> dict:
    """Get the current price, market cap, and basic info for a stock ticker."""
    t = yf.Ticker(ticker)
    info = t.info
    return {
        "ticker": ticker,
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "sector": info.get("sector"),
    }


@tool
def screen_stocks(tickers: list[str], max_pe: float = 25.0) -> list[dict]:
    """Screen a list of tickers and return only those with a P/E ratio
    below max_pe. Use this when the user wants stocks matching a criterion,
    not just a lookup of a single stock."""
    results = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            pe = info.get("trailingPE")
            if pe is not None and pe < max_pe:
                results.append({
                    "ticker": ticker,
                    "pe_ratio": pe,
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "sector": info.get("sector"),
                })
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return results


@tool
def get_price_history(ticker: str, period: str = "1mo") -> dict:
    """Get recent price history stats (period like '5d', '1mo', '6mo', '1y')."""
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return {"ticker": ticker, "error": "no data"}
    start_close = float(hist["Close"].iloc[0])
    end_close = float(hist["Close"].iloc[-1])
    pct_change = round((end_close - start_close) / start_close * 100, 2)
    return {
        "ticker": ticker,
        "period": period,
        "start_close": round(start_close, 2),
        "end_close": round(end_close, 2),
        "pct_change": pct_change,
        "high": round(float(hist["High"].max()), 2),
        "low": round(float(hist["Low"].min()), 2),
    }


TOOLS = [get_stock_price, screen_stocks, get_price_history]


# ---------- State ----------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------- LLM + graph ----------

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "You are a stock screener agent. You have tools to fetch live prices, "
    "history, and to screen a list of tickers against a P/E cutoff. "
    "Always use the tools for real data instead of guessing numbers. "
    "When a user gives a vague sector or theme (e.g. 'undervalued tech "
    "stocks'), pick a reasonable list of 5-8 real tickers for that theme "
    "yourself before screening them."
)


def agent_node(state: AgentState):
    messages = state["messages"]
    if not messages or messages[0].type != "system":
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(TOOLS))

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


# ---------- Runner ----------

def run(query: str):
    from langchain_core.messages import HumanMessage
    state = {"messages": [HumanMessage(content=query)]}
    for event in graph.stream(state, stream_mode="values"):
        last = event["messages"][-1]
        last.pretty_print()
    return event


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Screen AAPL, MSFT, GOOGL, NVDA, INTC for a P/E under 30"
    run(q)
