"""
FastAPI wrapper around the LangGraph stock screener agent.
Exposes a single POST /screen endpoint that runs the agent loop and
returns the final answer plus a trace of every tool call it made along
the way, so callers can see the agent's reasoning, not just the output.
"""

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from agent import graph

app = FastAPI(
    title="Stock Screener Agent API",
    description="LangGraph agent that screens stocks using live yfinance data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScreenRequest(BaseModel):
    query: str


class ToolCallTrace(BaseModel):
    tool: str
    input: dict[str, Any]
    output: Any


class ScreenResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallTrace]


@app.get("/")
def health():
    return {"status": "ok", "service": "stock-screener-agent"}


@app.post("/screen", response_model=ScreenResponse)
def screen(req: ScreenRequest):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(500, "GROQ_API_KEY not configured on server")

    state = {"messages": [HumanMessage(content=req.query)]}
    result = graph.invoke(state)
    messages = result["messages"]

    # Build a trace: match each AI tool_call to its ToolMessage result.
    tool_calls: list[ToolCallTrace] = []
    tool_results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls.append(
                    ToolCallTrace(
                        tool=tc["name"],
                        input=tc["args"],
                        output=tool_results.get(tc["id"], None),
                    )
                )

    final = messages[-1]
    return ScreenResponse(answer=final.content, tool_calls=tool_calls)
