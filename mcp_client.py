import asyncio
import json
from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

import mcp_news_server

# Find the FastMCP server object in mcp_news_server.py, whatever it is named
news_server = next(
    (v for v in vars(mcp_news_server).values() if isinstance(v, FastMCP)), None
)
if news_server is None:
    raise ImportError(
        "mcp_news_server.py has no FastMCP server (expected: mcp = FastMCP('news-server')). "
        "Replace it with the provided mcp_news_server.py."
    )

MAX_TOOL_ROUNDS = 4

SYSTEM = (
    "You are a research specialist. You have tools for LIVE news and Wikipedia. "
    "Always start by calling get_latest_news for the topic. Use wikipedia_current_events "
    "if useful and wikipedia_search for background facts. Base time-sensitive statements "
    "on tool results and mention source names and dates."
)


def _to_openai_tools(mcp_tools) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


async def _research_async(llm, research_prompt: str) -> Tuple[str, List[str]]:
    used: List[str] = []

    async with create_connected_server_and_client_session(news_server) as session:
        tools = (await session.list_tools()).tools
        openai_tools = _to_openai_tools(tools)

        messages = [SystemMessage(content=SYSTEM), HumanMessage(content=research_prompt)]

        for round_no in range(MAX_TOOL_ROUNDS):
            # First round: force at least one tool call. Later rounds: model decides.
            choice = "required" if round_no == 0 else "auto"
            ai = await llm.bind_tools(openai_tools, tool_choice=choice).ainvoke(messages)
            messages.append(ai)
            if not getattr(ai, "tool_calls", None):
                return str(ai.content), used

            for call in ai.tool_calls:
                used.append(call["name"])
                try:
                    result = await session.call_tool(call["name"], call["args"])
                    text = "\n".join(
                        c.text for c in result.content if getattr(c, "text", None)
                    ) or json.dumps(str(result))
                except Exception as exc:  # noqa: BLE001
                    text = f"Tool error: {exc}"
                messages.append(ToolMessage(content=text, tool_call_id=call["id"]))

        # Out of rounds: force a final answer without tools
        final = await llm.ainvoke(messages)
        return str(final.content), used


def research_with_mcp(llm, research_prompt: str) -> Tuple[str, List[str]]:
    """Sync wrapper. Returns (research_text, tools_used). Raises on failure."""
    return asyncio.run(_research_async(llm, research_prompt))
