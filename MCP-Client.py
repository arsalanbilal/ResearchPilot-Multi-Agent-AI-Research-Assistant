"""
MCP CLIENT + HOST LOGIC  (used only by the Researcher agent)
------------------------------------------------------------
Host   : the Streamlit app / Researcher agent. It owns the LLM and decides
         when tools are used.
Client : an MCP ClientSession opened by the host. It launches
         mcp_news_server.py as a subprocess (stdio), lists its tools and
         forwards the LLM's tool calls to it.
Server : mcp_news_server.py (same folder)
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = str(Path(__file__).with_name("mcp_news_server.py"))
MAX_TOOL_ROUNDS = 4

SYSTEM = (
    "You are a research specialist. You have tools for LIVE news and Wikipedia. "
    "Use get_latest_news (and wikipedia_current_events if useful) when the topic involves "
    "recent, latest, today, breaking or live information. Use wikipedia_search for background "
    "facts. Base time-sensitive statements on tool results and mention source names and dates."
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
    params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])
    used: List[str] = []

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            llm_tools = llm.bind_tools(_to_openai_tools(tools))

            messages = [SystemMessage(content=SYSTEM), HumanMessage(content=research_prompt)]

            for _ in range(MAX_TOOL_ROUNDS):
                ai = await llm_tools.ainvoke(messages)
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
