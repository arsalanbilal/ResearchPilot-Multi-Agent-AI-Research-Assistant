import os
import re
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, MessagesState, StateGraph

from mcp_client import research_with_mcp

load_dotenv()

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

LANGSMITH_ENDPOINTS = {
    "US": "https://api.smith.langchain.com",
    "EU": "https://eu.api.smith.langchain.com",
}

ICONS = {"superviser": "🧭", "researcher": "🔎", "analyst": "📊", "writer": "✍️"}


class SuperviserState(MessagesState):
    next_agent: str
    current_task: str
    research_data: str
    analysis_data: str
    final_report: str


def build_graph(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Superviser managing a team of agents:

1. Researcher - Gathers information and data
2. Analyst - Analyzes data and provides insights
3. Writer - Creates reports and summaries

Based on the current state, decide which agent should work next.

Current state:
- Has research_data : {has_research}
- Has analysis : {has_analysis}

Respond with ONLY the agent name (researcher/analyst/writer) or 'Done'.""",
            ),
            ("human", "{task}"),
        ]
    )
    superviser_chain = prompt | llm

    def superviser_agent(state: SuperviserState):
        task = state.get("current_task") or state["messages"][0].content
        has_research = bool(state.get("research_data"))
        has_analysis = bool(state.get("analysis_data"))

        # The LLM is asked for its decision (shows up in the trace), but the
        # order is fixed in code so the pipeline can never loop.
        superviser_chain.invoke(
            {"task": task, "has_research": has_research, "has_analysis": has_analysis}
        )

        if not has_research:
            next_agent, text = "researcher", "Let's start the research. Assigning to Researcher."
        elif not has_analysis:
            next_agent, text = "analyst", "Research done. Time for analysis. Assigning to Analyst."
        else:
            next_agent, text = "writer", "Analysis completed. Let's create the report. Assigning to Writer."

        return {
            "messages": [AIMessage(content=f"Superviser : {text}")],
            "next_agent": next_agent,
            "current_task": task,
        }

    def researcher_agent(state: SuperviserState):
        task = state["current_task"]
        prompt = f"""As a research specialist, provide comprehensive information about: {task}

Include:
1. Key facts and background
2. Current trends or developments
3. Important statistics or data points
4. Notable examples or case studies

Be concise but thorough."""
        try:
            research, tools_used = research_with_mcp(llm, prompt)
        except Exception as exc:  # MCP server/network problem -> plain LLM answer
            research, tools_used = llm.invoke(prompt).content, []
            st.warning(f"MCP tools unavailable, answered from model knowledge only: {exc}")
        used = ", ".join(tools_used) if tools_used else "none"
        return {
            "messages": [AIMessage(content=f"Researcher : Research done on '{task}'. Tools used: {used}\n\n{research}")],
            "research_data": research,
        }

    def analyst_agent(state: SuperviserState):
        prompt = f"""As an Analyst, analyse the research data and provide insights:

Research Data:
{state['research_data']}

Provide:
1. Key insights and patterns
2. Strategic implications
3. Risks and opportunities
4. Recommendations

Focus on actionable insights related to: {state['current_task']}"""
        analysis = llm.invoke(prompt).content
        return {
            "messages": [AIMessage(content=f"Analyst : Analysis done.\n\n{analysis}")],
            "analysis_data": analysis,
        }

    def writer_agent(state: SuperviserState):
        task = state["current_task"]
        prompt = f"""As a Professional Writer, create an executive report based on:

Task: {task}

Research Findings:
{state['research_data'][:4000]}

Analysis:
{state['analysis_data'][:4000]}

Create a well structured report in Markdown with these sections:
1. Executive Summary
2. Key Findings
3. Analysis & Insights
4. Recommendations
5. Conclusion

Keep it professional and concise."""
        report = llm.invoke(prompt).content

        final_report = (
            f"# Final Report\n\n"
            f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}  \n"
            f"**Topic:** {task}\n\n---\n\n"
            f"{report}\n\n---\n\n"
            f"*Report compiled by a Multi-Agent AI system powered by Groq.*\n"
        )
        return {
            "messages": [AIMessage(content="Writer : Report complete! See the final report below.")],
            "final_report": final_report,
        }

    workflow = StateGraph(SuperviserState)
    workflow.add_node("superviser", superviser_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("writer", writer_agent)

    workflow.set_entry_point("superviser")
    workflow.add_conditional_edges(
        "superviser",
        lambda state: state["next_agent"],
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer"},
    )
    workflow.add_edge("researcher", "superviser")
    workflow.add_edge("analyst", "superviser")
    workflow.add_edge("writer", END)

    return workflow.compile()


def main():
    st.set_page_config(page_title="Supervisor Multi-AI-Agent", page_icon="🤖", layout="wide")
    st.title("🤖 Supervisor Multi-AI-Agent System")
    st.caption("Superviser → Researcher → Analyst → Writer  |  LangGraph + Groq + LangSmith")

    with st.sidebar:
        st.header("🔑 API Keys")
        groq_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        ls_key = st.text_input(
            "LangSmith API Key (optional)",
            type="password",
            value=os.getenv("LANGSMITH_API_KEY", ""),
            help="Leave empty to disable tracing.",
        )
        ls_project = st.text_input("LangSmith Project", value=os.getenv("LANGSMITH_PROJECT", "Supervisor-Multi-Agent"))
        ls_region = st.selectbox("LangSmith Region", list(LANGSMITH_ENDPOINTS))

        st.header("⚙️ Model")
        model_name = st.selectbox("Groq model", GROQ_MODELS)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

    task = st.text_area("What should the team work on?", value="What are the benefits of AI in Healthcare", height=100)

    if st.button("🚀 Run Multi-Agent Team", type="primary"):
        if not groq_key.strip():
            st.error("Please enter your Groq API key in the sidebar.")
            st.stop()
        if not task.strip():
            st.error("Please enter a task.")
            st.stop()

        # LangSmith tracing: LangChain / LangGraph pick these up automatically
        if ls_key.strip():
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = ls_key.strip()
            os.environ["LANGSMITH_PROJECT"] = ls_project.strip()
            os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINTS[ls_region]
        else:
            os.environ["LANGSMITH_TRACING"] = "false"

        llm = ChatGroq(model=model_name, api_key=groq_key.strip(), temperature=temperature, max_retries=2)
        graph = build_graph(llm)

        log, final_report = [], ""
        try:
            with st.status("Agents are working...", expanded=True) as status:
                for chunk in graph.stream(
                    {"messages": [HumanMessage(content=task.strip())]},
                    config={"recursion_limit": 25, "run_name": "Supervisor-Multi-Agent"},
                    stream_mode="updates",
                ):
                    for node, update in chunk.items():
                        for msg in update.get("messages", []):
                            st.markdown(f"**{ICONS[node]} {node.capitalize()}**")
                            st.write(msg.content)
                            log.append((node, msg.content))
                        if update.get("final_report"):
                            final_report = update["final_report"]
                status.update(label="✅ All agents finished", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            st.stop()

        st.session_state["log"] = log
        st.session_state["report"] = final_report

    if st.session_state.get("report"):
        st.divider()
        st.subheader("📄 Final Report")
        st.markdown(st.session_state["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=st.session_state["report"],
            file_name="final_report.md",
            mime="text/markdown",
        )
        with st.expander("Show agent conversation"):
            for node, content in st.session_state["log"]:
                st.markdown(f"**{ICONS[node]} {node.capitalize()}**")
                st.write(content)


if __name__ == "__main__":
    main()
