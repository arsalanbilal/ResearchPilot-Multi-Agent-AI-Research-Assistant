"""
Supervisor Multi-AI-Agent System  -  Streamlit app
--------------------------------------------------
Supervisor -> Researcher -> Analyst -> Writer, orchestrated with LangGraph,
powered by Groq, optionally traced with LangSmith.

Run locally:
    pip install -r requirements.txt
    streamlit run superviser.py
"""

import os
import re
from datetime import datetime
from typing import Dict, Literal

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import LangChainTracer
from langchain_groq import ChatGroq
from langgraph.graph import END, MessagesState, StateGraph
from langsmith import Client

load_dotenv()

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "Custom (type below)",
]

AGENT_ICONS = {
    "superviser": "🧭",
    "researcher": "🔎",
    "analyst": "📊",
    "writer": "✍️",
}


# ---------------------------------------------------------------------------
# 1. STATE
# ---------------------------------------------------------------------------
class SuperviserState(MessagesState):
    """State for the Superviser Multi AI Agent."""

    next_agent: str
    research_data: str
    analysis_data: str
    final_report: str
    task_complete: bool
    current_task: str


# ---------------------------------------------------------------------------
# 2. GRAPH BUILDER (LLM is injected, so the user's Groq key is never global)
# ---------------------------------------------------------------------------
def build_graph(llm):
    """Builds and compiles the supervisor workflow using the given LLM."""

    superviser_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Superviser managing a team of agents:

1. Researcher - Gathers information and data
2. Analyst - Analyzes data and provides insights
3. Writer - Creates reports and summaries

Based on the current state and conversation, decide which agent should work next.

Current state:
- Has research_data : {has_research}
- Has analysis : {has_analysis}
- Has report : {has_report}

Respond with ONLY the agent name (researcher/analyst/writer) or 'Done'.""",
            ),
            ("human", "{task}"),
        ]
    )
    superviser_chain = superviser_prompt | llm

    # ----------------------------- AGENT 01 --------------------------------
    def superviser_agent(state: SuperviserState) -> Dict:
        """Superviser decides the next agent."""

        # Task = original user request (never an intermediate agent message)
        task = state.get("current_task", "")
        if not task:
            human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
            task = human_msgs[0].content if human_msgs else "No task"

        has_research = bool(state.get("research_data", ""))
        has_analysis = bool(state.get("analysis_data", ""))
        has_report = bool(state.get("final_report", ""))

        # Ask the LLM (advisory)
        decision = superviser_chain.invoke(
            {
                "task": task,
                "has_research": has_research,
                "has_analysis": has_analysis,
                "has_report": has_report,
            }
        )
        decision_text = str(decision.content).strip().lower()
        match = re.search(r"researcher|analyst|writer|done", decision_text)
        llm_choice = match.group(0) if match else ""

        # Guard-rail: enforce the correct pipeline order so it can never loop
        if has_report:
            required = "end"
        elif not has_research:
            required = "researcher"
        elif not has_analysis:
            required = "analyst"
        else:
            required = "writer"

        next_agent = required  # LLM choice is respected only when it is valid
        if llm_choice and llm_choice == required:
            next_agent = llm_choice

        messages = {
            "end": "Superviser : All tasks complete. Great work team!",
            "researcher": "Superviser : Let's start the research. Assigning to Researcher.",
            "analyst": "Superviser : Research done. Time for analysis. Assigning to Analyst.",
            "writer": "Superviser : Analysis completed. Let's create the report. Assigning to Writer.",
        }

        return {
            "messages": [AIMessage(content=messages[next_agent])],
            "next_agent": next_agent,
            "current_task": task,
        }

    # ----------------------------- AGENT 02 --------------------------------
    def researcher_agent(state: SuperviserState) -> Dict:
        task = state.get("current_task", "research topic")

        research_prompt = f"""As a research specialist, provide comprehensive information about: {task}

Include:
1. Key facts and background
2. Current trends or developments
3. Important statistics or data points
4. Notable examples or case studies

Be concise but thorough."""

        research_data = str(llm.invoke([HumanMessage(content=research_prompt)]).content)

        return {
            "messages": [
                AIMessage(
                    content=f"Researcher : I've completed the research on '{task}'.\n\n"
                    f"Key findings:\n{research_data}"
                )
            ],
            "research_data": research_data,
            "next_agent": "superviser",
        }

    # ----------------------------- AGENT 03 --------------------------------
    def analyst_agent(state: SuperviserState) -> Dict:
        research_data = state.get("research_data", "")
        task = state.get("current_task", "")

        analysis_prompt = f"""As an Analyst, analyse the research data and provide insights:

Research Data:
{research_data}

Provide:
1. Key insights and patterns
2. Strategic implications
3. Risks and opportunities
4. Recommendations

Focus on actionable insights related to: {task}"""

        analysis = str(llm.invoke([HumanMessage(content=analysis_prompt)]).content)

        return {
            "messages": [
                AIMessage(content=f"Analyst : I've completed the analysis.\n\nTop insights:\n{analysis}")
            ],
            "analysis_data": analysis,
            "next_agent": "superviser",
        }

    # ----------------------------- AGENT 04 --------------------------------
    def writer_agent(state: SuperviserState) -> Dict:
        research_data = state.get("research_data", "")
        analysis_data = state.get("analysis_data", "")
        task = state.get("current_task", "")

        writing_prompt = f"""As a Professional Writer, create an executive report based on:

Task: {task}

Research Findings:
{research_data[:4000]}

Analysis:
{analysis_data[:4000]}

Create a well structured report in Markdown with these sections:
1. Executive Summary
2. Key Findings
3. Analysis & Insights
4. Recommendations
5. Conclusion

Keep it professional and concise."""

        report = str(llm.invoke([HumanMessage(content=writing_prompt)]).content)

        final_report = (
            f"# Final Report\n\n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
            f"**Topic:** {task}\n\n---\n\n"
            f"{report}\n\n---\n\n"
            f"*Report compiled by a Multi-Agent AI system powered by Groq.*\n"
        )

        return {
            "messages": [AIMessage(content="Writer : Report complete! See the final report below.")],
            "final_report": final_report,
            "next_agent": "superviser",
            "task_complete": True,
        }

    # ------------------------------ ROUTER ---------------------------------
    def router(
        state: SuperviserState,
    ) -> Literal["superviser", "researcher", "analyst", "writer", "__end__"]:
        next_agent = state.get("next_agent", "superviser")

        if next_agent == "end" or state.get("task_complete", False):
            return END
        if next_agent in ["superviser", "researcher", "analyst", "writer"]:
            return next_agent
        return "superviser"

    # ----------------------------- WORKFLOW --------------------------------
    workflow = StateGraph(SuperviserState)
    workflow.add_node("superviser", superviser_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("writer", writer_agent)
    workflow.set_entry_point("superviser")

    for node in ["superviser", "researcher", "analyst", "writer"]:
        workflow.add_conditional_edges(
            node,
            router,
            {
                "superviser": "superviser",
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )

    return workflow.compile()


# ---------------------------------------------------------------------------
# 3. RUNNER
# ---------------------------------------------------------------------------
def run_workflow(graph, task: str, tracer=None):
    """Yields (node_name, update_dict) as each agent finishes."""
    config = {
        "recursion_limit": 25,
        "run_name": "Supervisor-Multi-Agent",
        "tags": ["streamlit", "supervisor-multi-agent"],
    }
    if tracer is not None:
        config["callbacks"] = [tracer]

    for chunk in graph.stream(
        {"messages": [HumanMessage(content=task)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            yield node_name, update or {}


LANGSMITH_ENDPOINTS = {
    "US (smith.langchain.com)": "https://api.smith.langchain.com",
    "EU (eu.smith.langchain.com)": "https://eu.api.smith.langchain.com",
    "Custom / self-hosted": "",
}


def make_langsmith_client(api_key: str, api_url: str, workspace_id: str = "", errors=None):
    """Creates a LangSmith client; background tracing errors are collected in `errors`."""
    kwargs = {"api_key": api_key, "api_url": api_url}
    if workspace_id:
        kwargs["workspace_id"] = workspace_id
    if errors is not None:
        kwargs["tracing_error_callback"] = lambda e: errors.append(str(e))
    try:
        return Client(**kwargs)
    except TypeError:  # older langsmith without these options
        kwargs.pop("workspace_id", None)
        kwargs.pop("tracing_error_callback", None)
        return Client(**kwargs)


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if "401" in low or "invalid api key" in low or "authentication" in low:
        return "Groq rejected the API key (401). Please check that it is correct and active."
    if "429" in low or "rate limit" in low:
        return "Groq rate limit reached (429). Wait a moment, or choose a smaller model."
    if "model" in low and ("not found" in low or "decommissioned" in low or "does not exist" in low):
        return "The selected Groq model is unavailable. Pick another model in the sidebar."
    return text


# ---------------------------------------------------------------------------
# 4. STREAMLIT UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Supervisor Multi-AI-Agent", page_icon="🤖", layout="wide")

    st.title("🤖 Supervisor Multi-AI-Agent System")
    st.caption("Superviser → Researcher → Analyst → Writer  |  LangGraph + Groq + LangSmith")

    # ---------------------------- SIDEBAR -----------------------------------
    with st.sidebar:
        st.header("🔑 API Keys")
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            placeholder="gsk_...",
            help="Get one free at https://console.groq.com/keys",
        )
        langsmith_key = st.text_input(
            "LangSmith API Key (optional)",
            type="password",
            value=os.getenv("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", "")),
            placeholder="lsv2_...",
            help="Enables tracing at https://smith.langchain.com. Leave empty to disable.",
        )
        langsmith_project = st.text_input(
            "LangSmith Project", value=os.getenv("LANGSMITH_PROJECT", "Supervisor-Multi-Agent"),
            help="Traces appear in THIS project (not 'default') on the LangSmith website.",
        )
        region = st.selectbox(
            "LangSmith Region",
            list(LANGSMITH_ENDPOINTS),
            help="Choose EU if you log in at eu.smith.langchain.com.",
        )
        ls_url = LANGSMITH_ENDPOINTS[region]
        if not ls_url:
            ls_url = st.text_input("LangSmith API URL", placeholder="https://your-host/api").strip()
        ls_workspace = st.text_input(
            "LangSmith Workspace ID (only if your key is org-scoped)",
            value=os.getenv("LANGSMITH_WORKSPACE_ID", ""),
        ).strip()

        if st.button("🔌 Test LangSmith connection"):
            if not langsmith_key.strip():
                st.warning("Enter a LangSmith API key first.")
            else:
                try:
                    c = make_langsmith_client(langsmith_key.strip(), ls_url, ls_workspace)
                    list(c.list_projects(limit=1))
                    st.success("LangSmith key and region are valid ✔")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"LangSmith connection failed: {exc}")

        st.header("⚙️ Model")
        model_choice = st.selectbox("Groq model", GROQ_MODELS, index=0)
        custom_model = ""
        if model_choice.startswith("Custom"):
            custom_model = st.text_input("Custom model id", placeholder="e.g. llama-3.3-70b-versatile")
        model_name = custom_model.strip() if model_choice.startswith("Custom") else model_choice
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

        st.info("Keys are used only for this session and are never written to disk.")

    # ----------------------------- MAIN -------------------------------------
    task = st.text_area(
        "What should the team work on?",
        value="What are the benefits of AI in Healthcare",
        height=100,
    )
    run_clicked = st.button("🚀 Run Multi-Agent Team", type="primary")

    if run_clicked:
        if not groq_key.strip():
            st.error("Please enter your Groq API key in the sidebar.")
            st.stop()
        if not task.strip():
            st.error("Please enter a task.")
            st.stop()
        if not model_name:
            st.error("Please enter a model id.")
            st.stop()

        llm = ChatGroq(
            model=model_name,
            api_key=groq_key.strip(),
            temperature=temperature,
            max_retries=2,
        )
        graph = build_graph(llm)

        tracer, ls_client, ls_errors = None, None, []
        project = langsmith_project.strip() or "Supervisor-Multi-Agent"
        if langsmith_key.strip():
            ls_client = make_langsmith_client(
                langsmith_key.strip(), ls_url, ls_workspace, errors=ls_errors
            )
            tracer = LangChainTracer(project_name=project, client=ls_client)

        log, final_report = [], ""
        try:
            with st.status("Agents are working...", expanded=True) as status:
                for node, update in run_workflow(graph, task.strip(), tracer):
                    for msg in update.get("messages", []):
                        icon = AGENT_ICONS.get(node, "🤖")
                        st.markdown(f"**{icon} {node.capitalize()}**")
                        st.write(msg.content)
                        log.append((node, msg.content))
                    if update.get("final_report"):
                        final_report = update["final_report"]
                status.update(label="✅ All agents finished", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Run failed: {friendly_error(exc)}")
            st.stop()
        finally:
            if ls_client is not None:
                try:
                    ls_client.flush()
                except Exception as exc:  # noqa: BLE001
                    ls_errors.append(str(exc))

        trace_msg, trace_ok = "", None
        if ls_client is not None:
            if ls_errors:
                trace_ok, trace_msg = False, "LangSmith tracing error: " + ls_errors[0]
            else:
                try:
                    runs = list(ls_client.list_runs(project_name=project, is_root=True, limit=1))
                    if runs:
                        trace_ok = True
                        trace_msg = f"Trace sent to LangSmith project '{project}' ✔"
                    else:
                        trace_ok = False
                        trace_msg = (
                            f"No trace found yet in project '{project}'. Wait a few seconds and "
                            "refresh LangSmith, and check the Region / Workspace ID in the sidebar."
                        )
                except Exception as exc:  # noqa: BLE001
                    trace_ok, trace_msg = False, f"Could not verify trace: {exc}"

        st.session_state["log"] = log
        st.session_state["report"] = final_report
        st.session_state["trace_ok"] = trace_ok
        st.session_state["trace_msg"] = trace_msg

    # --------------------------- RESULTS ------------------------------------
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
        if st.session_state.get("trace_ok") is True:
            st.success(st.session_state["trace_msg"])
        elif st.session_state.get("trace_ok") is False:
            st.warning(st.session_state["trace_msg"])

        with st.expander("Show agent conversation"):
            for node, content in st.session_state.get("log", []):
                st.markdown(f"**{AGENT_ICONS.get(node, '🤖')} {node.capitalize()}**")
                st.write(content)


if __name__ == "__main__":
    main()
