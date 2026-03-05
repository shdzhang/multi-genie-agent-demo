"""
Multi-Genie Agent Supervisor

LangGraph-based supervisor that routes queries to 3 domain-specific Genie spaces:
  1. Sales & Revenue Analytics
  2. HR & People Analytics
  3. Supply Chain & Operations

Uses OBO (on-behalf-of-user) authentication so each request runs with the
calling user's credentials, enforcing Unity Catalog ACLs per-user.

Implements the MLflow ResponsesAgent interface for OpenAI Responses API
compatibility.
"""

import functools
import uuid
from typing import Generator, Literal

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from databricks_langchain.genie import GenieAgent

from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from mlflow.entities import SpanType

# ---------------------------------------------------------------------------
# Configuration loaded from MLflow model config
# ---------------------------------------------------------------------------
cfg = mlflow.models.ModelConfig()

SALES_GENIE_SPACE_ID = cfg.get("sales_genie_space_id")
HR_GENIE_SPACE_ID = cfg.get("hr_genie_space_id")
SUPPLY_CHAIN_GENIE_SPACE_ID = cfg.get("supply_chain_genie_space_id")
LLM_ENDPOINT = cfg.get("llm_endpoint")
MAX_ITERATIONS = cfg.get("max_iterations")

# ---------------------------------------------------------------------------
# Worker descriptions (used by the supervisor for routing)
# ---------------------------------------------------------------------------
WORKER_DESCRIPTIONS = {
    "SalesAgent": (
        "Handles questions about sales performance, revenue trends, order volumes, "
        "customer segments, product performance, regional breakdowns, average order value, "
        "and daily sales summaries."
    ),
    "HRAgent": (
        "Handles questions about workforce analytics, headcount trends, attrition rates, "
        "department budgets, performance reviews, salary distributions, hiring pipeline, "
        "and organizational structure."
    ),
    "SupplyChainAgent": (
        "Handles questions about supply chain operations, on-time delivery rates, "
        "inventory turnover, warehouse utilization, supplier lead times, logistics costs, "
        "shipping performance, and stock levels."
    ),
}

# ---------------------------------------------------------------------------
# Keyword-based direct routing for obvious single-domain queries.
# Skips the LLM supervisor call on the first iteration when the domain is
# unambiguous, saving ~1-2s latency and one LLM invocation.
# ---------------------------------------------------------------------------
DOMAIN_KEYWORDS = {
    "SalesAgent": {
        "revenue", "sales", "order", "orders", "customer segment", "product performance",
        "average order value", "aov", "top products", "sales volume", "margin",
        "margins", "discount", "pricing", "customer", "customers", "purchase",
    },
    "HRAgent": {
        "headcount", "attrition", "turnover", "employee", "employees", "hiring",
        "recruitment", "salary", "salaries", "compensation", "performance review",
        "performance score", "department budget", "workforce", "hr", "people",
        "organizational", "retention", "onboarding",
    },
    "SupplyChainAgent": {
        "supply chain", "delivery", "warehouse", "warehouses", "inventory",
        "supplier", "suppliers", "lead time", "logistics", "shipping", "freight",
        "stock", "fulfillment", "procurement", "capacity", "on-time",
    },
}


def _try_keyword_route(question: str) -> str | None:
    """Return agent name if exactly one domain matches, else None."""
    lower = question.lower()
    matched = [
        agent_name
        for agent_name, keywords in DOMAIN_KEYWORDS.items()
        if any(kw in lower for kw in keywords)
    ]
    unique = set(matched)
    if len(unique) == 1:
        return unique.pop()
    return None


SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor routing user questions to specialized data agents.

Available agents:
{worker_info}

Rules:
1. Analyze the user's question and route to the SINGLE most appropriate agent.
2. If the question spans multiple domains, pick the primary domain first.
3. After receiving an agent's response, decide whether to route to another agent or finish.
4. When you have sufficient information, select FINISH.
5. NEVER route to the same agent twice in a row.
"""


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class SupervisorState(MessagesState):
    next_node: str = ""
    iteration_count: int = 0


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------
def build_graph(user_client: WorkspaceClient) -> StateGraph:
    """Construct a LangGraph with OBO GenieAgent workers + supervisor."""

    llm = ChatDatabricks(endpoint=LLM_ENDPOINT, max_retries=3)

    sales_agent = GenieAgent(
        genie_space_id=SALES_GENIE_SPACE_ID,
        genie_agent_name="SalesAgent",
        description=WORKER_DESCRIPTIONS["SalesAgent"],
        client=user_client,
    )

    hr_agent = GenieAgent(
        genie_space_id=HR_GENIE_SPACE_ID,
        genie_agent_name="HRAgent",
        description=WORKER_DESCRIPTIONS["HRAgent"],
        client=user_client,
    )

    supply_chain_agent = GenieAgent(
        genie_space_id=SUPPLY_CHAIN_GENIE_SPACE_ID,
        genie_agent_name="SupplyChainAgent",
        description=WORKER_DESCRIPTIONS["SupplyChainAgent"],
        client=user_client,
    )

    agents = {
        "SalesAgent": sales_agent,
        "HRAgent": hr_agent,
        "SupplyChainAgent": supply_chain_agent,
    }

    options = list(WORKER_DESCRIPTIONS.keys()) + ["FINISH"]

    class NextNode(BaseModel):
        next_node: Literal[tuple(options)]

    worker_info = "\n".join(
        f"- {name}: {desc}" for name, desc in WORKER_DESCRIPTIONS.items()
    )
    system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(worker_info=worker_info)

    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    supervisor_chain = preprocessor | llm.with_structured_output(NextNode)

    def agent_node(state: dict, agent, name: str) -> dict:
        """Execute a GenieAgent worker and return its response.
        Catches errors so a single failing domain degrades gracefully."""
        try:
            result = agent.invoke({"messages": state["messages"]})
            last_msg = result["messages"][-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        except Exception as e:
            content = (
                f"[{name} unavailable] Unable to retrieve data from this domain. "
                f"Error: {type(e).__name__}: {e}"
            )
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": content,
                    "name": name,
                }
            ]
        }

    @mlflow.trace(span_type=SpanType.AGENT, name="supervisor")
    def supervisor_node(state: dict) -> dict:
        count = state.get("iteration_count", 0) + 1

        if count > MAX_ITERATIONS:
            return {"next_node": "FINISH", "iteration_count": count}

        # Fast path: keyword-based routing on the first iteration
        if count == 1:
            first_msg = state["messages"][0]
            question = (
                first_msg.get("content", "")
                if isinstance(first_msg, dict)
                else getattr(first_msg, "content", str(first_msg))
            )
            direct = _try_keyword_route(question)
            if direct:
                return {"next_node": direct, "iteration_count": count}

        result = supervisor_chain.invoke(state)
        next_node = result.next_node

        if state.get("next_node") == next_node and next_node != "FINISH":
            return {"next_node": "FINISH", "iteration_count": count}

        return {"next_node": next_node, "iteration_count": count}

    @mlflow.trace(span_type=SpanType.AGENT, name="final_answer")
    def final_answer_node(state: dict) -> dict:
        """Synthesize a final answer from all worker responses."""
        worker_messages = []
        for m in state["messages"]:
            name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            if name and name in WORKER_DESCRIPTIONS:
                worker_messages.append({"name": name, "content": content})

        if not worker_messages:
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I wasn't able to find the relevant information. Could you rephrase your question?",
                    }
                ]
            }

        worker_context = "\n\n".join(
            f"### {m['name']}:\n{m['content']}" for m in worker_messages
        )

        first_msg = state["messages"][0]
        question_text = first_msg.get("content", "") if isinstance(first_msg, dict) else getattr(first_msg, "content", str(first_msg))

        synthesis_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Synthesize the information from the "
                    "specialized agents below into a clear, concise final answer for the user. "
                    "If the data includes tables or numbers, preserve them. "
                    "Do not mention the internal agent names."
                ),
            },
            {
                "role": "user",
                "content": f"User question: {question_text}\n\nAgent responses:\n{worker_context}",
            },
        ]

        response = llm.invoke(synthesis_prompt)
        return {
            "messages": [
                {"role": "assistant", "content": response.content}
            ]
        }

    workflow = StateGraph(SupervisorState)

    for name, agent in agents.items():
        node_fn = functools.partial(agent_node, agent=agent, name=name)
        workflow.add_node(name, node_fn)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("final_answer", final_answer_node)

    workflow.set_entry_point("supervisor")

    for name in agents:
        workflow.add_edge(name, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_node"],
        {**{k: k for k in agents}, "FINISH": "final_answer"},
    )

    workflow.add_edge("final_answer", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# ResponsesAgent wrapper with OBO
# ---------------------------------------------------------------------------
class MultiGenieAgentSupervisor(ResponsesAgent):
    """
    ResponsesAgent that builds the LangGraph per-request using OBO credentials.

    Key OBO pattern: GenieAgents are created inside predict(), NOT at module
    level, so each request gets the calling user's downscoped token.

    Implements the OpenAI Responses API compatible interface.
    """

    def _get_obo_client(self) -> WorkspaceClient:
        from databricks_ai_bridge import ModelServingUserCredentials
        return WorkspaceClient(credentials_strategy=ModelServingUserCredentials())

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming: run graph to completion, return only the final answer."""
        user_client = self._get_obo_client()
        graph = build_graph(user_client)

        messages = to_chat_completions_input(
            [i.model_dump() for i in request.input]
        )

        result = graph.invoke({"messages": messages})

        final_messages = result.get("messages", [])
        answer = ""
        if final_messages:
            last = final_messages[-1]
            answer = last.content if hasattr(last, "content") else str(last)

        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(
                    text=answer, id=f"msg_{uuid.uuid4().hex[:8]}"
                )
            ]
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Stream graph execution, emitting agent-name annotations as each node
        runs so the Playground shows live progress during long Genie calls.
        """
        user_client = self._get_obo_client()
        graph = build_graph(user_client)

        messages = to_chat_completions_input(
            [i.model_dump() for i in request.input]
        )

        def _mid(m):
            return getattr(m, "id", None) or id(m)

        first_message = True
        seen_ids = set()

        for _, events in graph.stream(
            {"messages": messages}, stream_mode=["updates"]
        ):
            new_msgs = [
                msg
                for v in events.values()
                for msg in v.get("messages", [])
                if _mid(msg) not in seen_ids
            ]

            if first_message:
                seen_ids.update(_mid(msg) for msg in new_msgs[: len(messages)])
                new_msgs = new_msgs[len(messages) :]
                first_message = False
            else:
                seen_ids.update(_mid(msg) for msg in new_msgs)
                node_name = tuple(events.keys())[0]
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        text=f"<name>{node_name}</name>",
                        id=str(uuid.uuid4()),
                    ),
                )

            for msg in new_msgs:
                content = msg.content if hasattr(msg, "content") else str(msg)
                if isinstance(content, list):
                    import json
                    content = json.dumps(content, indent=2)
                if content:
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=self.create_text_output_item(
                            text=content, id=str(uuid.uuid4()),
                        ),
                    )


# ---------------------------------------------------------------------------
# MLflow model registration
# ---------------------------------------------------------------------------
mlflow.langchain.autolog()
AGENT = MultiGenieAgentSupervisor()
mlflow.models.set_model(AGENT)
