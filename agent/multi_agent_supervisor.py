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
from typing import Generator, Literal

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
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
LLM_ENDPOINT = cfg.get("llm_endpoint", "databricks-claude-sonnet-4-5")
MAX_ITERATIONS = cfg.get("max_iterations", 3)

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

    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)

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
        """Execute a GenieAgent worker and return its response."""
        result = agent.invoke({"messages": state["messages"]})
        last_msg = result["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
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

        result = supervisor_chain.invoke(state)
        next_node = result.next_node

        if state.get("next_node") == next_node and next_node != "FINISH":
            return {"next_node": "FINISH", "iteration_count": count}

        return {"next_node": next_node, "iteration_count": count}

    @mlflow.trace(span_type=SpanType.AGENT, name="final_answer")
    def final_answer_node(state: dict) -> dict:
        """Synthesize a final answer from all worker responses."""
        worker_messages = [
            m for m in state["messages"]
            if isinstance(m, dict) and m.get("name") in WORKER_DESCRIPTIONS
        ]

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

        user_question = state["messages"][0]
        question_text = user_question.get("content", "") if isinstance(user_question, dict) else str(user_question)

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
        from databricks_ai_bridge.utils.auth import ModelServingUserCredentials

        return WorkspaceClient(
            credentials_strategy=ModelServingUserCredentials()
        )

    @mlflow.trace(span_type=SpanType.AGENT, name="multi_genie_supervisor")
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=request.custom_inputs
        )

    @mlflow.trace(span_type=SpanType.AGENT, name="multi_genie_supervisor_stream")
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        user_client = self._get_obo_client()
        graph = build_graph(user_client)

        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

        for _, events in graph.stream(
            {"messages": cc_msgs}, stream_mode=["updates"]
        ):
            for node_data in events.values():
                msgs = node_data.get("messages", [])
                if msgs:
                    yield from output_to_responses_items_stream(msgs)


# ---------------------------------------------------------------------------
# MLflow model registration
# ---------------------------------------------------------------------------
mlflow.langchain.autolog()
AGENT = MultiGenieAgentSupervisor()
mlflow.models.set_model(AGENT)
