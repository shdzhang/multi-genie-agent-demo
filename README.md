# Multi-Genie Agent Demo

A reusable demo showing how to build a **multi-agent supervisor** on top of multiple Genie spaces using **LangGraph**, **OBO (on-behalf-of-user) authentication**, and **Databricks Asset Bundles**.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Query                            │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Keyword Router │  (fast path: skips LLM if
              │ _try_keyword_  │   single domain is obvious)
              │    route()     │
              └───────┬────────┘
                      │
           ┌──── hit ─┤─── miss ──┐
           │          │           │
           │          ▼           │
           │  ┌───────────────┐   │
           │  │  Supervisor   │   │
           │  │  (LLM call)   │   │
           │  │  Structured   │   │
           │  │  output →     │   │
           │  │  route/FINISH │   │
           │  └───────┬───────┘   │
           │          │           │
           ▼          ▼           ▼
     ┌──────────┐ ┌─────────┐ ┌────────────────┐
     │  Sales   │ │   HR    │ │  Supply Chain  │
     │  Genie   │ │  Genie  │ │    Genie       │
     │  Agent   │ │  Agent  │ │    Agent       │
     └──────────┘ └─────────┘ └────────────────┘
           │          │           │
           └──────────┼───────────┘
                      ▼
              ┌───────────────┐
              │ Final Answer  │  (LLM synthesizes all
              │    Node       │   worker responses)
              └───────────────┘
```

## Key Features

- **Multi-agent supervisor** — LangGraph routes queries to domain-specific Genie spaces via an LLM supervisor with structured output
- **Keyword-based direct routing** — Obvious single-domain queries bypass the LLM supervisor, saving ~1-2s latency and one LLM call
- **Graceful error handling** — Individual GenieAgent failures degrade gracefully instead of crashing the entire pipeline
- **OBO authentication** — Each request runs with the calling user's credentials, enforcing per-user Unity Catalog ACLs
- **MLflow ResponsesAgent** — OpenAI Responses API compatible interface with both `predict()` and `predict_stream()` methods
- **mlflow.genai.evaluate()** — Automated evaluation with LLM judges (Correctness, Guidelines) and custom code scorers
- **Multi-cloud deployment** — DABs targets for both GCP and Azure workspaces with per-target variable overrides
- **Idempotent deployment** — Endpoint creation skips re-deploy when already serving the target model+version
- **Rate limit resilience** — `max_retries=3` on `ChatDatabricks` and retry logic in evaluation for FMAPI rate limits
- **Databricks Asset Bundles** — Single `databricks bundle deploy && databricks bundle run demo` for the full pipeline

## Three Business Domains

| Domain | Tables | Key Metrics |
|--------|--------|-------------|
| **Sales & Revenue** | orders (500K), customers (5K), products (200), daily_sales_summary (750) | Revenue, order volumes, customer segments, product performance |
| **HR & People** | employees (2K), departments (20), performance_reviews (10K), headcount_history (5K) | Attrition, headcount, performance ratings, salary distributions |
| **Supply Chain** | shipments (300K), warehouses (50), inventory_levels (100K), supplier_performance (5K) | On-time delivery, warehouse utilization, inventory turnover, supplier scores |

## OBO vs PAT Authentication

This demo uses **OBO (on-behalf-of-user)** rather than a shared PAT:

| Aspect | PAT (legacy) | OBO (this demo) |
|--------|-------------|------------------|
| Auth init | `w.tokens.create()` at module level | `ModelServingUserCredentials()` inside `predict()` |
| Security | Shared token, no per-user ACLs | Per-user downscoped tokens, UC ACLs enforced |
| MLflow logging | `resources=[DatabricksGenieSpace(...)]` | `auth_policy=AuthPolicy(system_auth_policy=..., user_auth_policy=...)` |
| Deployment | `environment_vars={"DATABRICKS_GENIE_PAT": ...}` | No env vars needed |

## Prerequisites

1. **OBO enabled** on the workspace (Public Preview, admin must enable)
2. MLflow >= 3.0
3. Users need: `CAN RUN` on Genie spaces + `SELECT` on tables + `CAN USE` on SQL warehouse
4. Databricks CLI configured for the target workspace

## Quick Start

```bash
# 1. Deploy the bundle (creates schema, registered model, experiment, and job)
databricks bundle deploy -t prod

# 2. Run the full end-to-end pipeline (data → Genie spaces → agent → deploy → evaluate)
databricks bundle run demo -t prod
```

### Multi-Cloud Targets

| Target | Profile | Catalog | Workspace |
|--------|---------|---------|-----------|
| `dev` | `azure-szwestus-stable` | `szwestus_stable` | Azure (West US) |
| `prod` (default) | `fe-prod-dbx-ws` | `shidong_catalog` | GCP |

Override variables per target in `databricks.yml`. The `catalog_name` variable is overridden for `dev` to match the Azure workspace's catalog.

## Project Structure

```
multi-genie-agent-demo/
├── databricks.yml                    # Bundle config (variables, multi-cloud targets)
├── resources/
│   ├── model_artifacts.yml           # Schema, registered model, experiment (with resource refs)
│   └── demo_job.yml                  # Single job: data → Genie → agent → deploy → evaluate
├── notebooks/
│   ├── 01_setup_data.py              # Create catalog/schema, generate 12 tables (Azure-safe)
│   ├── 02_create_genie_spaces.py     # Create/update 3 Genie spaces via REST API
│   ├── 03_agent_build.py             # Log ResponsesAgent to MLflow with OBO AuthPolicy
│   ├── 04_deploy_agent.py            # Deploy to Model Serving (idempotent, wait for ready)
│   └── 05_test_and_evaluate.py       # mlflow.genai.evaluate() with LLM judges + custom scorers
├── agent/
│   └── multi_agent_supervisor.py     # LangGraph supervisor + keyword routing + OBO + ResponsesAgent
└── README.md
```

### Pipeline Flow (demo_job.yml)

All parameters (`catalog_name`, `schema_name`, `model_name`, `llm_endpoint`, `experiment_name`) are passed from bundle variables via `base_parameters` — notebooks have no hardcoded defaults.

```
generate_data → create_genie_spaces → build_agent → deploy_agent → test_evaluate
```

## Customization

To adapt this demo for a different customer or use case:

1. **Change domains**: Edit `01_setup_data.py` to generate different tables
2. **Change Genie spaces**: Edit `02_create_genie_spaces.py` with new table configs and sample questions
3. **Change worker descriptions**: Edit `WORKER_DESCRIPTIONS` in `agent/multi_agent_supervisor.py`
4. **Change workspace**: Update `targets` in `databricks.yml`
5. **Change LLM**: Update `llm_endpoint` variable in `databricks.yml`

## Troubleshooting

### `[CANNOT_MERGE_TYPE] Can not merge type 'DoubleType' and 'LongType'`

**Symptom**: `spark.createDataFrame(rows)` fails when creating tables without an explicit schema.

**Root cause**: Python's `min()` and `max()` preserve the type of whichever argument wins the comparison. When mixing int literals with float values, some rows get `int` and others get `float`:

```python
# These return int:
max(0, -3.5)   # → 0 (int, because 0 > -3.5)
min(100, 105.3) # → 100 (int, because 100 < 105.3)

# These return float:
max(0, 3.5)    # → 3.5 (float)
min(100, 95.3) # → 95.3 (float)
```

Spark's schema inference sees `LongType` for some rows and `DoubleType` for others and refuses to merge them.

**Fix**: Wrap the result in `float()` to ensure consistent typing, or provide an explicit `StructType` schema to `createDataFrame()`.

```python
# Before (broken):
"value": round(min(100, max(0, random.gauss(50, 10))), 1)

# After (fixed):
"value": float(round(min(100, max(0, random.gauss(50, 10))), 1))
```

### `HTTP 401: Credential was not sent or was of an unsupported type`

**Symptom**: REST API calls fail with 401 when running notebooks as a job, even though SDK calls like `w.warehouses.list()` work fine.

**Root cause**: When a notebook runs as a Databricks job, `WorkspaceClient()` uses `runtime` authentication (implicit cluster auth). This auth type does not expose a raw token via `w.config.token` (it returns `None`). Manually constructing `Authorization: Bearer None` headers causes a 401.

**Fix**: Use the SDK's built-in API client instead of manual HTTP requests. `w.api_client.do()` handles authentication correctly regardless of the auth method (PAT, OAuth, runtime).

```python
# Before (broken in job context):
token = w.config.token
req.add_header("Authorization", f"Bearer {token}")
urllib.request.urlopen(req)

# After (works everywhere):
result = w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)
```

### `Invalid serialized_space: Cannot find field: table_identifiers`

**Symptom**: Genie Space creation API returns `InvalidParameterValue` rejecting the `serialized_space` payload.

**Root cause**: The `serialized_space` JSON must follow the v2 `GenieSpaceExport` protobuf schema. The correct structure uses `data_sources.tables[].identifier`, not a top-level `table_identifiers` array.

**Fix**: Use the correct v2 payload structure:

```python
serialized_space = {
    "version": 2,
    "data_sources": {
        "tables": [{"identifier": t} for t in sorted(table_list)],
    },
    "config": {
        "sample_questions": [
            {"id": uuid.uuid4().hex, "question": [q]}
            for q in questions
        ],
    },
}
```

### `Invalid export proto: data_sources.tables must be sorted by identifier`

**Symptom**: Genie Space creation fails even with the correct v2 payload structure.

**Root cause**: The API requires `data_sources.tables` to be sorted alphabetically by the `identifier` field.

**Fix**: Sort the table identifiers before building the payload:

```python
"tables": [{"identifier": t} for t in sorted(config["table_identifiers"])]
```

### `ModelConfig.get() takes 2 positional arguments but 3 were given`

**Symptom**: Agent fails to load during `mlflow.pyfunc.log_model` or at serving time.

**Root cause**: `mlflow.models.ModelConfig.get()` only accepts a key name — it does not support a default-value argument.

**Fix**: Remove the default argument and handle missing keys separately:

```python
# Before (broken):
cfg.get("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct")

# After (fixed):
cfg.get("llm_endpoint")
```

### `ModuleNotFoundError: No module named 'databricks_ai_bridge.utils.auth'`

**Symptom**: Agent import fails at serving time with `ModuleNotFoundError: No module named 'databricks_ai_bridge.utils.auth'`.

**Root cause**: The import path `databricks_ai_bridge.utils.auth` does not exist. `ModelServingUserCredentials` lives directly under `databricks_ai_bridge`.

**Fix**: Use the correct import path:

```python
# Before (broken):
from databricks_ai_bridge.utils.auth import ModelServingUserCredentials

# After (fixed):
from databricks_ai_bridge import ModelServingUserCredentials
```

### `Invalid user API scope(s): genie, serving-endpoints`

**Symptom**: `mlflow.pyfunc.log_model` rejects the `auth_policy` with invalid scope names.

**Root cause**: API scope names are namespaced. The short forms `genie` and `serving-endpoints` are not valid.

**Fix**: Use the fully-qualified scope names:

```python
# Before (broken):
api_scopes=["genie", "serving-endpoints"]

# After (fixed):
api_scopes=["dashboards.genie", "serving.serving-endpoints"]
```

### `ValueError: Endpoint ... is currently updating`

**Symptom**: `agents.deploy()` fails because the endpoint is still processing a previous deployment.

**Root cause**: Re-running the job while the endpoint is mid-update causes a conflict. The deploy call does not wait for the previous update to finish.

**Fix**: Poll the endpoint state before deploying and wait for `IN_PROGRESS` to clear. Do **not** check for `"UPDATING"` — that substring matches `"NOT_UPDATING"` and causes an infinite loop (see next entry).

```python
ep = w.serving_endpoints.get(ENDPOINT_NAME)
while "IN_PROGRESS" in str(ep.state.config_update):
    time.sleep(10)
    ep = w.serving_endpoints.get(ENDPOINT_NAME)
```

### `Unable to get space ... Node with resource name does not exist`

**Symptom**: Deployed endpoint returns `InternalError` / `BAD_REQUEST` when the agent tries to query a Genie space. The error mentions `Node with resource name Some(datarooms/<space_id>) does not exist`.

**Root cause**: Genie spaces must be declared as `DatabricksGenieSpace` resources in the `auth_policy`'s `system_auth_policy.resources`. Without this, the serving runtime cannot resolve the Genie space resource for OBO token scoping, even if the correct `api_scopes` are set.

**Fix**: Add all Genie spaces alongside the LLM endpoint in `system_auth_policy.resources`:

```python
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksGenieSpace

auth_policy = AuthPolicy(
    system_auth_policy=SystemAuthPolicy(
        resources=[
            DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
            DatabricksGenieSpace(genie_space_id=SALES_SPACE_ID),
            DatabricksGenieSpace(genie_space_id=HR_SPACE_ID),
            DatabricksGenieSpace(genie_space_id=SC_SPACE_ID),
        ]
    ),
    user_auth_policy=UserAuthPolicy(
        api_scopes=["dashboards.genie", "serving.serving-endpoints"]
    ),
)
```

### Endpoint state check stuck in infinite loop (`NOT_UPDATING` contains `UPDATING`)

**Symptom**: The deploy notebook hangs forever waiting for the endpoint to finish updating, even though the UI shows the endpoint is Ready.

**Root cause**: `str(state.config_update)` returns `"NOT_UPDATING"` when the endpoint is idle. The check `"UPDATING" in "NOT_UPDATING"` is `True` (substring match), so the wait loop never exits.

**Fix**: Check for `"IN_PROGRESS"` instead, which only matches when an update is actually happening:

```python
# Before (broken - matches NOT_UPDATING too):
"UPDATING" in str(state.config_update)

# After (fixed):
"IN_PROGRESS" in str(state.config_update)
```

### `ValueError: Endpoint already serves model ... version N`

**Symptom**: `agents.deploy()` fails when re-running the job because the endpoint already has the same model+version deployed.

**Root cause**: `agents.deploy()` does not accept a no-op update — it raises an error if the endpoint already serves the exact same model and version.

**Fix**: Check the currently served entity before calling `agents.deploy()` and skip if it matches:

```python
if existing_ep.config and existing_ep.config.served_entities:
    se = existing_ep.config.served_entities[0]
    if se.entity_name == UC_MODEL_NAME and str(se.entity_version) == str(MODEL_VERSION):
        print("Already serving target version. Skipping deploy.")
        already_serving = True
```

### `AttributeError: module 'mlflow' has no attribute 'deployments'`

**Symptom**: Smoke test fails with `mlflow.deployments.predict()`.

**Root cause**: `mlflow.deployments` is not available in all environments. The Databricks SDK provides a direct way to query serving endpoints.

**Fix**: Use `w.api_client.do()` to query the endpoint:

```python
result = w.api_client.do(
    "POST",
    f"/serving-endpoints/{ENDPOINT_NAME}/invocations",
    body={"input": [{"role": "user", "content": "test question"}]},
)
```

### ResponsesAgent returns empty `output: []`

**Symptom**: Endpoint returns `{"object":"response","output":[],"id":"..."}` with no content. The Playground shows "Received an invalid and unexpected value from the API: undefined".

**Root cause**: The `predict_stream` method used `output_to_responses_items_stream(msgs)` to convert LangGraph messages to Responses API events. This function could not handle plain dict messages from LangGraph nodes, so it yielded zero events.

**Fix**: Use `self.create_text_output_item()` instead of `output_to_responses_items_stream()` to construct response items. See the "Streaming output" entry below for the full current pattern.

### All responses return "I wasn't able to find the relevant information"

**Symptom**: The agent runs without errors but every query gets the fallback response, regardless of domain.

**Root cause**: LangGraph's `MessagesState` reducer automatically converts the plain dicts returned by `agent_node` (e.g., `{"role": "assistant", "content": "...", "name": "SalesAgent"}`) into LangChain `AIMessage` objects. The `final_answer_node` filtered with `isinstance(m, dict)`, which always returned `False` for `AIMessage` objects. So `worker_messages` was always empty.

**Fix**: Handle both dicts and LangChain message objects using `getattr` fallback:

```python
worker_messages = []
for m in state["messages"]:
    name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
    content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
    if name and name in WORKER_DESCRIPTIONS:
        worker_messages.append({"name": name, "content": content})
```

### Genie agents return "permission restrictions" despite user having access

**Symptom**: The agent returns apology messages about "permission restrictions" or "data access issues" even though the user owns the tables and the Genie spaces work correctly when queried directly in the UI.

**Root cause**: `WorkspaceClient()` without a `credentials_strategy` uses the endpoint's **system service principal**, not the calling user's OBO token. The service principal typically lacks `SELECT` on the data tables. To use the user's identity, you must explicitly pass `ModelServingUserCredentials()`.

**Fix**: Use `ModelServingUserCredentials` from `databricks_ai_bridge`:

```python
from databricks_ai_bridge import ModelServingUserCredentials

# Before (uses service principal - no table access):
user_client = WorkspaceClient()

# After (uses calling user's OBO token):
user_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
```

This must be called inside `predict()` / `predict_stream()`, not at module level, because OBO credentials are only available during request handling.

### Traces appear under notebook experiment instead of shared experiment

**Symptom**: MLflow traces from test runs appear under the notebook's auto-experiment (e.g., `.bundle/.../04_deploy_agent`) instead of the shared project experiment.

**Root cause**: Each Databricks notebook gets an auto-experiment by default. Unless you explicitly call `mlflow.set_experiment()` before any MLflow operations, all traces and runs go to the notebook's auto-experiment.

**Fix**: Pass the shared `experiment_name` variable to all notebooks via the job config and call `mlflow.set_experiment()` early in each notebook:

```python
# In each notebook, before any mlflow operations:
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
if EXPERIMENT_NAME:
    mlflow.set_experiment(EXPERIMENT_NAME)
```

In `demo_job.yml`, pass the variable to every task:

```yaml
base_parameters:
  experiment_name: ${resources.experiments.multi_genie_experiment.name}
```

### `AttributeError: 'dict' object has no attribute 'id'` during model logging

**Symptom**: `mlflow.pyfunc.log_model` fails during input validation with `AttributeError: 'dict' object has no attribute 'id'`.

**Root cause**: When using `graph.stream()` with `stream_mode=["updates"]`, node outputs are yielded as raw dicts *before* the `MessagesState` reducer converts them into LangChain message objects. Accessing `.id` on these dicts fails. This only affects custom graph nodes that return plain dicts in their `messages` list (e.g., `agent_node`, `final_answer_node`).

**Fix**: Use `getattr` with a fallback for safe id access:

```python
def _mid(m):
    return getattr(m, "id", None) or id(m)

new_msgs = [msg for ... if _mid(msg) not in seen_ids]
```

### Streaming output includes raw `<name>` tags and intermediate agent messages

**Symptom**: When `predict()` delegates to `predict_stream()`, the response includes all intermediate outputs — `<name>SalesAgent</name>`, raw dict representations, and intermediate supervisor messages — instead of a clean final answer.

**Root cause**: `predict_stream()` emits events for *every* node (agent names, intermediate responses, supervisor routing). If `predict()` collects all of these, the output is a messy concatenation of intermediate state.

**Fix**: Make `predict()` and `predict_stream()` independent methods, each running the graph separately:

```python
def predict(self, request):
    # Blocking: run graph to completion, return only the final answer
    result = graph.invoke({"messages": messages})
    last = result["messages"][-1]
    return ResponsesAgentResponse(
        output=[self.create_text_output_item(text=last.content, ...)]
    )

def predict_stream(self, request):
    # Streaming: emit <name> annotations + content as each node completes
    for _, events in graph.stream({"messages": messages}, stream_mode=["updates"]):
        node_name = tuple(events.keys())[0]
        yield event_with_text(f"<name>{node_name}</name>")
        for msg in new_msgs:
            yield event_with_text(msg.content)
```

`predict()` is used by API consumers (clean output). `predict_stream()` is used by AI Playground (live progress).

### `NotFound: Model version '1' does not exist` during deployment

**Symptom**: `agents.deploy()` fails with `NotFound: Model version '1' does not exist` even though the correct model version (e.g., v9) is being deployed.

**Root cause**: The serving endpoint was previously serving a feedback model (e.g., `shidong_catalog.multi_genie_demo.feedback v1`). If that model or its versions were deleted (e.g., by cleaning up experiments), `agents.deploy()` internally tries to reference the stale feedback model and fails.

**Fix**: Delete the stale serving endpoint and let `agents.deploy()` create a fresh one:

```bash
databricks serving-endpoints delete <endpoint_name> -p <profile>
```

Then re-run the deploy task.

### `cannot update mlflow experiment: Node ... is in the trash`

**Symptom**: `databricks bundle deploy` fails with `cannot update mlflow experiment: Node /Users/.../Trash/prod_multi_genie_supervisor is in the trash; refusing to move it`.

**Root cause**: The MLflow experiment was soft-deleted (moved to Trash). The bundle's Terraform state still references the old experiment ID, and Terraform tries to update it, but the workspace refuses to move a trashed node.

**Fix**: Restore the experiment from trash first, then redeploy:

```bash
# Find the experiment ID (check bundle deploy error message or use list-experiments)
databricks experiments restore-experiment <experiment_id> -p <profile>

# Then redeploy — the bundle will update the restored experiment in place
databricks bundle deploy -t <target> -p <profile> --force-lock
```

Do **not** delete the experiment after restoring — `delete-experiment` is a soft delete that puts it right back in the trash.

### `REQUEST_LIMIT_EXCEEDED` during `mlflow.pyfunc.log_model()`

**Symptom**: `mlflow.pyfunc.log_model()` fails with `RateLimitError: REQUEST_LIMIT_EXCEEDED: Exceeded workspace input tokens per minute rate limit for databricks-claude-sonnet-4-5`.

**Root cause**: `log_model()` with `input_example` automatically runs the example through the model for validation. For this agent, that triggers the full LangGraph pipeline: a supervisor LLM call, one or more GenieAgent calls, and a final synthesis LLM call. If the workspace rate limit is already near capacity, this validation exceeds it.

**Fix**: Add `max_retries=3` to `ChatDatabricks` in the agent code so transient rate limits are retried transparently. If the limit is persistently hit, wait and re-run the job.

```python
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, max_retries=3)
```

### `INVALID_STATE: Metastore storage root URL does not exist` (Azure Default Storage)

**Symptom**: `CREATE CATALOG IF NOT EXISTS` fails with `INVALID_STATE` on Azure workspaces with Default Storage enabled, even when the catalog already exists.

**Root cause**: Azure workspaces with Default Storage perform a storage root validation during `CREATE CATALOG`. This validation fails because Default Storage manages storage implicitly, but `IF NOT EXISTS` still triggers the check.

**Fix**: Wrap the catalog creation in a `try/except` that catches `INVALID_STATE` errors and continues:

```python
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception as e:
    if "INVALID_STATE" in str(e) or "storage root" in str(e).lower():
        print(f"Catalog '{CATALOG}' already exists or cannot be auto-created. Continuing.")
    else:
        raise
```

### `Unable to detect credentials for user authorization` (OBO warmup)

**Symptom**: Endpoint returns `model_serving_user_credentials auth: Unable to detect credentials for user authorization` immediately after deployment.

**Root cause**: Model Serving endpoints with OBO authentication require a warm-up period for the OBO authorization layer to initialize. Queries made in the first 1-2 minutes after deployment (smoke tests, evaluation calls) hit this window.

**Fix**: Retry with a 60-second backoff for OBO warmup errors. The evaluation notebook (`05_test_and_evaluate.py`) includes this logic:

```python
except Exception as e:
    err = str(e)
    if "model_serving_user_credentials" in err or "Unable to detect credentials" in err:
        wait = 60
        print(f"Endpoint OBO auth warming up, retrying in {wait}s...")
        time.sleep(wait)
```

The deploy notebook's smoke test treats this as expected behavior and prints a guidance message instead of failing the task.

### `cannot create registered model: Schema ... does not exist` (bundle deploy ordering)

**Symptom**: `databricks bundle deploy` fails because Terraform tries to create the registered model before the schema it belongs to.

**Root cause**: When both the schema and registered model are declared in the same bundle, Terraform may process them in parallel. Using `${var.schema_name}` in the registered model's `schema_name` field gives Terraform no hint that the schema resource must be created first.

**Fix**: Use a resource reference (`${resources.schemas.<key>.name}`) for `schema_name`, which creates an implicit Terraform dependency:

```yaml
registered_models:
  multi_genie_supervisor:
    catalog_name: ${var.catalog_name}
    schema_name: ${resources.schemas.demo_schema.name}  # implicit dependency
    name: ${var.model_name}
```
