# Multi-Genie Agent Demo

A reusable demo showing how to build a **multi-agent supervisor** on top of multiple Genie spaces using **LangGraph**, **OBO (on-behalf-of-user) authentication**, and **Databricks Asset Bundles**.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Query                            │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Supervisor (LangGraph)                       │
│  Analyzes query → routes to best agent → synthesizes     │
└──────┬──────────────┬──────────────────┬─────────────────┘
       │              │                  │
       ▼              ▼                  ▼
┌──────────┐  ┌──────────────┐  ┌────────────────┐
│  Sales   │  │     HR       │  │  Supply Chain  │
│  Genie   │  │    Genie     │  │    Genie       │
│  Space   │  │    Space     │  │    Space       │
└──────────┘  └──────────────┘  └────────────────┘
```

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
| MLflow logging | `resources=[DatabricksGenieSpace(...)]` | `auth_policy=AuthPolicy(user_auth_policy=...)` |
| Deployment | `environment_vars={"DATABRICKS_GENIE_PAT": ...}` | No env vars needed |

## Prerequisites

1. **OBO enabled** on the workspace (Public Preview, admin must enable)
2. MLflow >= 3.0
3. Users need: `CAN RUN` on Genie spaces + `SELECT` on tables + `CAN USE` on SQL warehouse
4. Databricks CLI configured for the target workspace

## Quick Start

```bash
# 1. Deploy the bundle
databricks bundle deploy -t dev

# 2. Run the full end-to-end pipeline (data → Genie spaces → agent → deploy → test)
databricks bundle run demo -t dev
```

## Project Structure

```
multi-genie-agent-demo/
├── databricks.yml                    # Bundle config (variables, targets)
├── resources/
│   ├── model_artifacts.yml           # Schema, registered model, experiment
│   └── demo_job.yml                  # Single job: data → Genie → agent → deploy → test
├── notebooks/
│   ├── 01_setup_data.py              # Create catalog/schema, generate 12 tables
│   ├── 02_create_genie_spaces.py     # Create 3 Genie spaces via REST API
│   ├── 03_agent_build.py             # Log multi-agent to MLflow with AuthPolicy
│   ├── 04_deploy_agent.py            # Register in UC, deploy endpoint
│   └── 05_test_and_evaluate.py       # E2E testing across all 3 domains
├── agent/
│   └── multi_agent_supervisor.py     # LangGraph + OBO agent code
└── README.md
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

**Symptom**: Agent import fails at serving time when trying to use `ModelServingUserCredentials`.

**Root cause**: With the `ResponsesAgent` interface and `auth_policy` configuration, OBO credentials are injected automatically. `WorkspaceClient()` auto-detects the credential type in the serving context — no manual credential strategy is needed.

**Fix**: Replace explicit credential handling with a plain `WorkspaceClient()`:

```python
# Before (broken):
from databricks_ai_bridge.utils.auth import ModelServingUserCredentials
creds = ModelServingUserCredentials()
return WorkspaceClient(credentials_strategy=creds)

# After (fixed):
return WorkspaceClient()
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

**Fix**: Poll the endpoint state before deploying and wait for `UPDATING` to clear:

```python
ep = w.serving_endpoints.get(ENDPOINT_NAME)
while "UPDATING" in str(ep.state.config_update):
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
