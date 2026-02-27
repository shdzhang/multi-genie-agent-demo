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

# 2. Generate data + create Genie spaces
databricks bundle run setup -t dev

# 3. Verify Genie spaces work in the UI

# 4. Build + deploy agent
databricks bundle run agent_build_deploy -t dev

# 5. Test the endpoint
databricks bundle run agent_build_deploy -t dev  # runs test_evaluate task
```

## Project Structure

```
multi-genie-agent-demo/
├── databricks.yml                    # Bundle config (variables, targets)
├── resources/
│   ├── model_artifacts.yml           # Schema, registered model, experiment
│   ├── setup_job.yml                 # Job: data gen + Genie creation
│   └── agent_job.yml                 # Job: agent build + deploy + test
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
