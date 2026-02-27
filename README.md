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
