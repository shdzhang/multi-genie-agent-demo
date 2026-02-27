# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Deploy Multi-Genie Agent
# MAGIC
# MAGIC Deploys the agent as a Model Serving endpoint using OBO (no PAT env vars needed).

# COMMAND ----------

# MAGIC %pip install mlflow>=3.0 databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "shidong_catalog")
dbutils.widgets.text("schema_name", "multi_genie_demo")
dbutils.widgets.text("model_name", "multi_genie_supervisor")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Model Version

# COMMAND ----------

try:
    MODEL_VERSION = dbutils.jobs.taskValues.get(
        taskKey="build_agent", key="model_version"
    )
    print(f"Model version from task values: {MODEL_VERSION}")
except Exception:
    from mlflow import MlflowClient
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    alias_info = client.get_model_version_by_alias(UC_MODEL_NAME, "candidate")
    MODEL_VERSION = alias_info.version
    print(f"Model version from 'candidate' alias: {MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Endpoint
# MAGIC
# MAGIC With OBO, no `environment_vars` are needed - the endpoint automatically
# MAGIC uses the calling user's credentials for Genie API access.

# COMMAND ----------

from databricks import agents

ENDPOINT_NAME = f"multi_genie_supervisor_{CATALOG}"

deployment = agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    endpoint_name=ENDPOINT_NAME,
    tags={"endpointSource": "playground", "demo": "multi-genie-agent"},
    # No environment_vars needed - OBO handles auth per-user
)

print(f"Deployment initiated: {ENDPOINT_NAME}")
print(f"Model: {UC_MODEL_NAME} v{MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Endpoint to be Ready

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

print(f"Waiting for endpoint '{ENDPOINT_NAME}' to be ready...")
for attempt in range(60):
    try:
        endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
        state = endpoint.state
        if state and str(state.ready) == "Ready.READY":
            print(f"Endpoint is READY after ~{attempt * 10}s")
            break
        print(f"  [{attempt * 10}s] State: {state}")
    except Exception as e:
        print(f"  [{attempt * 10}s] Waiting... ({e})")
    time.sleep(10)
else:
    print("WARNING: Endpoint did not become ready within 10 minutes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke Test

# COMMAND ----------

import mlflow

result = mlflow.deployments.predict(
    endpoint=ENDPOINT_NAME,
    inputs={
        "messages": [
            {"role": "user", "content": "What was total revenue last quarter?"}
        ]
    },
)

print("Smoke test result:")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Task Values

# COMMAND ----------

dbutils.jobs.taskValues.set(key="endpoint_name", value=ENDPOINT_NAME)
print(f"Endpoint deployed: {ENDPOINT_NAME}")
