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

import time
from databricks import agents
from databricks.sdk import WorkspaceClient

ENDPOINT_NAME = f"multi_genie_supervisor_{CATALOG}"
w = WorkspaceClient()

# Wait for endpoint to finish any in-progress update before deploying
try:
    ep = w.serving_endpoints.get(ENDPOINT_NAME)
    state = ep.state
    if state and "UPDATING" in str(state.config_update):
        print(f"Endpoint '{ENDPOINT_NAME}' is currently updating. Waiting...")
        for i in range(60):
            time.sleep(10)
            ep = w.serving_endpoints.get(ENDPOINT_NAME)
            state = ep.state
            if state and "UPDATING" not in str(state.config_update):
                print(f"  Update finished after ~{(i+1)*10}s")
                break
            print(f"  [{(i+1)*10}s] Still updating...")
        else:
            print("WARNING: Endpoint still updating after 10 minutes, attempting deploy anyway.")
except Exception:
    print(f"Endpoint '{ENDPOINT_NAME}' does not exist yet, will be created.")

deployment = agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    endpoint_name=ENDPOINT_NAME,
    tags={"endpointSource": "playground", "demo": "multi-genie-agent"},
    deploy_feedback_model=True,
)

print(f"Deployment initiated: {ENDPOINT_NAME}")
print(f"Model: {UC_MODEL_NAME} v{MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Endpoint to be Ready

# COMMAND ----------

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
        "input": [
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
