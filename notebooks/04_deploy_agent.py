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

def _wait_for_endpoint_ready(w, name, timeout_minutes=15):
    """Block until endpoint leaves UPDATING / NOT_READY state."""
    for i in range(timeout_minutes * 6):
        try:
            ep = w.serving_endpoints.get(name)
        except Exception:
            return None
        state = ep.state
        updating = state and "IN_PROGRESS" in str(state.config_update)
        not_ready = state and "NOT_READY" in str(state.ready) and "READY" != str(state.ready)
        if not updating and not not_ready:
            return ep
        if i == 0:
            print(f"Endpoint '{name}' is not ready yet. Waiting...")
        if i % 6 == 0:
            print(f"  [{i*10}s] config_update={state.config_update}, ready={state.ready}")
        time.sleep(10)
    print(f"WARNING: Endpoint still not ready after {timeout_minutes} min.")
    return w.serving_endpoints.get(name)

existing_ep = _wait_for_endpoint_ready(w, ENDPOINT_NAME)

already_serving = False
if existing_ep is not None:
    current_entity = current_version = None
    if existing_ep.config and existing_ep.config.served_entities:
        se = existing_ep.config.served_entities[0]
        current_entity = se.entity_name
        current_version = se.entity_version
    print(f"Endpoint '{ENDPOINT_NAME}' already exists (serving {current_entity} v{current_version}).")
    if current_entity == UC_MODEL_NAME and str(current_version) == str(MODEL_VERSION):
        print(f"Already serving the target model+version. Skipping deploy.")
        already_serving = True
    else:
        print(f"Updating to {UC_MODEL_NAME} v{MODEL_VERSION} ...")
else:
    print(f"Endpoint '{ENDPOINT_NAME}' does not exist. Creating with {UC_MODEL_NAME} v{MODEL_VERSION} ...")

if not already_serving:
    deployment = agents.deploy(
        UC_MODEL_NAME,
        MODEL_VERSION,
        endpoint_name=ENDPOINT_NAME,
        tags={"endpointSource": "playground", "demo": "multi-genie-agent"},
        deploy_feedback_model=False,
    )
    action = "Updated" if existing_ep else "Created"
    print(f"{action} endpoint: {ENDPOINT_NAME}")
    print(f"Model: {UC_MODEL_NAME} v{MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Endpoint to be Ready

# COMMAND ----------

print(f"Waiting for endpoint '{ENDPOINT_NAME}' to be ready...")
for attempt in range(90):
    try:
        endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
        state = endpoint.state
        config_done = state and "IN_PROGRESS" not in str(state.config_update)
        is_ready = state and "NOT_READY" not in str(state.ready)
        if config_done and is_ready:
            print(f"Endpoint is READY after ~{attempt * 10}s")
            break
        if attempt % 6 == 0:
            print(f"  [{attempt * 10}s] config_update={state.config_update}, ready={state.ready}")
    except Exception as e:
        print(f"  [{attempt * 10}s] Waiting... ({e})")
    time.sleep(10)
else:
    print("WARNING: Endpoint did not become ready within 15 minutes.")

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
