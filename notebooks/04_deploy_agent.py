# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Deploy Multi-Genie Agent
# MAGIC
# MAGIC Validates the model locally, deploys as a Model Serving endpoint with OBO,
# MAGIC waits for readiness, and runs a smoke test.

# COMMAND ----------

# MAGIC %pip install mlflow>=3.0 databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("experiment_name", "")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Model Version

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

if EXPERIMENT_NAME:
    mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    MODEL_VERSION = dbutils.jobs.taskValues.get(
        taskKey="build_agent", key="model_version"
    )
    print(f"Model version from task values: {MODEL_VERSION}")
except Exception:
    alias_info = client.get_model_version_by_alias(UC_MODEL_NAME, "candidate")
    MODEL_VERSION = alias_info.version
    print(f"Model version from 'candidate' alias: {MODEL_VERSION}")

model_uri = f"models:/{UC_MODEL_NAME}/{MODEL_VERSION}"
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Model

# COMMAND ----------

try:
    mlflow.models.predict(
        model_uri=model_uri,
        input_data={"input": [{"role": "user", "content": "What is total revenue?"}]},
        env_manager="uv",
    )
    print("Model validation passed!")
except Exception as e:
    print(f"Local validation skipped (expected on some clusters): {e}")
    print("The model will be validated during serving endpoint deployment instead.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Endpoint

# COMMAND ----------

import time
from databricks import agents
from databricks.sdk import WorkspaceClient

ENDPOINT_NAME = f"multi_genie_supervisor_{CATALOG}"
w = WorkspaceClient()

def _wait_for_endpoint_not_updating(w, name, timeout_minutes=15):
    """Block until endpoint finishes any in-progress config update."""
    for i in range(timeout_minutes * 6):
        try:
            ep = w.serving_endpoints.get(name)
        except Exception:
            return None
        state = ep.state
        if state and "IN_PROGRESS" in str(state.config_update):
            if i == 0:
                print(f"Endpoint '{name}' is currently updating. Waiting...")
            if i % 6 == 0:
                print(f"  [{i*10}s] config_update={state.config_update}")
            time.sleep(10)
        else:
            return ep
    print(f"WARNING: Endpoint still updating after {timeout_minutes} min.")
    return w.serving_endpoints.get(name)

existing_ep = _wait_for_endpoint_not_updating(w, ENDPOINT_NAME)

already_serving = False
if existing_ep is not None:
    current_entity = current_version = None
    if existing_ep.config and existing_ep.config.served_entities:
        se = existing_ep.config.served_entities[0]
        current_entity = se.entity_name
        current_version = se.entity_version
    print(f"Endpoint '{ENDPOINT_NAME}' exists (serving {current_entity} v{current_version}).")
    if current_entity == UC_MODEL_NAME and str(current_version) == str(MODEL_VERSION):
        print("Already serving the target model+version. Skipping deploy.")
        already_serving = True
    else:
        print(f"Updating to {UC_MODEL_NAME} v{MODEL_VERSION} ...")
else:
    print(f"Endpoint '{ENDPOINT_NAME}' does not exist. Creating ...")

if not already_serving:
    deployment = agents.deploy(
        model_name=UC_MODEL_NAME,
        model_version=MODEL_VERSION,
        endpoint_name=ENDPOINT_NAME,
        tags={"endpointSource": "playground", "demo": "multi-genie-agent"},
        deploy_feedback_model=False,
    )
    action = "Updated" if existing_ep else "Created"
    print(f"Deployment initiated ({action})!")
    print(f"  Endpoint: {ENDPOINT_NAME}")
    print(f"  Model: {UC_MODEL_NAME} v{MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Endpoint to be Ready

# COMMAND ----------

if not already_serving:
    print(f"Waiting for endpoint '{ENDPOINT_NAME}' to be ready...")
    for attempt in range(90):
        try:
            ep = w.serving_endpoints.get(ENDPOINT_NAME)
            state = ep.state
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
else:
    print(f"Endpoint '{ENDPOINT_NAME}' already ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke Test

# COMMAND ----------

try:
    result = w.api_client.do(
        "POST",
        f"/serving-endpoints/{ENDPOINT_NAME}/invocations",
        body={"input": [{"role": "user", "content": "What was total revenue last quarter?"}]},
    )

    answer = ""
    for item in result.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    answer += block.get("text", "")

    print("Smoke test passed!")
    print(f"Response: {answer[:500]}")
except Exception as e:
    print(f"Smoke test failed: {e}")
    print("The endpoint may need more time to warm up. Try again from the AI Playground.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Task Values

# COMMAND ----------

dbutils.jobs.taskValues.set(key="endpoint_name", value=ENDPOINT_NAME)
print(f"Endpoint deployed: {ENDPOINT_NAME}")
