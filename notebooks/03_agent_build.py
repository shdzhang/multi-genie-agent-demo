# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Build & Log Multi-Genie Agent
# MAGIC
# MAGIC Logs the LangGraph multi-agent supervisor to MLflow with OBO AuthPolicy.

# COMMAND ----------

# MAGIC %pip install mlflow>=3.0 databricks-langchain langgraph databricks-agents databricks-ai-bridge
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "shidong_catalog")
dbutils.widgets.text("schema_name", "multi_genie_demo")
dbutils.widgets.text("model_name", "multi_genie_supervisor")
dbutils.widgets.text("llm_endpoint", "databricks-claude-sonnet-4-5")
dbutils.widgets.text("experiment_name", "")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
print(f"UC Model: {UC_MODEL_NAME}")
print(f"LLM Endpoint: {LLM_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve Genie Space IDs
# MAGIC
# MAGIC Get space IDs from the setup job's task values, or set them manually.

# COMMAND ----------

try:
    SALES_SPACE_ID = dbutils.jobs.taskValues.get(
        taskKey="create_genie_spaces", key="sales_genie_space_id"
    )
    HR_SPACE_ID = dbutils.jobs.taskValues.get(
        taskKey="create_genie_spaces", key="hr_genie_space_id"
    )
    SC_SPACE_ID = dbutils.jobs.taskValues.get(
        taskKey="create_genie_spaces", key="supply_chain_genie_space_id"
    )
    print(f"Sales Genie Space: {SALES_SPACE_ID}")
    print(f"HR Genie Space: {HR_SPACE_ID}")
    print(f"Supply Chain Genie Space: {SC_SPACE_ID}")
except Exception as e:
    print(f"Could not retrieve task values ({e}). Set space IDs manually below:")
    # SALES_SPACE_ID = "<your-sales-space-id>"
    # HR_SPACE_ID = "<your-hr-space-id>"
    # SC_SPACE_ID = "<your-supply-chain-space-id>"
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Agent Config

# COMMAND ----------

agent_config = {
    "sales_genie_space_id": SALES_SPACE_ID,
    "hr_genie_space_id": HR_SPACE_ID,
    "supply_chain_genie_space_id": SC_SPACE_ID,
    "llm_endpoint": LLM_ENDPOINT,
    "max_iterations": 3,
}

print("Agent config:")
for k, v in agent_config.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Model with OBO AuthPolicy

# COMMAND ----------

import mlflow
from mlflow.models.auth_policy import (
    AuthPolicy,
    SystemAuthPolicy,
    UserAuthPolicy,
)
from mlflow.models.resources import DatabricksServingEndpoint

# Set experiment
if EXPERIMENT_NAME:
    mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.set_registry_uri("databricks-uc")

# Input example for model signature
input_example = {
    "messages": [
        {"role": "user", "content": "What was total revenue last quarter?"}
    ]
}

# OBO AuthPolicy - the key difference from PAT-based approach
auth_policy = AuthPolicy(
    system_auth_policy=SystemAuthPolicy(
        resources=[
            DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
        ]
    ),
    user_auth_policy=UserAuthPolicy(
        api_scopes=["genie", "serving-endpoints"]
    ),
)

agent_file = "../agent/multi_agent_supervisor.py"

with mlflow.start_run(run_name="multi_genie_supervisor_obo") as run:
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=agent_file,
        model_config=agent_config,
        input_example=input_example,
        auth_policy=auth_policy,
        pip_requirements=[
            "mlflow>=3.0",
            "databricks-langchain>=0.5.0",
            "langgraph>=0.3.0",
            "databricks-agents>=0.17.0",
            "databricks-ai-bridge>=0.5.0",
            "pydantic>=2.0",
        ],
    )
    print(f"Logged model URI: {logged_agent_info.model_uri}")
    print(f"Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model in Unity Catalog

# COMMAND ----------

uc_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name=UC_MODEL_NAME,
)

print(f"Registered model: {UC_MODEL_NAME}")
print(f"Version: {uc_model_info.version}")

# Set alias
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name=UC_MODEL_NAME,
    alias="candidate",
    version=uc_model_info.version,
)
print(f"Set alias 'candidate' -> version {uc_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Task Values

# COMMAND ----------

dbutils.jobs.taskValues.set(key="model_uri", value=logged_agent_info.model_uri)
dbutils.jobs.taskValues.set(key="model_version", value=uc_model_info.version)
dbutils.jobs.taskValues.set(key="uc_model_name", value=UC_MODEL_NAME)
print("Task values set for deployment notebook.")
