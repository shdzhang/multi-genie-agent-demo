# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Test & Evaluate Multi-Genie Agent
# MAGIC
# MAGIC End-to-end testing across all 3 domains plus cross-domain queries.

# COMMAND ----------

# MAGIC %pip install "mlflow[databricks]>=3.1.0" databricks-agents pandas
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

import mlflow
from databricks.sdk import WorkspaceClient

if EXPERIMENT_NAME:
    mlflow.set_experiment(EXPERIMENT_NAME)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Endpoint Name

# COMMAND ----------

try:
    ENDPOINT_NAME = dbutils.jobs.taskValues.get(
        taskKey="deploy_agent", key="endpoint_name"
    )
except Exception:
    ENDPOINT_NAME = f"multi_genie_supervisor_{CATALOG}"

print(f"Testing endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Evaluation Dataset
# MAGIC
# MAGIC Each row has `inputs` (passed to `predict_fn`) and `expectations` (used by scorers).

# COMMAND ----------

eval_dataset = [
    # Sales & Revenue
    {
        "inputs": {"question": "What was total revenue last quarter?"},
        "expectations": {"expected_response": "A numerical revenue figure with currency, broken down or summarized by quarter."},
    },
    {
        "inputs": {"question": "Top 10 products by sales volume this year"},
        "expectations": {"expected_response": "A ranked list of products with their sales volumes or counts."},
    },
    {
        "inputs": {"question": "Show average order value by customer segment"},
        "expectations": {"expected_response": "Average order values segmented by customer type or category, with numerical values."},
    },
    # HR & People Analytics
    {
        "inputs": {"question": "What is our attrition rate by department?"},
        "expectations": {"expected_response": "Attrition or turnover percentages broken down by department."},
    },
    {
        "inputs": {"question": "Show average performance score by department"},
        "expectations": {"expected_response": "Performance scores or ratings averaged per department."},
    },
    {
        "inputs": {"question": "Headcount growth over last 12 months"},
        "expectations": {"expected_response": "Employee count trends or growth figures over the past year."},
    },
    # Supply Chain & Operations
    {
        "inputs": {"question": "What is our on-time delivery rate this month?"},
        "expectations": {"expected_response": "A percentage or rate indicating on-time delivery performance."},
    },
    {
        "inputs": {"question": "Which warehouses are over 90% capacity?"},
        "expectations": {"expected_response": "A list of warehouse names or IDs with capacity utilization above 90%."},
    },
    {
        "inputs": {"question": "Top suppliers by lead time reliability"},
        "expectations": {"expected_response": "A ranked list of suppliers with lead time or reliability metrics."},
    },
    # Cross-domain
    {
        "inputs": {"question": "How does our sales team headcount compare to revenue growth?"},
        "expectations": {"expected_response": "A comparison of sales headcount trends against revenue trends, with numbers."},
    },
    {
        "inputs": {"question": "What is the shipping cost impact on product margins?"},
        "expectations": {"expected_response": "Analysis of shipping costs relative to product margins, with numerical data."},
    },
]

print(f"Evaluation dataset: {len(eval_dataset)} test cases")
for row in eval_dataset:
    print(f"  Q: {row['inputs']['question']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define predict_fn
# MAGIC
# MAGIC Wraps the deployed agent endpoint so `mlflow.genai.evaluate()` can call it.
# MAGIC The function receives `**kwargs` unpacked from each row's `inputs` dict.

# COMMAND ----------

import time

def predict_fn(question: str) -> str:
    """Query the deployed ResponsesAgent and return the text answer.
    Retries on rate-limit (429) and OBO auth warmup errors."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = w.api_client.do(
                "POST",
                f"/serving-endpoints/{ENDPOINT_NAME}/invocations",
                body={"input": [{"role": "user", "content": question}]},
            )
            parts = []
            for item in result.get("output", []):
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") == "output_text":
                            parts.append(block.get("text", ""))
            return "".join(parts)
        except Exception as e:
            err = str(e)
            if "429" in err or "REQUEST_LIMIT_EXCEEDED" in err:
                wait = 2 ** (attempt + 1)
                print(f"Rate limited, retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            elif "model_serving_user_credentials" in err or "Unable to detect credentials" in err:
                wait = 60
                print(f"Endpoint OBO auth warming up, retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Custom Scorers

# COMMAND ----------

import re
from mlflow.genai.scorers import scorer

@scorer
def non_empty(outputs: str) -> bool:
    """Checks the agent returned a substantive answer, not empty or a refusal."""
    if not outputs or not outputs.strip():
        return False
    refusal_phrases = [
        "i wasn't able to find",
        "i don't have enough information",
        "i cannot answer",
        "no relevant information",
    ]
    lower = outputs.lower()
    return not any(phrase in lower for phrase in refusal_phrases)


@scorer
def contains_data(outputs: str) -> bool:
    """Checks the response includes numerical data (numbers, percentages, currency)."""
    if not outputs:
        return False
    return bool(re.search(r"\d+[\.,]?\d*[%$]?", outputs))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation
# MAGIC
# MAGIC Uses `mlflow.genai.evaluate()` with:
# MAGIC - **Correctness**: LLM judge that checks semantic match against `expected_response`
# MAGIC - **Guidelines (relevance)**: LLM judge that checks the answer is on-topic
# MAGIC - **Guidelines (no_hallucination)**: LLM judge that checks for fabricated claims
# MAGIC - **non_empty**: Code scorer that rejects empty or refusal responses
# MAGIC - **contains_data**: Code scorer that verifies numerical data is present

# COMMAND ----------

from mlflow.genai.scorers import Correctness, Guidelines

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[
        Correctness(),
        Guidelines(
            name="relevance",
            guidelines=(
                "The response must directly answer the user's question. "
                "It should not provide information about an unrelated topic."
            ),
        ),
        Guidelines(
            name="no_hallucination",
            guidelines=(
                "The response must only present data that could plausibly come "
                "from the underlying database. It should not fabricate specific "
                "numbers or claim to have data it cannot access."
            ),
        ),
        non_empty,
        contains_data,
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Results
# MAGIC
# MAGIC Results are logged as an MLflow evaluation run. Open the experiment in the
# MAGIC MLflow UI to see per-row scores, LLM judge rationales, and aggregated metrics.

# COMMAND ----------

print("Evaluation complete.")
if EXPERIMENT_NAME:
    print(f"  Experiment: {EXPERIMENT_NAME}")
print(f"  Endpoint:   {ENDPOINT_NAME}")

try:
    print("\nAggregated metrics:")
    for k, v in eval_results.metrics.items():
        print(f"  {k}: {v}")
except AttributeError:
    print("\nView detailed results in the MLflow experiment UI (Traces tab).")
