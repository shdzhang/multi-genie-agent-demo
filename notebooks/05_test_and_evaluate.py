# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Test & Evaluate Multi-Genie Agent
# MAGIC
# MAGIC End-to-end testing across all 3 domains plus cross-domain queries.
# MAGIC Verifies routing accuracy via MLflow traces.

# COMMAND ----------

# MAGIC %pip install mlflow>=3.0 databricks-agents pandas
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
# MAGIC ## Define Test Questions
# MAGIC
# MAGIC 11 test questions across all 3 domains + cross-domain queries.

# COMMAND ----------

TEST_CASES = [
    # Sales & Revenue (expected: SalesAgent)
    {
        "question": "What was total revenue last quarter?",
        "expected_domain": "SalesAgent",
        "domain": "Sales",
    },
    {
        "question": "Top 10 products by sales volume this year",
        "expected_domain": "SalesAgent",
        "domain": "Sales",
    },
    {
        "question": "Show average order value by customer segment",
        "expected_domain": "SalesAgent",
        "domain": "Sales",
    },
    # HR & People Analytics (expected: HRAgent)
    {
        "question": "What is our attrition rate by department?",
        "expected_domain": "HRAgent",
        "domain": "HR",
    },
    {
        "question": "Show average performance score by department",
        "expected_domain": "HRAgent",
        "domain": "HR",
    },
    {
        "question": "Headcount growth over last 12 months",
        "expected_domain": "HRAgent",
        "domain": "HR",
    },
    # Supply Chain & Operations (expected: SupplyChainAgent)
    {
        "question": "What is our on-time delivery rate this month?",
        "expected_domain": "SupplyChainAgent",
        "domain": "SupplyChain",
    },
    {
        "question": "Which warehouses are over 90% capacity?",
        "expected_domain": "SupplyChainAgent",
        "domain": "SupplyChain",
    },
    {
        "question": "Top suppliers by lead time reliability",
        "expected_domain": "SupplyChainAgent",
        "domain": "SupplyChain",
    },
    # Cross-domain (supervisor should pick the best match)
    {
        "question": "How does our sales team headcount compare to revenue growth?",
        "expected_domain": "HRAgent",  # Headcount is HR
        "domain": "Cross-domain",
    },
    {
        "question": "What is the shipping cost impact on product margins?",
        "expected_domain": "SupplyChainAgent",  # Shipping is supply chain
        "domain": "Cross-domain",
    },
]

print(f"Total test cases: {len(TEST_CASES)}")
for tc in TEST_CASES:
    print(f"  [{tc['domain']}] {tc['question']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Tests

# COMMAND ----------

import mlflow
import time

results = []

for i, tc in enumerate(TEST_CASES):
    print(f"\n--- Test {i+1}/{len(TEST_CASES)}: [{tc['domain']}] ---")
    print(f"Q: {tc['question']}")

    start_time = time.time()
    try:
        response = mlflow.deployments.predict(
            endpoint=ENDPOINT_NAME,
            inputs={"messages": [{"role": "user", "content": tc["question"]}]},
        )
        elapsed = time.time() - start_time

        # Extract the final answer
        if isinstance(response, dict) and "choices" in response:
            answer = response["choices"][0]["message"]["content"]
        elif isinstance(response, dict) and "messages" in response:
            answer = response["messages"][-1]["content"]
        else:
            answer = str(response)

        print(f"A: {answer[:200]}...")
        print(f"Time: {elapsed:.1f}s")

        results.append({
            "question": tc["question"],
            "domain": tc["domain"],
            "expected_agent": tc["expected_domain"],
            "answer_preview": answer[:300],
            "latency_s": round(elapsed, 1),
            "status": "success",
            "error": None,
        })

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR: {e}")
        results.append({
            "question": tc["question"],
            "domain": tc["domain"],
            "expected_agent": tc["expected_domain"],
            "answer_preview": None,
            "latency_s": round(elapsed, 1),
            "status": "error",
            "error": str(e),
        })

    time.sleep(2)  # Brief pause between requests

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

import pandas as pd

results_df = pd.DataFrame(results)

total = len(results_df)
success = len(results_df[results_df["status"] == "success"])
errors = len(results_df[results_df["status"] == "error"])
avg_latency = results_df[results_df["status"] == "success"]["latency_s"].mean()

print("=" * 60)
print("TEST RESULTS SUMMARY")
print("=" * 60)
print(f"Total tests:     {total}")
print(f"Successful:      {success}")
print(f"Errors:          {errors}")
print(f"Avg latency:     {avg_latency:.1f}s")
print("=" * 60)

display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results by Domain

# COMMAND ----------

domain_summary = (
    results_df.groupby("domain")
    .agg(
        tests=("status", "count"),
        successes=("status", lambda x: (x == "success").sum()),
        avg_latency=("latency_s", "mean"),
    )
    .reset_index()
)

display(domain_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: MLflow Evaluate
# MAGIC
# MAGIC Run `mlflow.genai.evaluate()` with built-in scorers for more rigorous evaluation.

# COMMAND ----------

# Uncomment to run MLflow evaluation:

# eval_dataset = [
#     {
#         "inputs": {"messages": [{"role": "user", "content": tc["question"]}]},
#         "expectations": {"expected_agent": tc["expected_domain"]},
#     }
#     for tc in TEST_CASES
# ]
#
# from mlflow.genai.scorers import Safety, Guidelines
#
# eval_results = mlflow.genai.evaluate(
#     data=eval_dataset,
#     predict_fn=lambda inputs: mlflow.deployments.predict(
#         endpoint=ENDPOINT_NAME, inputs=inputs
#     ),
#     scorers=[
#         Safety(),
#         Guidelines(
#             name="relevance",
#             guidelines="The response should directly answer the user's question with specific data or metrics.",
#         ),
#     ],
# )
#
# display(eval_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify MLflow Traces
# MAGIC
# MAGIC Check the MLflow experiment for traces showing supervisor routing decisions.

# COMMAND ----------

print(f"\nTo inspect traces, navigate to the MLflow experiment for model: {UC_MODEL_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")
print("\nLook for:")
print("  - 'supervisor' spans showing routing decisions")
print("  - 'SalesAgent', 'HRAgent', 'SupplyChainAgent' spans")
print("  - 'final_answer' spans with synthesized responses")
