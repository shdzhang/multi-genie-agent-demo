# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Create Genie Spaces
# MAGIC
# MAGIC Creates 3 Genie spaces via the Databricks SDK:
# MAGIC 1. Sales & Revenue Analytics
# MAGIC 2. HR & People Analytics
# MAGIC 3. Supply Chain & Operations

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "shidong_catalog")
dbutils.widgets.text("schema_name", "multi_genie_demo")
dbutils.widgets.text("warehouse_id", "")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")

print(f"Catalog: {CATALOG}, Schema: {SCHEMA}, Warehouse: {WAREHOUSE_ID or '(auto-detect)'}")

# COMMAND ----------

import json
import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve SQL Warehouse

# COMMAND ----------

if not WAREHOUSE_ID:
    warehouses = list(w.warehouses.list())
    # Prefer serverless, then pro, then classic; pick first running one
    running = [wh for wh in warehouses if str(wh.state) == "State.RUNNING"]
    if running:
        WAREHOUSE_ID = running[0].id
        print(f"Using running warehouse: {running[0].name} ({WAREHOUSE_ID})")
    elif warehouses:
        WAREHOUSE_ID = warehouses[0].id
        print(f"Using warehouse: {warehouses[0].name} ({WAREHOUSE_ID}) - may need to start")
    else:
        raise RuntimeError("No SQL warehouses found. Please provide a warehouse_id.")
else:
    print(f"Using provided warehouse: {WAREHOUSE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Genie Space Configurations

# COMMAND ----------

GENIE_CONFIGS = [
    {
        "title": "Sales & Revenue Analytics",
        "description": (
            "Analyze sales performance, revenue trends, customer segments, and product metrics. "
            "Includes order-level detail, daily aggregations, product catalog, and customer firmographics."
        ),
        "table_identifiers": [
            f"{CATALOG}.{SCHEMA}.orders",
            f"{CATALOG}.{SCHEMA}.customers",
            f"{CATALOG}.{SCHEMA}.products",
            f"{CATALOG}.{SCHEMA}.daily_sales_summary",
        ],
        "sample_questions": [
            "What was total revenue last quarter?",
            "Top 10 products by sales volume",
            "Revenue by region this year",
            "Average order value trend by month",
            "Which customer segment generates the most revenue?",
            "Show cancelled order rate over time",
        ],
    },
    {
        "title": "HR & People Analytics",
        "description": (
            "Explore workforce metrics including headcount trends, attrition rates, department budgets, "
            "performance distributions, and organizational structure."
        ),
        "table_identifiers": [
            f"{CATALOG}.{SCHEMA}.employees",
            f"{CATALOG}.{SCHEMA}.departments",
            f"{CATALOG}.{SCHEMA}.performance_reviews",
            f"{CATALOG}.{SCHEMA}.headcount_history",
        ],
        "sample_questions": [
            "What is our attrition rate by department?",
            "Average performance score by team",
            "Headcount growth over last 12 months",
            "Salary distribution by level",
            "Which departments have the most open positions?",
            "Show termination trends by quarter",
        ],
    },
    {
        "title": "Supply Chain & Operations",
        "description": (
            "Monitor supply chain performance including on-time delivery rates, inventory turnover, "
            "warehouse utilization, supplier lead times, and logistics costs."
        ),
        "table_identifiers": [
            f"{CATALOG}.{SCHEMA}.shipments",
            f"{CATALOG}.{SCHEMA}.warehouses",
            f"{CATALOG}.{SCHEMA}.inventory_levels",
            f"{CATALOG}.{SCHEMA}.supplier_performance",
        ],
        "sample_questions": [
            "What is our on-time delivery rate this month?",
            "Which warehouses are over 90% capacity?",
            "Top suppliers by lead time reliability",
            "Average shipping cost by carrier",
            "Show inventory levels at low stock status",
            "Total logistics spend by region",
        ],
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Genie Spaces via REST API
# MAGIC
# MAGIC Uses the Genie REST API to create spaces with table identifiers and sample questions.

# COMMAND ----------

host = w.config.host.rstrip("/")


def create_genie_space(config: dict) -> dict:
    """Create a Genie space using the SDK API client."""
    import uuid

    serialized_space = {
        "version": 2,
        "data_sources": {
            "tables": [{"identifier": t} for t in sorted(config["table_identifiers"])],
        },
        "config": {
            "sample_questions": [
                {"id": uuid.uuid4().hex, "question": [q]}
                for q in config["sample_questions"]
            ],
        },
    }

    body = {
        "title": config["title"],
        "description": config["description"],
        "warehouse_id": WAREHOUSE_ID,
        "serialized_space": json.dumps(serialized_space),
    }

    try:
        result = w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)
        print(f"Created Genie space: {config['title']} -> {result.get('space_id', 'N/A')}")
        return result
    except Exception as e:
        print(f"Error creating {config['title']}: {e}")
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create All Three Genie Spaces

# COMMAND ----------

space_ids = {}
space_key_map = {
    "Sales & Revenue Analytics": "sales_genie_space_id",
    "HR & People Analytics": "hr_genie_space_id",
    "Supply Chain & Operations": "supply_chain_genie_space_id",
}

for config in GENIE_CONFIGS:
    result = create_genie_space(config)
    space_id = result.get("space_id")
    key = space_key_map[config["title"]]
    space_ids[key] = space_id
    print(f"  {key} = {space_id}")
    time.sleep(2)  # Brief pause between API calls

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Task Values for Downstream Notebooks

# COMMAND ----------

for key, space_id in space_ids.items():
    dbutils.jobs.taskValues.set(key=key, value=space_id)
    print(f"Set task value: {key} = {space_id}")

dbutils.jobs.taskValues.set(key="warehouse_id", value=WAREHOUSE_ID)
print(f"Set task value: warehouse_id = {WAREHOUSE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

org_id = host.split("//")[1].split(".")[0]

print("\n=== Genie Spaces Created ===")
for key, space_id in space_ids.items():
    print(f"  {key}: {space_id}")
    print(f"  URL: {host}/genie/rooms/{space_id}?o={org_id}")
print(f"\nWarehouse ID: {WAREHOUSE_ID}")
print("\nNext step: Verify each Genie space works in the UI before building the agent.")
