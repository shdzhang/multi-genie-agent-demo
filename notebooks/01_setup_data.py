# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Setup Synthetic Data
# MAGIC
# MAGIC Generates 12 tables across 3 business domains:
# MAGIC 1. **Sales & Revenue Analytics** - orders, customers, products, daily_sales_summary
# MAGIC 2. **HR & People Analytics** - employees, departments, performance_reviews, headcount_history
# MAGIC 3. **Supply Chain & Operations** - shipments, warehouses, inventory_levels, supplier_performance

# COMMAND ----------

# MAGIC %pip install faker
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "shidong_catalog")
dbutils.widgets.text("schema_name", "multi_genie_demo")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")

print(f"Catalog: {CATALOG}, Schema: {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Catalog and Schema

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Domain 1: Sales & Revenue Analytics

# COMMAND ----------

import random
from datetime import datetime, timedelta
from faker import Faker
from pyspark.sql import functions as F
from pyspark.sql.types import *

fake = Faker()
Faker.seed(42)
random.seed(42)

# COMMAND ----------

# --- Products (200 rows) ---
CATEGORIES = ["Electronics", "Software", "Services", "Hardware", "Accessories"]
product_rows = []
for i in range(1, 201):
    cat = random.choice(CATEGORIES)
    base_price = random.lognormvariate(4.5, 1.0)
    product_rows.append({
        "product_id": f"PROD-{i:04d}",
        "product_name": f"{fake.word().title()} {cat} {fake.word().title()}",
        "category": cat,
        "subcategory": fake.word().title(),
        "unit_price": round(min(base_price, 5000.0), 2),
        "cost_price": round(min(base_price * random.uniform(0.3, 0.7), 3500.0), 2),
        "launch_date": fake.date_between(start_date="-5y", end_date="-6M").isoformat(),
        "is_active": random.random() > 0.1,
    })

products_df = spark.createDataFrame(product_rows)
products_df.write.mode("overwrite").saveAsTable("products")
spark.sql("""ALTER TABLE products SET TBLPROPERTIES (
    'comment' = 'Product catalog with pricing and category information'
)""")
print(f"products: {products_df.count()} rows")

# COMMAND ----------

# --- Customers (5K rows) ---
SEGMENTS = ["Enterprise", "SMB", "Startup", "Government", "Education"]
REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
customer_rows = []
for i in range(1, 5001):
    segment = random.choices(SEGMENTS, weights=[25, 35, 20, 10, 10])[0]
    customer_rows.append({
        "customer_id": f"CUST-{i:05d}",
        "company_name": fake.company(),
        "segment": segment,
        "region": random.choice(REGIONS),
        "country": fake.country(),
        "industry": fake.bs().split()[0].title(),
        "annual_revenue_usd": round(random.lognormvariate(14, 2.0), 2),
        "employee_count": int(random.lognormvariate(5, 1.5)),
        "created_date": fake.date_between(start_date="-5y", end_date="-1M").isoformat(),
        "is_active": random.random() > 0.05,
    })

customers_df = spark.createDataFrame(customer_rows)
customers_df.write.mode("overwrite").saveAsTable("customers")
spark.sql("""ALTER TABLE customers SET TBLPROPERTIES (
    'comment' = 'Customer accounts with segmentation and firmographic data'
)""")
print(f"customers: {customers_df.count()} rows")

# COMMAND ----------

# --- Orders (500K rows) ---
product_ids = [r["product_id"] for r in product_rows]
customer_ids = [r["customer_id"] for r in customer_rows]
product_prices = {r["product_id"]: r["unit_price"] for r in product_rows}

order_rows = []
base_date = datetime(2023, 1, 1)
for i in range(1, 500_001):
    prod_id = random.choice(product_ids)
    qty = max(1, int(random.lognormvariate(1.5, 1.0)))
    unit_price = product_prices[prod_id] * random.uniform(0.85, 1.15)
    discount = random.choices([0, 0.05, 0.10, 0.15, 0.20], weights=[50, 20, 15, 10, 5])[0]
    total = round(qty * unit_price * (1 - discount), 2)
    order_date = base_date + timedelta(days=random.randint(0, 730))
    status = random.choices(
        ["completed", "shipped", "processing", "cancelled", "returned"],
        weights=[60, 15, 10, 10, 5],
    )[0]
    order_rows.append((
        f"ORD-{i:07d}",
        random.choice(customer_ids),
        prod_id,
        order_date.strftime("%Y-%m-%d"),
        qty,
        round(unit_price, 2),
        discount,
        total,
        status,
        random.choice(REGIONS),
        fake.word().title(),
    ))

order_schema = StructType([
    StructField("order_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("product_id", StringType()),
    StructField("order_date", StringType()),
    StructField("quantity", IntegerType()),
    StructField("unit_price", DoubleType()),
    StructField("discount_pct", DoubleType()),
    StructField("total_amount", DoubleType()),
    StructField("status", StringType()),
    StructField("region", StringType()),
    StructField("sales_channel", StringType()),
])
orders_df = spark.createDataFrame(order_rows, schema=order_schema)
orders_df = orders_df.withColumn("order_date", F.to_date("order_date"))
orders_df.write.mode("overwrite").saveAsTable("orders")
spark.sql("""ALTER TABLE orders SET TBLPROPERTIES (
    'comment' = 'Sales orders with line-level detail including pricing, discounts, and fulfillment status'
)""")
print(f"orders: {orders_df.count()} rows")

# COMMAND ----------

# --- Daily Sales Summary (750 rows) ---
summary_rows = []
start = datetime(2023, 1, 1)
for d in range(750):
    dt = start + timedelta(days=d)
    weekday = dt.weekday()
    base_orders = random.gauss(670, 80) * (0.7 if weekday >= 5 else 1.0)
    base_revenue = base_orders * random.gauss(150, 30)
    summary_rows.append({
        "summary_date": dt.strftime("%Y-%m-%d"),
        "total_orders": max(0, int(base_orders)),
        "total_revenue": round(max(0, base_revenue), 2),
        "avg_order_value": round(max(0, base_revenue / max(1, base_orders)), 2),
        "unique_customers": max(0, int(base_orders * random.uniform(0.6, 0.85))),
        "new_customers": max(0, int(base_orders * random.uniform(0.05, 0.15))),
        "cancelled_orders": max(0, int(base_orders * random.uniform(0.05, 0.12))),
        "returned_orders": max(0, int(base_orders * random.uniform(0.02, 0.06))),
    })

summary_df = spark.createDataFrame(summary_rows)
summary_df = summary_df.withColumn("summary_date", F.to_date("summary_date"))
summary_df.write.mode("overwrite").saveAsTable("daily_sales_summary")
spark.sql("""ALTER TABLE daily_sales_summary SET TBLPROPERTIES (
    'comment' = 'Aggregated daily sales metrics including revenue, order counts, and customer activity'
)""")
print(f"daily_sales_summary: {summary_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Domain 2: HR & People Analytics

# COMMAND ----------

# --- Departments (20 rows) ---
DEPT_NAMES = [
    "Engineering", "Sales", "Marketing", "Finance", "HR",
    "Legal", "Operations", "Product", "Data Science", "Customer Success",
    "IT", "Security", "Design", "QA", "DevOps",
    "Business Development", "Research", "Procurement", "Facilities", "Executive",
]
dept_rows = []
for i, name in enumerate(DEPT_NAMES, 1):
    budget = random.lognormvariate(14.5, 0.8)
    dept_rows.append({
        "department_id": f"DEPT-{i:03d}",
        "department_name": name,
        "department_head": fake.name(),
        "budget_usd": round(budget, 2),
        "cost_center": f"CC-{random.randint(1000, 9999)}",
        "location": random.choice(["San Francisco", "New York", "London", "Berlin", "Singapore"]),
        "created_date": fake.date_between(start_date="-10y", end_date="-2y").isoformat(),
    })

departments_df = spark.createDataFrame(dept_rows)
departments_df.write.mode("overwrite").saveAsTable("departments")
spark.sql("""ALTER TABLE departments SET TBLPROPERTIES (
    'comment' = 'Organization departments with budget and leadership information'
)""")
print(f"departments: {departments_df.count()} rows")

# COMMAND ----------

# --- Employees (2K rows) ---
LEVELS = ["IC1", "IC2", "IC3", "IC4", "IC5", "Manager", "Sr Manager", "Director", "VP", "C-Suite"]
dept_ids = [r["department_id"] for r in dept_rows]
emp_rows = []
for i in range(1, 2001):
    hire_date = fake.date_between(start_date="-8y", end_date="-1M")
    level = random.choices(LEVELS, weights=[15, 25, 25, 15, 8, 5, 3, 2, 1.5, 0.5])[0]
    base_salary = {
        "IC1": 70000, "IC2": 90000, "IC3": 115000, "IC4": 140000, "IC5": 170000,
        "Manager": 150000, "Sr Manager": 175000, "Director": 200000, "VP": 250000, "C-Suite": 350000,
    }[level]
    salary = base_salary * random.uniform(0.85, 1.20)
    is_active = random.random() > 0.08
    term_date = None if is_active else fake.date_between(start_date=hire_date, end_date="today").isoformat()
    emp_rows.append({
        "employee_id": f"EMP-{i:05d}",
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "email": fake.email(),
        "department_id": random.choice(dept_ids),
        "job_title": fake.job(),
        "level": level,
        "hire_date": hire_date.isoformat(),
        "termination_date": term_date,
        "is_active": is_active,
        "salary_usd": round(salary, 2),
        "location": random.choice(["San Francisco", "New York", "London", "Berlin", "Singapore", "Remote"]),
        "manager_id": f"EMP-{random.randint(1, min(i, 100)):05d}" if i > 10 else None,
    })

employees_df = spark.createDataFrame(emp_rows)
employees_df.write.mode("overwrite").saveAsTable("employees")
spark.sql("""ALTER TABLE employees SET TBLPROPERTIES (
    'comment' = 'Employee records with demographics, compensation, and organizational hierarchy'
)""")
print(f"employees: {employees_df.count()} rows")

# COMMAND ----------

# --- Performance Reviews (10K rows) ---
emp_ids = [r["employee_id"] for r in emp_rows if r["is_active"]]
review_rows = []
for i in range(1, 10_001):
    rating = random.choices([1, 2, 3, 4, 5], weights=[3, 12, 40, 35, 10])[0]
    review_rows.append({
        "review_id": f"REV-{i:06d}",
        "employee_id": random.choice(emp_ids),
        "review_period": random.choice(["2023-H1", "2023-H2", "2024-H1", "2024-H2", "2025-H1"]),
        "reviewer_id": random.choice(emp_ids),
        "overall_rating": rating,
        "goal_completion_pct": float(round(min(100, max(0, random.gauss(rating * 18, 10))), 1)),
        "strengths": fake.sentence(),
        "areas_for_improvement": fake.sentence(),
        "review_date": fake.date_between(start_date="-2y", end_date="today").isoformat(),
    })

reviews_df = spark.createDataFrame(review_rows)
reviews_df.write.mode("overwrite").saveAsTable("performance_reviews")
spark.sql("""ALTER TABLE performance_reviews SET TBLPROPERTIES (
    'comment' = 'Employee performance reviews with ratings, goal completion, and qualitative feedback'
)""")
print(f"performance_reviews: {reviews_df.count()} rows")

# COMMAND ----------

# --- Headcount History (5K rows) ---
hc_rows = []
start = datetime(2022, 1, 1)
for i in range(5000):
    dt = start + timedelta(days=random.randint(0, 1095))
    dept = random.choice(DEPT_NAMES)
    base_hc = random.randint(20, 200)
    hc_rows.append({
        "snapshot_date": dt.strftime("%Y-%m-%d"),
        "department_name": dept,
        "total_headcount": base_hc,
        "active_headcount": int(base_hc * random.uniform(0.88, 0.98)),
        "new_hires": max(0, int(random.gauss(5, 3))),
        "terminations": max(0, int(random.gauss(2, 2))),
        "open_positions": max(0, int(random.gauss(8, 4))),
        "contractor_count": max(0, int(random.gauss(10, 5))),
    })

hc_df = spark.createDataFrame(hc_rows)
hc_df = hc_df.withColumn("snapshot_date", F.to_date("snapshot_date"))
hc_df.write.mode("overwrite").saveAsTable("headcount_history")
spark.sql("""ALTER TABLE headcount_history SET TBLPROPERTIES (
    'comment' = 'Point-in-time headcount snapshots by department including hires, terminations, and open roles'
)""")
print(f"headcount_history: {hc_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Domain 3: Supply Chain & Operations

# COMMAND ----------

# --- Warehouses (50 rows) ---
wh_rows = []
for i in range(1, 51):
    capacity = random.randint(5000, 50000)
    wh_rows.append({
        "warehouse_id": f"WH-{i:03d}",
        "warehouse_name": f"{fake.city()} Distribution Center",
        "region": random.choice(REGIONS),
        "country": fake.country(),
        "capacity_units": capacity,
        "current_utilization_pct": round(random.uniform(0.45, 0.98), 3),
        "operating_cost_monthly_usd": round(capacity * random.uniform(2, 8), 2),
        "manager_name": fake.name(),
        "opened_date": fake.date_between(start_date="-15y", end_date="-1y").isoformat(),
    })

warehouses_df = spark.createDataFrame(wh_rows)
warehouses_df.write.mode("overwrite").saveAsTable("warehouses")
spark.sql("""ALTER TABLE warehouses SET TBLPROPERTIES (
    'comment' = 'Warehouse locations with capacity, utilization, and operating cost data'
)""")
print(f"warehouses: {warehouses_df.count()} rows")

# COMMAND ----------

# --- Supplier Performance (5K rows) ---
supplier_ids = [f"SUP-{i:04d}" for i in range(1, 201)]
sp_rows = []
for i in range(1, 5001):
    lead_time = max(1, int(random.gauss(14, 5)))
    sp_rows.append({
        "record_id": f"SP-{i:05d}",
        "supplier_id": random.choice(supplier_ids),
        "supplier_name": fake.company(),
        "evaluation_date": fake.date_between(start_date="-2y", end_date="today").isoformat(),
        "on_time_delivery_pct": round(min(1.0, max(0.5, random.gauss(0.88, 0.08))), 3),
        "quality_score": round(min(5.0, max(1.0, random.gauss(3.8, 0.7))), 1),
        "avg_lead_time_days": lead_time,
        "defect_rate_pct": float(round(max(0, random.gauss(2.5, 1.5)), 2)),
        "total_orders": random.randint(10, 500),
        "total_spend_usd": round(random.lognormvariate(11, 1.5), 2),
        "category": random.choice(["Raw Materials", "Components", "Packaging", "Logistics", "Services"]),
    })

sp_df = spark.createDataFrame(sp_rows)
sp_df.write.mode("overwrite").saveAsTable("supplier_performance")
spark.sql("""ALTER TABLE supplier_performance SET TBLPROPERTIES (
    'comment' = 'Supplier scorecard metrics including delivery performance, quality, and spend'
)""")
print(f"supplier_performance: {sp_df.count()} rows")

# COMMAND ----------

# --- Shipments (300K rows) ---
wh_ids = [r["warehouse_id"] for r in wh_rows]
CARRIERS = ["FedEx", "UPS", "DHL", "USPS", "Maersk", "DB Schenker", "Kuehne+Nagel"]
ship_rows = []
for i in range(1, 300_001):
    ship_date = fake.date_between(start_date="-2y", end_date="today")
    transit_days = max(1, int(random.gauss(7, 3)))
    expected_delivery = ship_date + timedelta(days=transit_days)
    is_late = random.random() < 0.12
    actual_delivery = expected_delivery + timedelta(days=random.randint(1, 5)) if is_late else expected_delivery - timedelta(days=random.randint(0, 1))
    status = random.choices(
        ["delivered", "in_transit", "at_warehouse", "delayed", "returned"],
        weights=[65, 15, 8, 8, 4],
    )[0]
    ship_rows.append((
        f"SHIP-{i:07d}",
        random.choice(wh_ids),
        random.choice(supplier_ids),
        random.choice(CARRIERS),
        ship_date.isoformat(),
        expected_delivery.isoformat(),
        actual_delivery.isoformat() if status == "delivered" else None,
        status,
        round(random.lognormvariate(5, 1.2), 2),
        round(random.lognormvariate(3.5, 1.0), 2),
        random.choice(REGIONS),
    ))

ship_schema = StructType([
    StructField("shipment_id", StringType()),
    StructField("origin_warehouse_id", StringType()),
    StructField("supplier_id", StringType()),
    StructField("carrier", StringType()),
    StructField("ship_date", StringType()),
    StructField("expected_delivery_date", StringType()),
    StructField("actual_delivery_date", StringType()),
    StructField("status", StringType()),
    StructField("weight_kg", DoubleType()),
    StructField("shipping_cost_usd", DoubleType()),
    StructField("destination_region", StringType()),
])
shipments_df = spark.createDataFrame(ship_rows, schema=ship_schema)
shipments_df = (
    shipments_df
    .withColumn("ship_date", F.to_date("ship_date"))
    .withColumn("expected_delivery_date", F.to_date("expected_delivery_date"))
    .withColumn("actual_delivery_date", F.to_date("actual_delivery_date"))
)
shipments_df.write.mode("overwrite").saveAsTable("shipments")
spark.sql("""ALTER TABLE shipments SET TBLPROPERTIES (
    'comment' = 'Shipment tracking with carrier, delivery dates, cost, and on-time performance'
)""")
print(f"shipments: {shipments_df.count()} rows")

# COMMAND ----------

# --- Inventory Levels (100K rows) ---
inv_rows = []
for i in range(1, 100_001):
    max_stock = random.randint(100, 5000)
    current = int(max_stock * random.uniform(0.1, 1.0))
    reorder = int(max_stock * random.uniform(0.15, 0.3))
    inv_rows.append((
        f"INV-{i:06d}",
        random.choice(wh_ids),
        random.choice(product_ids),
        fake.date_between(start_date="-1y", end_date="today").isoformat(),
        current,
        max_stock,
        reorder,
        "low" if current < reorder else ("medium" if current < max_stock * 0.5 else "healthy"),
        round(current * random.uniform(5, 100), 2),
        round(random.uniform(2, 30), 1),
    ))

inv_schema = StructType([
    StructField("inventory_id", StringType()),
    StructField("warehouse_id", StringType()),
    StructField("product_id", StringType()),
    StructField("snapshot_date", StringType()),
    StructField("quantity_on_hand", IntegerType()),
    StructField("max_capacity", IntegerType()),
    StructField("reorder_point", IntegerType()),
    StructField("stock_status", StringType()),
    StructField("inventory_value_usd", DoubleType()),
    StructField("days_of_supply", DoubleType()),
])
inv_df = spark.createDataFrame(inv_rows, schema=inv_schema)
inv_df = inv_df.withColumn("snapshot_date", F.to_date("snapshot_date"))
inv_df.write.mode("overwrite").saveAsTable("inventory_levels")
spark.sql("""ALTER TABLE inventory_levels SET TBLPROPERTIES (
    'comment' = 'Inventory stock levels by warehouse and product with reorder points and supply metrics'
)""")
print(f"inventory_levels: {inv_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
print(f"\nCreated {len(tables)} tables in {CATALOG}.{SCHEMA}:")
for t in tables:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{t.tableName}").count()
    print(f"  {t.tableName}: {count:,} rows")

# COMMAND ----------

# Set task values for downstream notebooks
dbutils.jobs.taskValues.set(key="catalog_name", value=CATALOG)
dbutils.jobs.taskValues.set(key="schema_name", value=SCHEMA)
print("Task values set for downstream notebooks.")
