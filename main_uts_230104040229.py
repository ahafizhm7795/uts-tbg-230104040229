from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pandas as pd
import random
from datetime import datetime, timedelta
import os

# =========================
# CREATE SPARK SESSION
# =========================
spark = SparkSession.builder \
    .appName("Smart Hospital Monitoring System") \
    .getOrCreate()

# =========================
# GENERATE DUMMY DATA
# =========================
rooms = ["ICU", "Emergency", "Pharmacy"]

start_time = datetime.now()

data = []

for i in range(120):
    current_time = start_time + timedelta(minutes=i)

    for room in rooms:
        patient_count = random.randint(5, 80)

        data.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "room": room,
            "patient_count": patient_count
        })

# CREATE PANDAS DATAFRAME
pdf = pd.DataFrame(data)

# CONVERT TO SPARK DATAFRAME
df = spark.createDataFrame(pdf)

# CONVERT TIMESTAMP
df = df.withColumn(
    "timestamp",
    to_timestamp(col("timestamp"))
)

print("=== DATA SAMPLE ===")
df.show(5)

# =========================
# TRANSFORMATION 1
# TOTAL PATIENT PER ROOM
# =========================
patient_total = df.groupBy("room") \
    .agg(sum("patient_count").alias("total_patient"))

print("=== TOTAL PATIENT ===")
patient_total.show()

# =========================
# TRANSFORMATION 2
# TREND PER 15 MINUTES
# =========================
patient_time = df.withColumn(
    "minute_group",
    floor(minute(col("timestamp")) / 15) * 15
)

patient_time = patient_time.groupBy(
    hour(col("timestamp")).alias("hour"),
    col("minute_group"),
    col("room")
).agg(
    avg("patient_count").alias("avg_patient")
).orderBy("hour", "minute_group")

print("=== PATIENT TREND ===")
patient_time.show()

# =========================
# TRANSFORMATION 3
# ML DATASET
# =========================
ml_data = df.withColumn(
    "hour",
    hour(col("timestamp"))
).select(
    "hour",
    "patient_count"
)

print("=== ML DATA ===")
ml_data.show(5)

# =========================
# ABSOLUTE PATH
# =========================
base_path = os.getcwd()

patient_total_path = f"{base_path}/output/patient_total"
patient_time_path = f"{base_path}/output/patient_time"
ml_data_path = f"{base_path}/output/ml_data"

# =========================
# SAVE PARQUET
# =========================
patient_total.write.mode("overwrite").parquet(patient_total_path)

patient_time.write.mode("overwrite").parquet(patient_time_path)

ml_data.write.mode("overwrite").parquet(ml_data_path)

print("\n=== PARQUET SUCCESSFULLY SAVED ===")
print(patient_total_path)
print(patient_time_path)
print(ml_data_path)

# STOP SPARK
spark.stop()