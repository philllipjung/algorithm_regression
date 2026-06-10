#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTTM Random Forest Feature Importance Analysis
PySpark version with MinIO and Iceberg support for Kubernetes
"""

import os
import sys
import datetime
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer, IndexToString
from pyspark.sql.types import DoubleType


def main():
    # MinIO Configuration - use environment variable for Docker/Kubernetes
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9011")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

    # Configure Spark with Iceberg and MinIO
    spark = (SparkSession
            .builder
            .appName("TTTM_RandomForest")
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.iceberg.type", "hadoop")
            .config("spark.sql.catalog.iceberg.warehouse", "s3a://ic-ias/structured")
            .config("spark.sql.catalog.iceberg.io-impl", "org.apache.iceberg.hadoop.HadoopFileIO")
            # S3A / MinIO Configuration
            .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
            .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            .getOrCreate())

    # Parse arguments (format: jobid:xxx area:xxx)
    job_id = ""
    cluster_area = ""

    for i in range(1, len(sys.argv)):
        parts = sys.argv[i].split(":")
        if len(parts) == 2:
            name = parts[0]
            value = parts[1]
            if name == "jobid":
                job_id = value
            elif name == "area":
                cluster_area = value

    # Set defaults if not provided
    if not cluster_area:
        cluster_area = "local"

    # Get current date for dt partition (format: YYYYMMDD)
    dt_partition = datetime.datetime.now().strftime("%Y%m%d")

    print(f"=== TTTM Random Forest Analysis (K8s) ===")
    print(f"Job ID: {job_id}")
    print(f"Area: {cluster_area}")
    print(f"Partition (dt): {dt_partition}")

    # Read input file
    if cluster_area == "local":
        local_path = f"/root/{job_id}"
        if os.path.exists(local_path):
            print(f"[INFO] Reading from local file: {local_path}")
            input_path = local_path
        else:
            print(f"[INFO] Local file not found, reading from MinIO/S3")
            input_path = f"s3a://ic-ias/structured/{job_id}"
    else:
        # For remote clusters (ich, wxh), read from MinIO/S3
        input_path = f"s3a://ic-ias/structured/{job_id}"

    print(f"Input path: {input_path}")

    # Read input file (CSV format with header)
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("sep", ",") \
        .csv(input_path)

    print("\n=== Input Data Schema ===")
    df.printSchema()

    print("\n=== Input Data Sample (first 5 rows) ===")
    df.show(5, truncate=False)

    # Get column names
    col_name_list = df.columns
    print(f"\n=== Columns ({len(col_name_list)}) ===")
    print(f"Label column: {col_name_list[0]}")
    print(f"Feature columns: {col_name_list[1:]}")

    # Convert all columns to DoubleType
    for col in col_name_list:
        df = df.withColumn(col, df[col].cast(DoubleType()))

    # Assemble features
    feature_cols = col_name_list[1:]  # All columns except Label
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    # Index the label column
    label_indexer = StringIndexer(
        inputCol=col_name_list[0],  # Use first column as label
        outputCol="indexedLabel"
    )

    # Index features
    feature_indexer = VectorIndexer(
        inputCol="features",
        outputCol="indexedFeatures",
        maxCategories=4
    )

    # Train a RandomForest model
    rf = RandomForestClassifier(
        labelCol="indexedLabel",
        featuresCol="indexedFeatures",
        maxDepth=30,
        numTrees=500
    )

    # Convert indexed labels back to original labels
    label_converter = IndexToString(
        inputCol="prediction",
        outputCol="predictedLabel",
        labels=label_indexer.fit(df).labels
    )

    # Create pipeline (assembler must come first to create 'features' column)
    pipeline = Pipeline(stages=[assembler, label_indexer, feature_indexer, rf, label_converter])

    # Train model
    print("\n=== Training Random Forest ===")
    print(f"Parameters: maxDepth=30, numTrees=500")
    model = pipeline.fit(df)

    # Extract feature importances (rf is at stages[3] after assembler, label_indexer, feature_indexer)
    rf_model = model.stages[3]
    feature_importances = rf_model.featureImportances.toArray()

    print("\n=== Feature Importances ===")
    print(f"{'Feature':<30} {'Importance':<15}")
    print("-" * 45)

    # Build INSERT statement with dt partition
    insert_sql = "INSERT INTO iceberg.ias.tttm_rf VALUES "
    values = []

    for i, importance in enumerate(feature_importances):
        feature_name = col_name_list[i + 1]  # Skip Label column
        print(f"{feature_name:<30} {importance:<15.6f}")
        # Format: (jobid, param, value, dt)
        values.append(f"('{job_id}', '{feature_name}', {importance}, '{dt_partition}')")

    # Execute INSERT
    insert_sql += ", ".join(values)

    print(f"\n=== Executing SQL ===")
    spark.sql(insert_sql)

    print(f"\n=== Results saved to iceberg.ias.tttm_rf (dt={dt_partition}) ===")
    print(f"Total features: {len(feature_importances)}")

    spark.stop()


if __name__ == "__main__":
    main()
