#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTTM PCA (Principal Component Analysis)
PySpark version with MinIO and Iceberg support for Kubernetes
"""

import os
import sys
import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.sql.functions import col, lit
from pyspark.ml.functions import vector_to_array


def main():
    # MinIO Configuration - use environment variable for Docker/Kubernetes
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9011")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

    # Configure Spark with Iceberg and MinIO
    spark = (SparkSession
            .builder
            .appName("TTTM_PCA")
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
        name = sys.argv[i].split(":")[0]
        value = sys.argv[i].split(":")[1]
        if name == "jobid":
            job_id = value
        elif name == "area":
            cluster_area = value

    # Set defaults if not provided
    if not cluster_area:
        cluster_area = "local"

    # Get current date for dt partition (format: YYYYMMDD) - same as regression.py
    dt_partition = datetime.datetime.now().strftime("%Y%m%d")

    print(f"=== TTTM PCA Analysis (K8s) ===")
    print(f"Job ID: {job_id}")
    print(f"Area: {cluster_area}")
    print(f"Partition (dt): {dt_partition}")

    # Read input file - MinIO logic for local/ich/wxh
    if cluster_area == "local":
        local_path = f"/root/{job_id}"
        if os.path.exists(local_path):
            print(f"[INFO] Reading from local file: {local_path}")
            Total = pd.read_csv(local_path)
            print(f"[INFO] Successfully read from local file")
            # Convert pandas DataFrame to Spark DataFrame
            df = spark.createDataFrame(Total)
        else:
            print(f"[INFO] Local file not found, reading from MinIO/S3")
            # Read from MinIO/S3 using Spark
            input_path = f"s3a://ic-ias/structured/{job_id}"
            try:
                df = spark.read.csv(input_path, header=False, inferSchema=True)
                print(f"[INFO] Successfully read from MinIO: {input_path}")
            except Exception as e:
                print(f"[ERROR] Failed to read from MinIO: {e}")
                sys.exit(1)
    else:
        # For remote clusters (ich, wxh), read from MinIO/S3
        input_path = f"s3a://ic-ias/structured/{job_id}"
        try:
            df = spark.read.csv(input_path, header=False, inferSchema=True)
            if cluster_area == "ich":
                print(f"[INFO] Successfully read from MinIO (ich): {input_path}")
            else:  # wxh
                print(f"[INFO] Successfully read from MinIO (wxh): {input_path}")
        except Exception as e:
            print(f"[ERROR] Failed to read from MinIO: {e}")
            sys.exit(1)

    print(f"Input path: {input_path if cluster_area != 'local' or not os.path.exists(local_path) else local_path}")

    # Handle null values
    df = df.na.drop("any")

    print(f"\n=== Input Data ===")
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    # Get column names (exclude first column which is the ID column)
    # If header=True, the first column name is the actual column name, not _c0
    first_col = df.columns[0]
    col_names = [c for c in df.columns if c != first_col]
    print(f"Feature columns: {len(col_names)}")

    # Assemble features
    assembler = VectorAssembler(
        inputCols=col_names,
        outputCol="features",
        handleInvalid="skip"
    )

    # Transform to create features column
    assembled_df = assembler.transform(df)

    # Standardize features
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )

    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)

    # Apply PCA with K=2
    pca = PCA(
        inputCol="scaledFeatures",
        outputCol="pcaFeatures",
        k=2
    )

    pca_model = pca.fit(scaled_df)
    pca_result = pca_model.transform(scaled_df)

    # Print explained variance
    explained_variance = pca_model.explainedVariance.toArray()
    print(f"\n=== PCA Results ===")
    print(f"Explained Variance Ratio:")
    print(f"  PC1: {explained_variance[0]:.4f} ({explained_variance[0]*100:.2f}%)")
    print(f"  PC2: {explained_variance[1]:.4f} ({explained_variance[1]*100:.2f}%)")
    print(f"  Total: {explained_variance.sum():.4f} ({explained_variance.sum()*100:.2f}%)")

    # Extract PCA features and convert to separate columns
    # Convert Vector to Array using Spark SQL built-in function
    result_df = pca_result.select(
        col(first_col).alias("id"),
        vector_to_array("pcaFeatures")[0].alias("pc1"),
        vector_to_array("pcaFeatures")[1].alias("pc2")
    )

    # Add jobid and dt columns for Iceberg table (partitioned by dt)
    result_df = result_df.withColumn("jobid", lit(job_id))
    result_df = result_df.withColumn("dt", lit(dt_partition))

    # Show sample results
    print(f"\n=== Sample Results (first 10 rows) ===")
    result_df.show(10, truncate=False)

    # Save results to Iceberg table
    print(f"\n=== Saving Results to Iceberg ===")

    # Write to Iceberg table (matches CREATE TABLE schema: id, pc1, pc2, jobid, dt)
    result_df.write \
        .format("iceberg") \
        .mode("append") \
        .save("iceberg.ias.tttm_pca")

    print(f"Results saved to iceberg.ias.tttm_pca (dt={dt_partition})")
    print(f"Total records: {result_df.count()}")

    spark.stop()


if __name__ == "__main__":
    main()
