# Random Forest PySpark 상세 분석 보고서

## 개요

본 문서는 `/root/rf_pyspark.py`에 구현된 PySpark 기반 Random Forest Feature Importance 분석 알고리즘을 상세히 분석한 것이다. 이 알고리즘은 반도체 제조 공정 데이터에서 수율에 영향을 미치는 주요 파라미터를 식별하고, 각 파라미터의 중요도를 수치화하여 MinIO/Iceberg 아키텍처에 저장한다.

---

## 1. 알고리즘 처리 과정 및 데이터 흐름

### 1.1 전체 파이프라인

```
[입력 데이터 로드] → [데이터 전처리] → [Feature Vector 생성] →
[Label 인덱싱] → [Feature 인덱싱] → [Random Forest 훈련] →
[Feature Importance 추출] → [Iceberg 저장]
```

### 1.2 단계별 상세 처리 과정

### **단계 1: Spark Session 설정 (라인 18-41)**

**처리 과정:**
- Iceberg 카탈로그 설정 (Hadoop 기반)
- MinIO/S3A 연결 설정
- 확장 기능 로드

**코드:**
```python
spark = SparkSession.builder \
    .appName("TTTM_RandomForest") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.type", "hadoop") \
    .config("spark.sql.catalog.iceberg.warehouse", "s3a://ic-ias/structured") \
    .config("spark.sql.catalog.iceberg.io-impl", "org.apache.iceberg.hadoop.HadoopFileIO") \
    # S3A/MinIO 설정
    .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
    .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .getOrCreate()
```

**중요 파라미터:**
- `iceberg.warehouse`: Iceberg 테이블 저장소 (MinIO S3)
- `fs.s3a.endpoint`: MinIO 서버 주소
- `fs.s3a.path.style.access`: Path-style access (MinIO 호환)

### **단계 2: 인자 파싱 (라인 43-67)**

**처리 과정:**
- 명령행 인자 파싱 (`jobid:xxx area:xxx`)
- 기본값 설정 (area="local")
- 파티션 키 생성 (dt=YYYYMMDD)

**코드:**
```python
for i in range(1, len(sys.argv)):
    parts = sys.argv[i].split(":")
    if len(parts) == 2:
        name = parts[0]
        value = parts[1]
        if name == "jobid":
            job_id = value
        elif name == "area":
            cluster_area = value

if not cluster_area:
    cluster_area = "local"

dt_partition = datetime.datetime.now().strftime("%Y%m%d")
```

**사용 예시:**
```bash
spark-submit rf_pyspark.py jobid:job_123 area:ich
```

### **단계 3: 입력 데이터 로드 (라인 69-89)**

**처리 과정:**
- `cluster_area`에 따라 로컬/MinIO 선택
- CSV 형식으로 로드 (헤더 포함, 스키마 추론)

**코드:**
```python
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

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ",") \
    .csv(input_path)
```

**데이터 예시:**
```
Label,Temperature,Pressure,Gas_Flow,Time
0.85,1000.0,50.0,30.0,120.0
0.92,1050.0,55.0,32.0,125.0
0.78,980.0,48.0,28.0,118.0
```

### **단계 4: 데이터 전처리 (라인 91-106)**

**처리 과정:**
- 스키마 출력
- 데이터 샘플 출력
- 컬럼 리스트 추출
- DoubleType으로 변환

**코드:**
```python
df.printSchema()
df.show(5, truncate=False)

col_name_list = df.columns
print(f"Label column: {col_name_list[0]}")
print(f"Feature columns: {col_name_list[1:]}")

# Convert all columns to DoubleType
for col in col_name_list:
    df = df.withColumn(col, df[col].cast(DoubleType()))
```

### **단계 5: Feature Vector 생성 (라인 107-112)**

**처리 과정:**
- VectorAssembler로 여러 컬럼을 하나의 벡터로 결합

**코드:**
```python
feature_cols = col_name_list[1:]  # All columns except Label
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)
```

**VectorAssembler의 역할:**
```
입력:
- Temperature: 1000.0
- Pressure: 50.0
- Gas_Flow: 30.0
- Time: 120.0

출력 (features):
- [1000.0, 50.0, 30.0, 120.0]
```

### **단계 6: Label 인덱싱 (라인 114-118)**

**처리 과정:**
- StringIndexer로 Label (카테고리)를 숫자 인덱스로 변환

**코드:**
```python
label_indexer = StringIndexer(
    inputCol=col_name_list[0],  # Use first column as label
    outputCol="indexedLabel"
)
```

**StringIndexer 예시:**
```
입력 (Label):
- "불량"
- "양품"
- "양품"
- "불량"

출력 (indexedLabel):
- 0.0
- 1.0
- 1.0
- 0.0
```

### **단계 7: Feature 인덱싱 (라인 120-125)**

**처리 과정:**
- VectorIndexer로 카테고리 Feature를 자동 인덱싱

**코드:**
```python
feature_indexer = VectorIndexer(
    inputCol="features",
    outputCol="indexedFeatures",
    maxCategories=4  # 카테고리가 4개 이하면 인덱싱
)
```

**VectorIndexer의 역할:**
- 숫자로 된 Feature라도 실제로 카테고리이면 인덱싱
- `maxCategories=4`: 중복값이 4개 이하면 카테고리로 간주

### **단계 8: Random Forest 모델 생성 (라인 127-133)**

**처리 과정:**
- RandomForestClassifier로 분석 모델 생성

**코드:**
```python
rf = RandomForestClassifier(
    labelCol="indexedLabel",
    featuresCol="indexedFeatures",
    maxDepth=30,
    numTrees=500
)
```

**중요 파라미터:**

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `maxDepth` | 30 | 트리 최대 깊이 (깊을수록 복잡한 패턴 학습) |
| `numTrees` | 500 | 생성할 트리 개수 (많을수록 안정적) |
| `labelCol` | "indexedLabel" | 타겟 변수 (인덱싱된 Label) |
| `featuresCol` | "indexedFeatures" | 입력 변수 (인덱싱된 Feature) |

### **단계 9: Prediction 변환기 (라인 135-140)**

**처리 과정:**
- IndexToString으로 예측 결과를 다시 원래 Label로 변환

**코드:**
```python
label_converter = IndexToString(
    inputCol="prediction",
    outputCol="predictedLabel",
    labels=label_indexer.fit(df).labels
)
```

**변환 예시:**
```
입력 (prediction):
- 0.0
- 1.0

출력 (predictedLabel):
- "불량"
- "양품"
```

### **단계 10: Pipeline 구성 및 훈련 (라인 142-148)**

**처리 과정:**
- Pipeline으로 모든 단계를 순차적으로 연결
- `.fit()`으로 훈련

**코드:**
```python
pipeline = Pipeline(stages=[
    assembler,         # 1. Feature Vector 생성
    label_indexer,     # 2. Label 인덱싱
    feature_indexer,   # 3. Feature 인덱싱
    rf,                # 4. Random Forest 훈련
    label_converter    # 5. 예측 결과 변환
])

model = pipeline.fit(df)
```

**Pipeline의 장점:**
- 모든 전처리와 훈련을 하나로 묶을 수 있음
- `.fit()` 한 번으로 모든 단계 자동 실행
- 훈련된 모델을 `.transform()`으로 바로 사용 가능

### **단계 11: Feature Importance 추출 (라인 150-156)**

**처리 과정:**
- 훈련된 모델에서 Random Forest 모델 추출
- Feature Importance 배열 추출

**코드:**
```python
rf_model = model.stages[3]  # Pipeline의 4번째 단계가 RF
feature_importances = rf_model.featureImportances.toArray()
```

**Pipeline stages 구조:**
```
stages[0] = assembler (VectorAssemblerModel)
stages[1] = label_indexer (StringIndexerModel)
stages[2] = feature_indexer (VectorIndexerModel)
stages[3] = rf (RandomForestClassificationModel)  ← 여기!
stages[4] = label_converter (IndexToStringModel)
```

### **단계 12: 결과 출력 및 SQL 생성 (라인 158-172)**

**처리 과정:**
- Feature Importance 출력
- Iceberg INSERT 문 생성

**코드:**
```python
insert_sql = "INSERT INTO iceberg.ias.tttm_rf VALUES "
values = []

for i, importance in enumerate(feature_importances):
    feature_name = col_name_list[i + 1]  # Skip Label column
    print(f"{feature_name:<30} {importance:<15.6f}")
    values.append(f"('{job_id}', '{feature_name}', {importance}, '{dt_partition}')")

insert_sql += ", ".join(values)
spark.sql(insert_sql)
```

**출력 예시:**
```
=== Feature Importances ===
Feature                        Importance
---------------------------------------------
Temperature                    0.245000
Pressure                       0.182000
Gas_Flow                       0.115000
Time                           0.089000
Power                          0.065000
...
```

**생성된 SQL:**
```sql
INSERT INTO iceberg.ias.tttm_rf VALUES
('job_123', 'Temperature', 0.245, '20250610'),
('job_123', 'Pressure', 0.182, '20250610'),
('job_123', 'Gas_Flow', 0.115, '20250610'),
...
```

---

## 2. Random Forest 알고리즘 상세 분석

### 2.1 Random Forest의 원리

**알고리즘 구조:**
```
1. Bootstrap Sampling
   - 원본 데이터에서 중복을 허용하여 샘플링
   - 1000개 데이터 → 1000개 샘플 (중복 포함)

2. Random Feature Selection
   - 각 분할에서 무작위로 일부 Feature만 고려
   - 전체 500개 Feature → 분할 시 50개만 랜덤 선택

3. Decision Tree 훈련
   - Bootstrap 샘플로 트리 훈련
   - 깊이 30까지 분할 반복

4. 500개 트리 생성
   - 1~3 단계를 500번 반복
   - 각 트리는 서로 다름 (랜덤성)

5. Aggregation (투표)
   - 500개 트리의 예측을 집계
   - Feature Importance는 모든 트리의 평균
```

### 2.2 Feature Importance 계산 방법

**Gini Importance (Mean Decrease in Impurity):**

```
1. 각 트리에서:
   - Node의 불순도 감소량 계산
   - Feature가 사용된 횟수 계산
   - (불순도 감소량 × 사용 횟수)를 해당 Feature의 중요도로

2. 모든 트리에서:
   - 각 트리의 Feature Importance를 평균
   - 전체 합이 1이 되도록 정규화
```

**수식:**
```
Importance(f) = Σ (노드에서의 불순도 감소량) / (모든 트리)

정규화:
Importance_norm(f) = Importance(f) / Σ Importance(all features)
```

### 2.3 중요 파라미터 및 튜너링

### maxDepth (트리 최대 깊이)

```python
maxDepth=30
```

**영향:**
- 너무 작으면 (5~10): 모델이 너무 단순 (과소적합)
- 너무 크면 (50~100): 과도하게 학습 (과적합)
- 30은 중간 정도로 복잡한 패턴 학습 가능

**튜닝 가이드:**
```
데이터 크기: 10,000 로트
Feature 수: 500개
권장 maxDepth: 20~30

데이터 크기: 100,000 로트
Feature 수: 100개
권장 maxDepth: 15~20
```

### numTrees (트리 개수)

```python
numTrees=500
```

**영향:**
- 10개: 너무 적음 (불안정)
- 100개: 적당
- 500개: 매우 안정적 (이 코드에서 사용)
- 1000개: 더 안정적 but 느림

**튜닝 가이드:**
```
데이터 크기: 1,000 로트
권장 numTrees: 50~100

데이터 크기: 10,000 로트
권장 numTrees: 100~500

데이터 크기: 100,000 로트
권장 numTrees: 200~1000
```

### 기타 파라미터

```python
# 암시적 기본값들
featureSubsetStrategy = "sqrt"  # 각 분할에서 √(전체 Feature) 개수만 고려
                                 # 500개 Feature → 22개 랜덤 선택

minInstancesPerNode = 1          # 노드의 최소 샘플 수
minInfoGain = 0.0                # 최소 정보 이득

bootstrap = True                 # Bootstrap 샘플링 사용
```

---

## 3. PySpark와 MinIO/Iceberg 아키텍처

### 3.1 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────┐
│                     MinIO (S3)                          │
│  s3a://ic-ias/structured/{job_id}  (입력 CSV)          │
│  s3a://ic-ias/structured/warehouse/ (Iceberg 테이블)   │
└─────────────────────────────────────────────────────────┘
                            ↑ ↓
┌─────────────────────────────────────────────────────────┐
│                    PySpark Driver                        │
│                                                          │
│  1. SparkSession 설정 (Iceberg + S3A)                   │
│  2. CSV 로드 (spark.read.csv)                           │
│  3. Pipeline 훈련 (pipeline.fit)                        │
│  4. Feature Importance 추출                             │
│  5. Iceberg INSERT (spark.sql)                         │
└─────────────────────────────────────────────────────────┘
                            ↑ ↓
┌─────────────────────────────────────────────────────────┐
│                   Spark Executors                       │
│                                                          │
│  - 데이터 파티션 처리                                    │
│  - Decision Tree 훈련 (병렬)                            │
│  - Feature Importance 계산                              │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Iceberg 테이블 구조

**테이블 DDL (추정):**
```sql
CREATE TABLE iceberg.ias.tttm_rf (
    jobid STRING,      -- 작업 ID
    param STRING,       -- 파라미터 이름
    value DOUBLE,       -- Feature Importance 값
    dt STRING           -- 파티션 키 (YYYYMMDD)
) PARTITIONED BY (dt)
```

**INSERT 예시:**
```sql
INSERT INTO iceberg.ias.tttm_rf VALUES
('job_123', 'Temperature', 0.245, '20250610'),
('job_123', 'Pressure', 0.182, '20250610'),
('job_123', 'Gas_Flow', 0.115, '20250610')
```

### 3.3 S3A Configuration 상세

```python
# MinIO 연결 설정
.config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
# 예: "http://minio.default.svc.cluster.local:9000"

.config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
# 예: "minioadmin"

.config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
# 예: "minioadmin"

.config("spark.hadoop.fs.s3a.path.style.access", "true")
# Path-style access: http://minio/bucket/key
# Virtual-hosted style: http://bucket.minio/key
# MinIO는 path-style 필요

.config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
# SSL 비활성화 (개발 환경)
```

---

## 4. 클러스터 영역별 처리 로직

### 4.1 Local 모드

```python
if cluster_area == "local":
    local_path = f"/root/{job_id}"
    if os.path.exists(local_path):
        # 로컬 파일 시스템에서 읽기
        input_path = local_path
    else:
        # 로컬에 없으면 MinIO에서 읽기
        input_path = f"s3a://ic-ias/structured/{job_id}"
```

### 4.2 Remote 모드 (ich, wxh)

```python
else:
    # 원격 클러스터는 무조건 MinIO에서 읽기
    input_path = f"s3a://ic-ias/structured/{job_id}"
```

**사용 예시:**
```bash
# Local (개발 환경)
spark-submit rf_pyspark.py jobid:job_123 area:local

# ICH 클러스터 (운영 환경)
spark-submit rf_pyspark.py jobid:job_123 area:ich

# WXH 클러스터 (운영 환경)
spark-submit rf_pyspark.py jobid:job_123 area:wxh
```

---

## 5. 성능 최적화

### 5.1 PySpark 병렬 처리

**자동 병렬화:**
```
DataFrame → 파티션 분할 (기본 200개)
    ↓
각 파티션을 Executor에 할당
    ↓
각 Executor에서 독립적으로 Random Forest 훈련
    ↓
Driver로 결과 수집 및 집계
```

**Executor 설정 예시:**
```bash
spark-submit \
  --num-executors 10 \
  --executor-cores 4 \
  --executor-memory 16G \
  rf_pyspark.py jobid:job_123 area:ich
```

### 5.2 메모리 최적화

```python
# DataFrame 지연 연산 (Lazy Evaluation)
df = spark.read.csv(...)  # 즉시 로드 안 함
df = df.withColumn(...)    # 변환만 정의
df.count()                  # 이때 실제 실행

# 캐싱으로 반복 계산 방지
df.cache()  # 메모리에 캐시
```

### 5.3 파티션 최적화

```python
# 파티션 수 조정
df = spark.read.csv(...).repartition(100)

# Coalesce로 파티션 줄이기 (Shuffle 없음)
df = df.coalesce(10)
```

---

## 6. 실전 분석 예시

### 6.1 입력 데이터

```csv
Label,Temperature,Pressure,Gas_Flow,Time,Power
0.85,1000.0,50.0,30.0,120.0,200.0
0.92,1050.0,55.0,32.0,125.0,210.0
0.78,980.0,48.0,28.0,118.0,195.0
0.88,1020.0,52.0,31.0,122.0,205.0
0.91,1045.0,54.0,31.5,124.0,208.0
...
(10,000 로트)
```

### 6.2 Feature Importance 결과

```
=== Feature Importances ===
Feature                        Importance
---------------------------------------------
Annealing_Temp.Step1           0.245000
Etch_Pressure.Step5            0.182000
Gas_Flow_AR.Step2              0.115000
Photo_Time.Step3               0.089000
Plasma_Power.Step4             0.065000
Clean_Time.Step6               0.042000
Bake_Temp.Step7                0.038000
Rinse_Pressure.Step8           0.025000
Dry_Time.Step9                 0.018000
Inspect_Speed.Step10           0.012000
...
```

### 6.3 결과 해석

```
상위 3개 파라미터 (54.2% 설명):
1. Annealing_Temp (24.5%)
   → 어닐링 온도가 수율의 24.5%를 설명
   → 가장 중요한 공정 파라미터

2. Etch_Pressure (18.2%)
   → 에칭 압력이 18.2% 설명
   → 2번째로 중요

3. Gas_Flow_AR (11.5%)
   → 가스 유량이 11.5% 설명
   → 상위 3개가 전체의 절반 이상 차지

통찰:
"이 3가지만 제대로 제어하면 수율의 54.2%를 설명할 수 있어!"
```

---

## 7. 제한사항 및 개선점

### 7.1 제한사항

```
❌ 해석 가능성
   - Feature Importance는 "얼마나 중요한지"만 알려줌
   - "왜 중요한지"는 설명 안 해
   - 상관관계 vs 인과관계 구분 안 됨

❌ 상관된 Feature
   - 서로 상관된 Feature들은 중요도를 분산
   - Feature A와 B가 0.8 상관이면:
     - 각각 0.3, 0.3으로 중요도 낮게 나올 수 있음
     - 실제는 합쳐서 0.6 정도여야 할 수도

❌ 비선형 관계
   - Random Forest는 비선형 관계 학습 가능
   - but Feature Importance는 단일 숫자라 복잡한 관계 표현 안 됨

❌ 스케일 민감
   - Feature 스케일이 다르면 편향 발생 가능
   - 코드에서는 DoubleType으로만 변환 (표준화 안 함)
```

### 7.2 개선 제안

```
1. 표준화 추가
   from pyspark.ml.feature import StandardScaler

   scaler = StandardScaler(
       inputCol="features",
       outputCol="scaledFeatures"
   )
   # RF는 표준화 안 해도 되지만, 일관성 위해 권장

2. Permutation Importance
   - Feature Importance의 안정성 강화
   - sklearn.inspection.permutation_importance 활용

3. SHAP Values
   - "왜 중요한지" 해석 가능
   - 각 예측에 대한 Feature 기여도 분석

4. 교차 검증
   from pyspark.ml.evaluation import BinaryClassificationEvaluator

   train_df, test_df = df.randomSplit([0.7, 0.3])
   # 훈련/테스트 분리로 과적합 방지
```

---

## 8. 결론

본 Random Forest PySpark 알고리즘은 반도체 제조 공정 데이터에서 Feature Importance를 효율적으로 추출한다. PySpark의 병렬 처리 능력과 MinIO/Iceberg의 저장소 아키텍처를 활용하여 대용량 데이터를 빠르게 분석하고 결과를 저장한다.

**주요 특징:**
- 500개 트리로 안정적인 Feature Importance 계산
- MinIO/S3와 Iceberg로 분산 저장소 활용
- 클러스터 영역별로 로컬/원격 데이터 소스 선택
- Pipeline으로 전처리와 훈련을 자동화

**활용:**
- 공정 파라미터 우선순위 선정
- 불필요한 파라미터 필터링
- Lasso, Step Forward의 입력으로 활용
