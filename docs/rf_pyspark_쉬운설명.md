# 🎯 Random Forest Feature Importance - 쉬운 설명

## 📖 이 알고리즘이 뭐하는 거야?

한마디로: **반도체 공정 데이터에서 "진짜 중요한 변수"를 찾아내는 똑똑한 AI!**

```
🏭 반도체 공정: 온도, 압력, 가스... 수백 개의 설정값
    ↓
🤖 Random Forest가 100명의 전문가처럼 투표
    ↓
💡 "이 5가지만 조절하면 수율이 올라요!"
```

---

## 🌲 Random Forest가 뭐야?

**비유:** 100명의 식물학자가 각자 의견을 내서 democracy로 결정!

```python
🌳🌳🌳 100명의 전문가 (Decision Tree)

전문가 1: "온도가 제일 중요해!"
전문가 2: "아니야, 압력이 더 중요!"
전문가 3: "가스 유량이 핵심이지!"
...
전문가 100: "역시 온도!"

📊 투표 결과:
1위: 온도 (25명 투표) → 중요도 0.25
2위: 압력 (18명 투표) → 중요도 0.18
3위: 가스 유량 (12명 투표) → 중요도 0.12
...
```

---

## 🎮 전체 처리 과정

### 1단계: 데이터 읽기 📖

```python
# MinIO에서 데이터 읽기 (로컬 또는 원격 클러스터)
if cluster_area == "local":
    # 내 컴퓨터에 있으면 그냥 읽기
    local_path = f"/root/{job_id}"
else:
    # MinIO (S3)에서 읽기
    input_path = f"s3a://ic-ias/structured/{job_id}"

# CSV 형식으로 읽기
df = spark.read.csv(input_path, header=True, inferSchema=True)
```

**예시 데이터:**
```
Label, Temperature, Pressure, Gas_Flow, Time
0.85, 1000, 50, 30, 120
0.92, 1050, 55, 32, 125
0.78, 980, 48, 28, 118
...
```

### 2단계: 데이터 전처리 🔧

```python
# 모든 컬럼을 숫자(Double)로 변환
for col in col_name_list:
    df = df.withColumn(col, df[col].cast(DoubleType()))

# Label은 첫 번째 컬럼
# Features는 나머지 컬럼
Label = col_name_list[0]      # 수율 또는 품질
Features = col_name_list[1:]  # 온도, 압력, 가스, 시간...
```

### 3단계: Feature Vector 만들기 📦

```python
# 여러 컬럼을 하나의 "features" 벡터로 합치기
assembler = VectorAssembler(
    inputCols=feature_cols,  # [Temperature, Pressure, Gas_Flow, Time]
    outputCol="features"
)

# 결과:
# features = [1000.0, 50.0, 30.0, 120.0]
```

### 4단계: Label 인덱싱 🏷️

```python
# Label (수율)을 숫자 인덱스로 변환
# 예: "불량" → 0, "양품" → 1
label_indexer = StringIndexer(
    inputCol=Label,
    outputCol="indexedLabel"
)
```

### 5단계: Feature 인덱싱 📊

```python
# Feature도 인덱스로 변환 (카테고리 처리)
feature_indexer = VectorIndexer(
    inputCol="features",
    outputCol="indexedFeatures",
    maxCategories=4  # 카테고리가 4개 이하면 인덱싱
)
```

### 6단계: Random Forest 훈련 🌲

```python
# Random Forest 모델 생성
rf = RandomForestClassifier(
    labelCol="indexedLabel",
    featuresCol="indexedFeatures",
    maxDepth=30,     # 트리 최대 깊이
    numTrees=500     # 500개의 트리 (전문가)!
)

# 파이프라인으로 한번에 처리
pipeline = Pipeline(stages=[
    assembler,         # Feature 벡터 생성
    label_indexer,     # Label 인덱싱
    feature_indexer,   # Feature 인덱싱
    rf,                # Random Forest 훈련
    label_converter    # 예측 결과를 다시 원래 Label로 변환
])

# 훈련!
model = pipeline.fit(df)
```

**왜 500개의 트리를 써?**
- 500명의 전문가가 투표하면 더 정확해져
- 1명보다 100명, 100명보다 500명이 더 좋아
- 너무 많으면 느려지고, 너무 적으면 부정확해

### 7단계: Feature Importance 추출 🎯

```python
# 훈련된 모델에서 Feature Importance 추출
rf_model = model.stages[3]  # Pipeline의 4번째 단계가 RF
feature_importances = rf_model.featureImportances.toArray()

# 결과 출력:
print(f"{'Feature':<30} {'Importance':<15}")
print("-" * 45)

Temperature              0.245
Pressure                 0.182
Gas_Flow                 0.115
Time                     0.089
...
```

**Feature Importance가 뭐야?**
- 0에 가까울수록: 중요하지 않아
- 1에 가까울수록: 매우 중요해
- 전체 합계: 1.0 (100%)

### 8단계: 결과 저장 💾

```python
# Iceberg 테이블에 INSERT
insert_sql = """
INSERT INTO iceberg.ias.tttm_rf VALUES
('job_123', 'Temperature', 0.245, '20250610'),
('job_123', 'Pressure', 0.182, '20250610'),
('job_123', 'Gas_Flow', 0.115, '20250610'),
...
"""

spark.sql(insert_sql)
```

**테이블 구조:**
```
CREATE TABLE iceberg.ias.tttm_rf (
    jobid STRING,      # 작업 ID
    param STRING,       # 파라미터 이름 (Temperature, Pressure...)
    value DOUBLE,       # Feature Importance 값
    dt STRING           # 파티션 (YYYYMMDD)
)
```

---

## 🛠️ 중요 파라미터

### Random Forest 설정

```python
maxDepth=30        # 트리 최대 깊이
                  # - 너무 작으면: 학습 부족 (과소적합)
                  # - 너무 크면: 과도하게 학습 (과적합)
                  # - 30은 깊은 편 (복잡한 패턴 학습)

numTrees=500       # 트리 개수
                  # - 10: 너무 적음 (불안정)
                  # - 100: 적당
                  # - 500: 매우 안정적
                  # - 1000: 너무 많음 (느려짐)
```

---

## 🚀 병렬 처리 - 빠르다!

### Spark의 병렬화

```
🐌 단일 머신:
500개 트리 × 10분 = 5,000분 (83시간!)

🚀 Spark 클러스터 (8노드):
500개 트리 / 8노드 = 63개/노드 × 10분 = 630분 (10.5시간)
```

**PySpark의 자동 병렬화:**
- 데이터를 여러 파티션으로 분할
- 각 노드에서 독립적으로 트리 훈련
- 결과를 Driver로 모아서 집계

---

## 🎭 왜 Random Forest를 써?

### 장점

```
✅ 안정적
   - 100명의 전문가 투표라 1명 실수해도 괜찮아
   - 노이즈 데이터에 강해

✅ 비선형 관계 학습
   - 온도가 너무 낮아도 안 좋고, 너무 높아도 안 좋아
   - 이 "최적점"을 찾을 수 있어

✅ Feature Importance 제공
   - 어느 변수가 제일 중요한지 바로 알 수 있어
   - 현장 엔지니어가 이해하기 쉬워

✅ 데이터 정규화 불필요
   - 각 트리가 독립적이라 스케일 안 중요해
```

### 단점

```
❌ 해석 어려움
   - 500개 트리를 다 볼 수는 없어
   - "왜 중요한지"는 안 알려줘 (중요도만 알려줌)

❌ 메모리 많이 사용
   - 500개 트리를 모두 저장해야 해
   - 클러스터 필수

❌ 훈련 느림
   - 단일 머신에서는 시간 오래 걸려
```

---

## 🏭 실전 예시: 반도체 공장

### 상황

```
🏭 반도체 공장
📦 제품: 28nm 로직 칩
📊 데이터: 10,000 로트, 500개 파라미터
🎯 문제: 수율이 78.5%... (목표 85%+)
```

### Random Forest 적용

```python
입력: 10,000 로트 × 500 파라미터

🤖 500개의 트리가 훈련

트리 1:
- 온도가 1000°C 이상이면 수율 좋음 (중요도 0.3)
- 압력이 50mTorr 이상이면 수율 좋음 (중요도 0.2)
...

트리 2:
- 가스 유량이 30sccm이면 수율 좋음 (중요도 0.25)
- 시간이 120초면 수율 좋음 (중요도 0.15)
...

...

📊 500개 트리의 평균 중요도:
1. Annealing_Temp (어닐링 온도): 0.245
2. Etch_Pressure (에칭 압력): 0.182
3. Gas_Flow_AR (가스 유량): 0.115
4. Photo_Time (노광 시간): 0.089
5. Plasma_Power (플라즈마 파워): 0.065
...
```

### 결과 해석

```
🎯 중요도 상위 5개 파라미터:

1. Annealing_Temp (24.5%)
   → 어닐링 온도가 수율의 24.5%를 설명해
   → 온도 제어가 가장 중요해!

2. Etch_Pressure (18.2%)
   → 에칭 압력이 18.2% 설명
   → 압력도 중요하지만 온도보다는 덜 중요해

3. Gas_Flow_AR (11.5%)
   → 가스 유량이 11.5% 설명
   → 상위 3개가 전체의 54.2%를 차지해!

💡 통찰:
"이 3가지만 제대로 제어하면 수율의 절반 이상을 설명할 수 있어!"
```

---

## 🔍 Lasso, Step Forward와의 비교

### Random Forest vs Lasso

```
🌲 Random Forest:
- 비선형 관계 학습 가능
- "어느 정도" 중요한지 알려줌
- 순위 매기기에 좋아

🤺 Lasso:
- 선형 관계만 학습
- "얼만큼" 중요한지 정확히 알려줌
- 정밀한 숫자를 얻기 좋아
```

### 조합해서 사용

```
1단계: Random Forest → 500개 → 100개 (대략적 순위)
2단계: Lasso → 100개 → 20개 (정밀한 계수)
3단계: Step Forward → 20개 → 5개 (최적 조합)

각 단계의 장점만 취해서 최고의 성능!
```

---

## 💡 한 줄 요약

**Random Forest는 "100명의 전문가가 투표해서 중요한 변수 찾는 AI"!** 🌲🗳️

```
🤖 500개의 Decision Tree가 투표
📊 각 변수의 중요도를 0~1로 점수화
🎯 "이게 제일 중요해!" 바로 알려줌
💾 MinIO → Iceberg에 결과 저장
```

**핵심:**
- 복잡한 관계도 학습 가능
- 500개 트리로 안정적 예측
- Feature Importance로 쉽게 해석
- PySpark로 빠르게 병렬 처리
