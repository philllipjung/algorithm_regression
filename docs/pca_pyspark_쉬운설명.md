# 🎯 PCA (Principal Component Analysis) - 쉬운 설명

## 📖 이 알고리즘이 뭐하는 거야?

한마디로: **수백 개의 복잡한 데이터를 2개의 그림으로 줄여서 "패턴"을 보여주는 압축 AI!**

```
📊 500개의 파라미터 (온도, 압력, 가스...)
    ↓
🤖 PCA가 "진짜 중요한 방향" 2개를 찾아
    ↓
📈 2차원 그래프로 데이터 군집을 한눈에 볼 수 있어!
```

---

## 🎮 PCA가 뭐야? (비유)

### 사진 압축과 비슷해!

```
📷 원본 사진:
- 1000만 픽셀 (각각 RGB 3색)
- 파일 크기: 50MB

🗜️ 압축:
- "진짜 중요한 특징"만 추출
- 눈, 코, 입 위치
- 밝은 곳, 어두운 곳

💾 압충 후:
- 100만 픽셀만 저장
- 파일 크기: 5MB
- 여전히 사람은 알아볼 수 있어!

📖 PCA도 똑같아:
- 500개 파라미터 → 2개 주성분 (Principal Component)
- 여전히 데이터의 "본질"은 보존
```

---

## 📊 PCA의 원리

### 1단계: 데이터 표준화 (StandardScaler)

**왜 표준화해?**

```
📏 원본 데이터:
온도: 1000°C ± 50°C (값이 큼)
압력: 50 mTorr ± 5 mTorr (값이 작음)
가스: 30 sccm ± 3 sccm (값이 작음)

문제: 온도가 숫자가 커서 PCA가 "온도가 제일 중요하다!"고 생각해

해결: 모두를 평등하게 만들어
온도: 0 ± 1 (표준화)
압력: 0 ± 1
가스: 0 ± 1

이제 모두 평등! 😊
```

**코드:**
```python
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,   # 평균을 0으로
    withStd=True     # 표준편차를 1로
)

# 결과: 모든 변수가 mean=0, std=1인 상태
```

### 2단계: PCA 변환

**PCA가 하는 일:**

```
📊 원본 데이터 (3D):

    압력
     ↑
     |    *
     |  *   *
     |*      *
     +----------→ 온도
    /
   /
 가스

문제: 3D 그래프는 한눈에 안 들어와

🎯 PCA 변환 (2D):

PC2 ↑
    |  *  *
    | * * *
    |*    *
    +----------→ PC1

이제 보기 쉬워! 2개의 축(PC1, PC2)만 있어
```

**PC1, PC2가 뭐야?**

```
🧐 PCA는 "진짜 중요한 방향"을 찾아

PC1 (제1주성분):
- 데이터가 "가장 많이 퍼져 있는 방향"
- 데이터의 차이를 제일 잘 설명해
- 예: "온도가 높을수록 압력도 높아" → 이 방향이 PC1

PC2 (제2주성분):
- PC1과 "수직인 방향"
- 남은 차이를 설명
- 예: "온도는 같은데 가스 유량만 다른 경우"

PC1 설명력: 65%
PC2 설명력: 25%
총: 90% (나머지 10%는 정보 손실)
```

**코드:**
```python
pca = PCA(
    inputCol="scaledFeatures",
    outputCol="pcaFeatures",
    k=2  # 2개의 주성분만 추출
)

pca_model = pca.fit(scaled_df)
pca_result = pca_model.transform(scaled_df)
```

### 3단계: 결과 해석

```python
# 설명된 분산 (Explained Variance)
explained_variance = pca_model.explainedVariance.toArray()

print(f"PC1: {explained_variance[0]:.4f} ({explained_variance[0]*100:.2f}%)")
print(f"PC2: {explained_variance[1]:.4f} ({explained_variance[1]*100:.2f}%)")
print(f"Total: {explained_variance.sum():.4f} ({explained_variance.sum()*100:.2f}%)")
```

**출력 예시:**
```
=== PCA Results ===
Explained Variance Ratio:
  PC1: 0.6543 (65.43%)
  PC2: 0.2456 (24.56%)
  Total: 0.8999 (89.99%)
```

**해석:**
- PC1 하나로 데이터의 65.43%를 설명해
- PC1 + PC2로 전체의 89.99% 설명
- 10% 손실 but 500개 → 2개로 줄였으니 괜찮아!

---

## 🔍 전체 처리 과정

### 입력 데이터

```python
# 예시: 반도체 공정 데이터
# ID, 온도, 압력, 가스, 시간, 파워, ...
ID, Temperature, Pressure, Gas_Flow, Time, Power, ...
001, 1000, 50, 30, 120, 200, ...
002, 1050, 55, 32, 125, 210, ...
003, 980, 48, 28, 118, 195, ...
...

10,000 로트 × 500 파라미터
```

### 1단계: Null 값 제거

```python
df = df.na.drop("any")

# 결측치가 있는 행은 모두 제거
# 10,000 → 9,823 로트 (177개 제거)
```

### 2단계: Feature 조립

```python
# ID를 제외한 모든 컬럼을 Feature로 사용
col_names = [c for c in df.columns if c != first_col]

assembler = VectorAssembler(
    inputCols=col_names,  # [Temperature, Pressure, Gas_Flow, ...]
    outputCol="features"
)

# 결과: features = [1000.0, 50.0, 30.0, 120.0, 200.0, ...]
```

### 3단계: 표준화

```python
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

# 결과: scaledFeatures = [0.5, -0.3, 0.8, 0.1, -0.2, ...]
# (모두 mean=0, std=1)
```

### 4단계: PCA 적용

```python
pca = PCA(
    inputCol="scaledFeatures",
    outputCol="pcaFeatures",
    k=2
)

pca_model = pca.fit(scaled_df)
pca_result = pca_model.transform(scaled_df)
```

### 5단계: 결과 추출

```python
# Vector를 Array로 변환하고 PC1, PC2 분리
result_df = pca_result.select(
    col(first_col).alias("id"),
    vector_to_array("pcaFeatures")[0].alias("pc1"),
    vector_to_array("pcaFeatures")[1].alias("pc2")
)

# 결과:
# id, pc1, pc2
# 001, 2.5, -1.3
# 002, 3.1, -0.8
# 003, 1.9, -1.9
```

### 6단계: Iceberg에 저장

```python
# jobid와 dt 컬럼 추가
result_df = result_df.withColumn("jobid", lit(job_id))
result_df = result_df.withColumn("dt", lit(dt_partition))

# Iceberg 테이블에 저장
result_df.write \
    .format("iceberg") \
    .mode("append") \
    .save("iceberg.ias.tttm_pca")
```

**테이블 구조:**
```
CREATE TABLE iceberg.ias.tttm_pca (
    id STRING,        # 로트 ID
    pc1 DOUBLE,       # 제1주성분 값
    pc2 DOUBLE,       # 제2주성분 값
    jobid STRING,     # 작업 ID
    dt STRING         # 파티션 (YYYYMMDD)
)
```

---

## 📈 PCA 결과의 활용

### 1. 시각화 및 군집 발견

```
📊 2차원 그래프 (PC1 vs PC2):

PC2 ↑
    |  🔴🔴🔴
    | 🔴🔴🔴
    |🔴🔴🔴
    |
    |         🔵🔵
    |        🔵🔵🔵
    |       🔵🔵🔵
    |
    |                🟢🟢
    |               🟢🟢🟢
    |
    +----------------------→ PC1

🔴 그룹: 온도 높음, 압력 높음 (수율 92%)
🔵 그룹: 온도 중간, 압력 중간 (수율 85%)
🟢 그룹: 온도 낮음, 압력 낮음 (수율 78%)

💡 통찰:
"온도와 압력이 함께 높을 때 수율이 최고야!"
```

### 2. 이상치 탐지

```
📊 정상 데이터:

PC2 ↑
    |   ⚫⚫⚫
    |  ⚫⚫⚫⚫
    | ⚫⚫⚫⚫⚫
    +----------→ PC1

📊 이상치 발견:

PC2 ↑
    |   ⚫⚫⚫
    |  ⚫⚫⚫⚫
    | ⚫⚫⚫⚫⚫
    |          🔴   ← 이상치!
    +----------→ PC1

🔴 "이 로트는 데이터가 이상해!"
→ 원인: 장비 오류, 측정 오류, 공정 이상
```

### 3. 차원 축소

```
원본: 500개 파라미터
→ 머신러닝에 넣기: 느리고 복잡해

PCA: 2개 주성분
→ 머신러닝에 넣기: 빠르고 간단해
→ 500개 중 90% 정보 보존

활용:
- K-Means 군집화 (2D라 빨라)
- 이상치 탐지 (거리 계산 쉬워)
- 시각화 (그래프로 바로 그려)
```

---

## 🎯 PCA의 장단점

### 장점 ✅

```
✅ 차원 축소
   - 500개 → 2개 (250배 축소!)
   - 계산 빨라짐
   - 메모리 적게 먹음

✅ 노이즈 제거
   - 중요하지 않은 방향은 버려
   - "진짜 패턴"만 남김

✅ 시각화 가능
   - 2D/3D 그래프로 바로 그려
   - 패턴 한눈에 확인

✅ 상관관계 해결
   - 온도와 압력이 같이 변하는 경우
   - PCA는 "이거 하나야!"라고 인지
```

### 단점 ❌

```
❌ 해석 어려움
   - PC1이 "온도*0.7 + 압력*0.3" 같은 복잡한 조합
   - "PC1이 높으면 수율이 좋다"는 이해하기 힘들어

❌ 정보 손실
   - 2개로 줄이면 일부 정보 손실
   - 90% 보존해도 10%는 사라짐

❌ 비선형 패턴 놓침
   - PCA는 선형 변환
   - 복잡한 비선형 관계는 못 찾아

❌ 스케일 민감
   - 표준화 안 하면 큰 숫자가 지배해
   - 반드시 StandardScaler 먼저!
```

---

## 🏭 실전 예시: 반도체 공장

### 상황

```
🏭 반도체 공장
📦 제품: 28nm 로직 칩
📊 데이터: 10,000 로트, 500개 파라미터
🎯 목적: "비슷한 공정 패턴" 찾기
```

### PCA 적용

```python
입력: 10,000 로트 × 500 파라미터

1. 표준화:
   - 모든 파라미터를 mean=0, std=1로 변환

2. PCA:
   - k=2로 2개 주성분 추출

3. 결과:
   PC1 설명력: 68.2%
   PC2 설명력: 21.5%
   총: 89.7%
```

### 결과 해석

```
📊 PC1 vs PC2 그래프:

PC2 ↑
    |  🔴🔴🔴   🔴🔴🔴   🔴🔴🔴
    | 🔴🔴🔴   🔴🔴🔴   🔴🔴🔴
    |🔴🔴🔴     🔴🔴🔴    🔴🔴🔴
    |
    |                🔵🔵🔵
    |              🔵🔵🔵🔵🔵
    |            🔵🔵🔵🔵🔵🔵🔵
    |
    |                      🟢🟢🟢
    |                   🟢🟢🟢🟢🟢
    +--------------------------→ PC1
   낮은                    중간    높은

🔴 그룹 A (PC1 낮음, PC2 높음):
   - 온도: 980°C, 압력: 45mTorr
   - 수율: 78.3% ± 6.2%

🔵 그룹 B (PC1 중간, PC2 낮음):
   - 온도: 1020°C, 압력: 52mTorr
   - 수율: 85.7% ± 3.1%

🟢 그룹 C (PC1 높음, PC2 낮음):
   - 온도: 1060°C, 압력: 58mTorr
   - 수율: 89.2% ± 2.8%

💡 통찰:
"PC1이 높을수록 수율이 좋고 편차가 작아!"
"그룹 C 조건으로 통일하면 수율 최적화 가능!"
```

### 군집별 분석

```
🔍 그룹 C의 특징:

PCA 역변환 (Inverse Transform):
- PC1=높음, PC2=낮음
→ 원래 파라미터 추정

결과:
- Annealing_Temp: 1050°C ± 10°C
- Etch_Pressure: 55 ± 3 mTorr
- Gas_Flow_AR: 32 ± 2 sccm
- Photo_Time: 125 ± 5 sec
- Plasma_Power: 210 ± 10 W

🎯 최적 공정 조건 제안:
1. 어닐링 온도를 1050°C로 제어
2. 에칭 압력을 55mTorr로 제어
3. 가스 유량을 32sccm으로 제어
```

---

## 🔍 PCA vs 다른 차원 축소

### PCA vs t-SNE vs UMAP

```
📊 PCA:
- 장점: 빠름, 해석 가능, 보존률 계산 가능
- 단점: 비선형 관계 놓침
- 용도: 전처리, 노이즈 제거, 빠른 탐색

🎨 t-SNE:
- 장점: 복잡한 비선형 구조 잘 보존
- 단점: 느림, 해석 어려움, 파라미터 민감
- 용도: 시각화, 군집 탐색

🗺️ UMAP:
- 장점: t-SNE보다 빠름, 글로벌 구조 보존
- 단점: 최신 기법, 설정 복잡
- 용도: 대용량 데이터 시각화
```

### 조합해서 사용

```
1단계: PCA
   500개 → 50개 (빠르게 줄이기)

2단계: t-SNE/UMAP
   50개 → 2개 (정밀하게 시각화)

최고의 효율성과 성능!
```

---

## 💡 한 줄 요약

**PCA는 "500개의 복잡한 데이터를 2개의 그림으로 줄여서 패턴을 보여주는 압축 AI"!** 📊🗜️

```
📏 1. 표준화: 모든 변수를 평등하게 (mean=0, std=1)
🎯 2. PCA: "진짜 중요한 방향" 2개 찾기
📈 3. 시각화: 2D 그래프로 패턴 확인
🔍 4. 해석: 군집, 이상치, 최적 조건 발견
💾 5. 저장: MinIO → Iceberg에 결과 저장
```

**핵심:**
- 500개 → 2개로 차원 축소
- 90% 정보 보존
- 2D 그래프로 시각화
- 군집 패턴 한눈에 확인
- PySpark로 빠르게 처리

---

## 🎓 수학적 배경 (심화)

### 주성분 찾는 방법

```
1. 공분산 행렬 계산 (Covariance Matrix)
   - 변수 간의 관계를 행렬로 표현

2. 고유값 분해 (Eigenvalue Decomposition)
   - 공분산 행렬의 고유값(Eigenvalue)과 고유벡터(Eigenvector) 계산

3. 고유값 정렬
   - 큰 고유값 순서대로 정렬
   - 큰 고유값 = "많은 정보를 담고 있는 방향"

4. 상위 k개 선택
   - k=2면 상위 2개 고유벡터 선택
   - 이것이 PC1, PC2!

5. 데이터 투영
   - 원본 데이터 × 고유벡터 = 주성분 값
```

### 설명된 분산 (Explained Variance)

```
고유값 λ₁, λ₂, λ₃, ... (λ₁ ≥ λ₂ ≥ λ₃ ≥ ...)

PC1 설명력 = λ₁ / (λ₁ + λ₂ + λ₃ + ...)
PC2 설명력 = λ₂ / (λ₁ + λ₂ + λ₃ + ...)
...

예시:
λ₁ = 500, λ₂ = 200, λ₃ = 100, ...
총합 = 1000

PC1 = 500 / 1000 = 0.5 (50%)
PC2 = 200 / 1000 = 0.2 (20%)
Total = 70%
```

**하지만 수학 몰라도 써! PySpark가 다 해줘!** 😊
