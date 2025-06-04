# dl_project
# 🛒 Amazon Product Review Sentiment & Rating Prediction

본 프로젝트는 **Amazon 상품 리뷰** 데이터를 기반으로,  
리뷰의 **감성(긍정/부정)**을 분류하고 **별점(1~5점)**을 예측하는 머신러닝/딥러닝 프로젝트입니다.

<br>

## 📌 프로젝트 개요

- **주제**: 자연어 처리(NLP)를 활용한 감성 분석 및 별점 예측
- **데이터**: [Amazon Product Review Dataset (Kaggle)](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- **목표**:
  - 리뷰 텍스트 → 감성(긍정/부정) 분류
  - 리뷰 텍스트 → 별점 예측 (1~5점)

<br>

## 🧾 데이터셋 소개

- 리뷰 텍스트 (Text)
- 별점 (Rating: 1~5)
- 제품 카테고리, 리뷰 제목, 작성자 등 부가 정보 포함

→ 본 프로젝트에서는 텍스트와 별점 중심으로 사용합니다.

<br>

## 🔧 전처리 과정

- 텍스트 정제: 소문자 변환, 특수문자 제거 등
- 토큰화 및 불용어 제거
- 리뷰 길이 분포 및 별점 분포 시각화
- 감성 라벨 생성:  
  `Rating 1~2 → 부정`, `4~5 → 긍정`, `3점은 제외`

<br>

## 📊 감성 분석 (Sentiment Classification)

### ✅ 사용 모델
- Baseline: TF-IDF + Logistic Regression
- 딥러닝: LSTM, BERT

### ✅ 평가지표
- Accuracy
- F1-score

### ✅ 결과 예시
> "This product is amazing!" → 긍정  
> "Worst purchase ever..." → 부정

<br>

## 🌟 별점 예측 (Rating Prediction)

### ✅ 접근 방법
- 회귀 (Regression) 또는 다중 분류 (Multiclass Classification)

### ✅ 사용 모델
- 머신러닝: RandomForest, XGBoost
- 딥러닝: LSTM, BERT

### ✅ 평가지표
- 회귀: MAE, RMSE  
- 분류: Accuracy, Confusion Matrix

<br>

## 📈 성능 비교 및 분석

| 모델 | 감성 분석 정확도 | 별점 예측 RMSE |
|------|------------------|----------------|
| TF-IDF + LR | 86% | - |
| LSTM | 89% | 0.84 |
| BERT | **92%** | **0.76** |

→ BERT 기반 모델이 가장 높은 성능을 보였습니다.

<br>

## 💡 결론 및 향후 계획

- 리뷰 텍스트만으로도 고객 감정을 상당히 정확히 파악 가능
- 실제 커머스에서 리뷰 모니터링/자동 분류 등에 응용 가능
- 향후 발전 방향:
  - 3점 리뷰 포함한 감정 분류 (중립 클래스 추가)
  - 리뷰 + 상품 정보 결합한 멀티모달 학습
  - 실시간 리뷰 분석 API 구현

<br>

## 📁 디렉토리 구조

