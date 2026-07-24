# Book Score Prediction

사용자 정보와 도서 메타데이터로 **1–10점 명시적 평점**을 예측한 추천시스템 기반 회귀 프로젝트입니다. 첨부된 전처리·Ridge·CatBoost 실험 코드를 기준으로 저장소를 재구성했습니다.

## Data Processing

- `Book-Rating=0`은 명시적 평점이 아닌 implicit feedback으로 보고 제외
- 제목·저자·출판사·지역 문자열을 소문자화하고 공백·결측 토큰 정리
- `Location`의 마지막 쉼표 뒤에서 `Location_country` 추출
- 출판연도 `-1`과 비정상 결측값 제거
- 사용자와 도서가 각각 최소 3회 이상 등장하도록 반복적인 **3×3 p-core filtering**
- 동일 사용자의 일부 평점을 test로 보존하는 **warm-start user holdout**

## Experiments

### Linear/Ridge baseline

- `Book-Title`, `Book-Author`, `Publisher`, `Location_country`: TF-IDF → TruncatedSVD
- `Age`, `Year-Of-Publication`: 결측치 처리와 표준화
- Linear Regression과 Ridge를 비교하고 K-fold CV/GridSearch 수행

### Final CatBoost model

- 범주형: `User-ID`, `Book-ID`, `Year_Decade`
- 텍스트: 제목·저자·출판사별 TF-IDF(최대 3,000) → SVD 30차원
- 수치형: 나이, 출판연도, 텍스트 길이, `!`·`?` 개수
- validation early stopping으로 최적 iteration을 찾고 train+valid로 재학습

## Results

동일 warm-start test split에서:

| Model | RMSE | MAE |
|---|---:|---:|
| Ridge final | 1.55868 | 1.20898 |
| CatBoost final | **1.55290** | **1.19926** |

최종 코드는 User Mean baseline도 동일 split에서 다시 계산하여 절대·상대 개선 폭을 출력합니다.

## Repository Structure

```text
src/preprocess.py
src/train_catboost.py
notebooks/01_preprocessing.ipynb
notebooks/02_ridge_svd_baseline.ipynb
notebooks/03_catboost_final.ipynb
data/sample/df_final1_sample.csv
docs/presentation.pdf
```

## Run

```bash
pip install -r requirements.txt
python src/preprocess.py --input data/raw/train.csv --output data/processed/df_final1.csv
python src/train_catboost.py --data data/processed/df_final1.csv
```

## Data Policy

첨부된 `df_final1.csv.gz`는 원천 데이터의 재배포 조건이 명확하지 않아 저장소에 전체 업로드하지 않았습니다. 대신 컬럼 구조를 확인할 수 있는 소규모 sample과 전처리 코드를 제공합니다.