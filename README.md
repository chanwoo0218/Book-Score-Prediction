# Book Score Prediction

유저 정보와 도서 메타데이터를 바탕으로 사용자가 특정 도서에 부여할 **1–10점 평점**을 예측한 추천시스템 기반 회귀 프로젝트입니다.

## Key Decisions

- `0`점은 명시적 평점이 아니라 관측되지 않은 implicit feedback으로 보고 제외
- 희소성과 콜드스타트 영향을 줄이기 위해 `(user=3, item=3)` k-core filtering
- 동일 유저가 학습·평가에 존재하는 warm-start user holdout
- 제목·저자·출판사·지역 텍스트를 TF-IDF → Truncated SVD로 축소
- 고차원 ID는 OOF Target Encoding
- Ridge Regression과 UserMean baseline 비교

## Run

```bash
pip install -r requirements.txt
python src/train.py --data data/merged_book_ratings.csv
```

## Repository Note

현재 공개용 코드는 노션 포트폴리오와 발표 자료에 기록된 방법론을 재현하도록 정리한 버전입니다. 원본 노트북을 확보하면 `notebooks/`를 추가하고 결과 수치를 다시 검증하는 것이 좋습니다.
