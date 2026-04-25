# 베이스 모델이랑 튜닝 완료된 모델 설명

## 베이스모델과 파인튜닝한 모델을 소개할게여
```
models/
├── base/
│   ├── EXAONE/             # 또래 페르소나 SFT 베이스 모델
│   └── roberta-large/      # 위험도 스코어링 베이스 모델
└── tuned/
└── risk_model_final/   # 위험도 스코어링 파인튜닝 완료 모델
```
## 베이스 모델

### EXAONE-3.5-7.8B (LG AI Research)
### 또래 페르소나 챗봇 SFT의 베이스 모델로써 중점적으로 대화를 하는 모델
### 파라미터는 보는대로 7.8B
### 히든 사이즈?는 4096
### 어텐션 헤드?는 32개래요
### sft데이터 학습할때 사용할거에요
#### sft데이터로 모델 학습하는 과정이 궁금하다면 -> [src/sft_code/README.md](../src/sft_code/README.md)

### klue/roberta-large
### 위험도 분류해주는 베이스 모델임
### 파라미터는 330M
### 히든사이즈는 1024
### 레이어?는 24개
### 어텐션 헤드는 16개
### 상담 내용을 토대로 위험도를 3개로 분류
### risk데이터로 학습할때 사용임
#### 자세한 학습 과정은 -> [src/risk_code/README.md](../src/risk_code/README.md)


### jhgan/ko-sbert-nli(110M)
### 위 2개의 모델과 달리 가벼워서 로컬로 다운안받아서 로컬에도 없는 비운의 임베딩 베이스 모델
### Vector DB를 만드는 과정에서 임베딩을 위해 사용한 모델
### rag데이터로 DB만들때 사용함
#### 자세한 DB 구축 과정에서의 임베딩 모델 사용 설명은->[src/rag_code/README.md](src/rag_code/README.md)

## 3개 모델의 역할
모델	크기	역할	입력 → 출력
jhgan/ko-sbert-nli	110M	텍스트 → 숫자벡터 변환	문장 → 768차원 벡터
klue/roberta-large	330M	위험도 분류	문장 → [안전/주의/위험]
EXAONE 7.8B	7.8B	답변 생성	대화 → 다음 문장
