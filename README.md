# Tiki-Tak-A

## 청소년 상담을 위한 AI 또래 상담 챗봇입니다.
## 위험도 판별-> RAG 기반 정보 검색-> 또래 페르소나 응답 생성의 3단계로 이루어져으어ㅏ어어 어렵다.

### 각 파트 별 세세한 내용은 각 하위 폴더의 READ.ME에 자세히 적어놨을거에요. 궁금하면 보세요...
### - 데이터 설명: [data/README.md](data/README.md)
### - 위치정보 RAG: [src/location_code/README.md](src/location_code/README.md)
### - Vector DB (RAG): [src/rag_code/README.md](src/rag_code/README.md)
### - 위험도 스코어링: [src/risk_code/README.md](src/risk_code/README.md)
### - 또래 페르소나 SFT: [src/sft_code/README.md](src/sft_code/README.md)
### - 베이스/튜닝 모델: [models/README.md](models/README.md)


## 환경구축 귀찮

### python=3.12
### 각자 컴퓨타에 맞는 torch 설치 알아서 하쇼
### pip install -r requirements.txt

## 실행하는 방법(내 컴에선 일케 하니 잘되요)

### > 전제 조건으로 모든 데이터 전처리 및 학습 완료된 모델이 준비되야해여
### ```bash
### python src/main_app.py

### 모델 불러오는 순서:
#### 위험도 판별 모델 (RoBERTa) 로드
#### RAG 시스템 (ChromaDB) 연결
#### EXAONE 또래 페르소나 챗봇 로드
#### 위치정보 데이터 로드

## 프로젝트 구조

Tiki-Tak-A/
├── data/
│   ├── rag/
│   │   ├── location/       # 청소년 지원시설 위치 데이터
│   │   ├── statistics/     # 청소년 실태조사 PDF 및 처리 결과
│   │   └── vector_db/      # ChromaDB Vector DB
│   ├── risk/               # 위험도 학습용 상담 데이터
│   └── sft/                # 또래 페르소나 SFT 학습 데이터
├── models/
│   ├── base/               # 베이스 모델 (EXAONE, RoBERTa-large)
│   └── tuned/              # 파인튜닝 완료 모델
├── src/
│   ├── location_code/      # 위치정보 수집 및 전처리
│   ├── rag_code/           # Vector DB 구축
│   ├── risk_code/          # 위험도 스코어링 모델 학습
│   ├── sft_code/           # 또래 페르소나 SFT 학습
│   └── main_app.py         # 통합 실행 진입점
└── requirements.txt

## 데이터
### 아동·청소년 상담데이터(AI Hub) / 심리상담데이터(AI Hub) -> data/risk/.json
### 한국어 SNS 멀티턴 대화 데이터(AI Hub) / 온라인 구어체 말뭉치 데이터(AI Hub) -> data/sft/.json
### 청소년을 위한 지원시설 위치정보 9개 -> data/rag/location/.json
### 청소년에 대한 전반적인 실태 조사 정보 4개(학교밖, 위기청소년 등)->data/rag/statistics/.json
#### <배신> statistics api 데이터 18개는 껍데기 데이터.... 고로 버리고 성평등가족부에서 공개한 pdf 4개 다운로드->data/rag/statistics/source


## 위치정보 전처리(RAG)

### 1. src/location_code/1_fetch_api.py로 9개의 위치 데이터(json) 수집
### 2. src/location_code/2_unify로 하나의 통합 json으로 전처리
#### (이름, 주소, 지역, 연락처 키)+RAG 검색을 위한 text 필드 추가
#### 전화번호 정규화 추가
#### unified_centers_lite.json가 최종 결과


## Vector DB구축과정(RAG)

### 1. src/rag_code/1_extract_text.py로 텍스트 추출(PDF->4개의 json)
### 2. src/rag_code/2_chunking.py로 추출 텍스트 적당히 자르면서 통합 json만들기(4개의 json->1개의 json)
#### 문장 경계 기반 청킹/오버랩도 문장 단위/짧은 청크 병합
### 3. src/rag_code/3_build_chroma.py로 Vector DB구축(1개의 json->data/rag/vector_db)
#### 이때 사용하는 임베딩 모델 "jhgan/ko-sbert-nli"
#### jhgan/ko-sbert-nli (110M)
### 4. src/_test_db_code.py로 db작동 테스트
#### rag/vector_db가 최종 결과

## 위험도 스코어링 모델 학습

### 1. src/risk_code/1_merge_data.py로 data/risk에 있는 모든 데이터를 정규화 및 위험도 분류 3등급 정의 후 통합->total_risk_data.jsonl, 상담 페르소나를 위한 sft_from_risk_data.jsonl도 생성
### 2. src/risk_code/2_train_roberta.py로 total_risk_data.jsonl를 모델 학습
#### 사용하는 모델: RoBERTa-large (330M)
#### 총 상담데이터 4387건
#### <분류>
#### 안전: 3220건, 주의:996건, 위험:171건
#### 데이터 불균형을 완화하기 위한 클래스 가중치: ['0.45', '1.47', '8.55']
#### 모델 주요 학습 파라미터: max_length=512, batch_size=32, EPOCHS = 15,LEARNING_RATE = 2e-5
#### 사용 VRAM: 31299MiB /  32607MiB 이 이상은 불가능함 천장 찍음
#### 최종 결과 models/tuned/risk_model_final

## 또래 페르소나 챗봇 모델 학습

### sft_from_risk_data.jsonl 362,107건 너무너무 많음
### src/sft_code/filter_counseling.py로 30,000건으로 추출
#### 추출 시 기준
#### 내담자 발화 최소 15자, 상담사 응답 최소 30자
#### 응답 상한 500자 추가
#### 단순 질문 기준 최소 30자
#### 비상담 패턴("다음 상담은", "예약", "동의서" 등) 탈락
#### 상담 패턴("힘들다", "괜찮아", "마음", "불안" 등) 공감 키워드 포함된 응답 추출
### 추출한 30,000건을 "또래 페르소나"를 위해 src/sft_code/convert_tone.py로 말투 변환(EXAONE_7.8B 로컬 모델 활용)->무려 6시간....
#### 변환예시_1:
#### 내담자: "다른 사람들 눈치 안 보고 소리를 지를 수 있어서 좋아요."
#### 변환답변_또래상담자: "최근 일주일간 짜증 났던 거 진짜 많았니?"
#### 원래 상담자답변: "최근 일주일 동안 크게 짜증이 났거나 화가 난 적이 있어?"
#### 변환예시_2
#### 내담자: "그냥 이 솔직히 좀 말하면은 그냥 담배를 아예 며칠 못 보는 곳으로 가든가 아니면은 그냥 주변에서 그냥 계속 말려주는 게 확실히 이거는 좀 많이 도움이 될 것 같아요."
#### 변환답변_또래상담자: "진짜? 주변 사람들이 도와주고, 담배 못 피우면 해결되나 봐. 그럴 때마다 힘내!"
#### 원래 상담자답변: "그래요. 그래요. 주변 사람들의 도움. 그다음에 아예 담배를 피울 수 없는 상황."