# 데이터

##### 학습 및 RAG에 들어가는 모든 데이터에 대한 간략한 설명(언제 다 써)
```
data/
├── risk/                   # 위험도 스코어링 모델 학습 원본 데이터
│   ├── psych/              # 심리상담 데이터 (AI Hub)
│   └── youth/              # 아동·청소년 상담 데이터 (AI Hub)
├── sft/                    # 한국어 SNS 멀티턴 대화 데이터로 또래 페르소나 SFT 학습 데이터
├── rag/
│   ├── location/           # 청소년 지원시설 위치 JSON
│   ├── statistics/         # 청소년 실태조사 PDF 및 처리 결과
│   └── vector_db/          # ChromaDB Vector DB (구축 완료본)
└── total_risk_data.jsonl   # 위험도 학습용 최종 통합 데이터
```
##### 아동·청소년 상담데이터(AI Hub) / 심리상담데이터(AI Hub) -> `data/risk/.json`
##### 한국어 SNS 멀티턴 대화 데이터(AI Hub) / 온라인 구어체 말뭉치 데이터(AI Hub) -> `data/sft/.json`
##### 청소년을 위한 지원시설 위치정보 9개 -> `data/rag/location/.json`
##### 청소년에 대한 전반적인 실태 조사 정보 4개(학교밖, 위기청소년 등)->`data/rag/statistics/.json`
##### <배신> statistics api 데이터 18개는 껍데기 데이터.... 고로 버리고 성평등가족부에서 공개한 pdf 4개 다운로드->`data/rag/statistics/source`


## 위치정보 전처리(RAG)
##### 청소년 지원시설 성평등가족부 오픈 API로 공개된 9개 가져옴
##### `src/location_code/1_fetch_api.py`로 9개의 위치 데이터(json) 수집
##### `청소년쉼터_위치현황.json`, `청소년상담복지센터_위치현황.json` 등등
##### `unified_centers_lite.json` | **최종 통합본** (이름/주소/연락처/RAG text 필드 포함)
##### 자세한 수집 및 전처리 과정은 여기서 보쇼 → [src/location_code/README.md](../src/location_code/README.md)


## Vector DB구축과정(RAG)
##### 청소년 실태조사 보고서 기반 RAG 데이터
##### rag/statistics/source/는 성평등가족부가 공개한 PDF 4개 원본 그대로
##### rag/statistics/processed/는 pdf에서 텍스트 추출한 중간물(JSON 4개요)
##### rag/statistics/db_input/은 청킹? 이란거 처리하고 Chroma DB에 넣을 통합 json 1개
##### 자세한 DB구축 과정은 → [src/rag_code/README.md](../src/rag_code/README.md)
|<수록 PDF 목록>|
|--------------|
|2024년 청소년 매체이용 및 유해환경 실태조사 최종 보고서|
|2023년 청소년종합실태조사|
|2023학교밖청소년실태조사 최종보고서|
|2024년 위기청소년 지원기관 이용자 생활실태조사|
##### data/rag/vector_db/는 ChromaDB로 구축된 Vector DB로 미리 스포하자면 `src/rag_code/3_build_chroma.py` 실행해서 만든거에요

## 위험도 스코어링 모델을 위한 데이터(risk)
##### AI Hub에서 수집한 상담 데이터들(심리상담데이터(psych)/아동청소년상담데이터(youth))
##### risk/psych/는 심리상담 데이터 (AI Hub) | 우울증 등 심리 상담 세션
##### risk/youth는 아동·청소년 상담 데이터 (AI Hub) | 청소년 대상 상담 세션
##### `src/risk_code/1_merge_data.py`로 data/risk에 있는 모든 데이터를 정규화 및 위험도 분류 3등급 정의 후 통합->`total_risk_data.jsonl`, 상담 페르소나를 위한 `sft_from_risk_data.jsonl`도 생성
|총 상담데이터 4387건|개별 건수/가중치|
|-------------------|--------------|
|<분류>|안전: 3220건, 주의:996건, 위험:171건|
|데이터 불균형을 완화하기 위한 클래스 가중치|['0.45', '1.47', '8.55']|

## 또래 페르소나 챗봇 모델 학습
##### 또래스러운 MZ는 아니라도 친근한 대화를 위한 데이터
#### <sft_from_risk_data.jsonl>
##### `sft_from_risk_data.jsonl`(risk 데이터에서 생성된 상담 페르소나 원본) 362,107건 너무너무 많음
##### `src/sft_code/filter_counseling.py`로 30,000건으로 품질 필터링 후 추출
##### 추출한 30,000건을 "또래 페르소나"를 위해 `src/sft_code/convert_tone.py`로 말투 변환(EXAONE_7.8B 로컬 모델 활용)->무려 6시간....
##### 자세한 필터링 및 변환 설명은->[src/sft_code/README.md](../src/sft_code/README.md)
#### <sft_from_sns_data.jsonl>
##### 한국어 sns데이터 json이 176,605개 또 너무너무너무 많음
##### 토픽별(건강및식음료,경제및사회,과학기술,문화생활및여가,미용과패션,스포츠및e스포츠,여행관광및명소,정치,콘텐츠소비)
##### 균등하게 11,000건만 선택해서 통합한게 `sft_from_sns_data.jsonl`
##### 균등 추출에 대한 자세한 설명은 해야겟지?->[src/sft_code/README.md](../src/sft_code/README.md)