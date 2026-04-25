# Tiki-Tak-A
## 청소년 상담을 위한 AI 또래 상담 챗봇입니다.
##### 위험도 판별-> RAG 기반 정보 검색-> 또래 페르소나 응답 생성의 3단계로 이루어져으어ㅏ어어 어렵다.

#### 각 파트 별 세세한 내용은 각 하위 폴더의 READ.ME에 자세히 적어놨을거에요. 궁금하면 보세요...
##### - 데이터 설명: [data/README.md](data/README.md)
##### - 위치정보 RAG: [src/location_code/README.md](src/location_code/README.md)
##### - Vector DB (RAG): [src/rag_code/README.md](src/rag_code/README.md)
##### - 위험도 스코어링: [src/risk_code/README.md](src/risk_code/README.md)
##### - 또래 페르소나 SFT: [src/sft_code/README.md](src/sft_code/README.md)
##### - 베이스/튜닝 모델: [models/README.md](models/README.md)


## 환경구축 귀찮

##### python=3.12
##### 각자 컴퓨타에 맞는 torch 설치 알아서 하쇼
##### pip install -r requirements.txt

## 실행하는 방법(내 컴에선 일케 하니 잘되요)
##### > 전제 조건으로 모든 데이터 전처리 및 학습 완료된 모델이 준비되야해여
```bash
python src/main_app.py
```

## 모델 불러오는 순서:
##### 1.위험도 판별 모델 (RoBERTa) 로드
##### 2.RAG 시스템 (ChromaDB) 연결
##### 3.EXAONE 또래 페르소나 챗봇 로드
##### 4.위치정보 데이터 로드

## 프로젝트 구조
```
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
│   └── main_app.py         # 최종 통합 실행 코드
└── requirements.txt        # 필요한 라이브러리 정리
```