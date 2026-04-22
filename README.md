# Tiki-Tak-A

## 환경구축 귀찮

### python=3.12
### 각자 컴퓨타에 맞는 torch 설치 알아서 하쇼
### pip install -r requirements.txt

## 데이터
### 아동·청소년 상담데이터(AI Hub) -> data/risk/.json
### 한국어 SNS 멀티턴 대화 데이터(AI Hub) -> data/sft/.json
### 심리상담 데이터(AI Hub) -> data/risk/.json

#### 다운로드 받은 데이터 중 특정 포맷 파일만 옮기기
#### -type f -name "*.json" -exec mv {} data/risk/psych/ \;

### API Key랑 27개 각 데이터 url를 통해서 .json으로 data/rag로 분류해서 수집
### src/fetch_api.py로 실행

