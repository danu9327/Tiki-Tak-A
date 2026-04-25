# pdf로 객관적인 청소년 관련 주요 정보를 챗봇이 확인할수 있는 Vector DB만드는 코드

## 청소년 실태조사 PDF 4개를 Vector DB로 구축하는 파이프라인
## 질문과 관련된 통계 정보를 검색하는 데 사용해서 할루시네이션(거짓말?) 못치게하는 용도겸사겸사

## 실행순서
### 1. src/rag_code/1_extract_text.py로 텍스트 추출(PDF->4개의 json)
### 2. src/rag_code/2_chunking.py로 추출 텍스트 적당히 자르면서 통합 json만들기(4개의 json->1개의 json)
### 3. src/rag_code/3_build_chroma.py로 Vector DB구축(1개의 json->data/rag/vector_db)


단계	입력	출력
1_extract_text	PDF 4개 (data/rag/statistics/source/)	JSON 4개 (statistics/processed/)
2_chunking	JSON 4개	final_rag_chunks.json (statistics/db_input/)
3_build_chroma	final_rag_chunks.json	ChromaDB (data/rag/vector_db/)

## 1_extract_text.py
### pdf에서 txt 추출하는 코드래
### 아래와 같은 출력 구조로 나옴
{
  "source": "2023학교밖청소년실태조사최종보고서.pdf",
  "page": 42,
  "content": "추출된 텍스트 내용...\n\n[표] ...",
  "has_table": true
}

## 2_chunking.py
### 추출한 텍스트를 ChromaDB에 넣기 위해 전처리하는 코드
#### 문장 경계 기반 청킹/오버랩도 문장 단위/짧은 청크 병합
### 청킹? 설정
파라미터	값	설명
MAX_CHUNK_SIZE	500자	청크 최대 길이
MIN_CHUNK_SIZE	100자	이보다 짧으면 인접 청크에 병합
OVERLAP_SENTENCES	1문장	앞 청크 마지막 1문장을 다음 청크 앞에 붙임
### 아래와 같은 출력 구조로 나옴
{
  "source": "2023학교밖청소년실태조사최종보고서.pdf",
  "page": 42,
  "chunk_index": 2,
  "content": "청킹된 텍스트..."
}

## 3_build_chroma.py
### 첰크?를 DB로 구축하는 코드>? 사실 이해가 안됨

항목	값
임베딩 모델	jhgan/ko-sbert-nli (110M, 최초 실행 시 자동 다운로드)
DB 엔진	ChromaDB (PersistentClient)
컬렉션명	youth_statistics
배치 크기	1,000개
저장 위치	data/rag/vector_db/
#### 이때 사용하는 임베딩 모델 "jhgan/ko-sbert-nli"
#### jhgan/ko-sbert-nli (110M)