import os
import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_JSON_PATH = os.path.join(BASE_DIR, "data/rag/statistics/db_input/final_rag_chunks.json")
# Vector DB가 실제로 저장될 물리적 폴더
DB_DIR = os.path.join(BASE_DIR, "data/rag/vector_db") 

os.makedirs(DB_DIR, exist_ok=True)

def build_vector_db():
    print("🚀 Vector DB 구축을 시작합니다...")
    
    # 2. 한국어 특화 임베딩 모델 로드 (허깅페이스에서 자동으로 다운로드됨)
    print("📥 한국어 임베딩 모델(Ko-SBERT)을 불러오는 중... (최초 실행 시 시간 소요)")
    ko_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jhgan/ko-sbert-nli"
    )

    # 3. ChromaDB 클라이언트 및 컬렉션(테이블) 생성
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # 기존에 같은 이름의 컬렉션이 있다면 초기화 (중복 방지)
    try:
        client.delete_collection(name="youth_statistics")
    except Exception:  # 종류에 상관없이 에러가 나면(DB가 없으면) 쿨하게 넘어갑니다.
        pass
        
    collection = client.create_collection(
        name="youth_statistics",
        embedding_function=ko_embedding
    )

    # 4. 청크 데이터 불러오기
    print("📂 청크 데이터를 읽는 중...")
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 5. 데이터를 ChromaDB 포맷으로 분리
    documents = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        documents.append(chunk["content"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"]
        })
        # 각 청크의 고유 ID (예: doc_1, doc_2 ...)
        ids.append(f"doc_{idx}")

    # 6. 한 번에 넣기 (배치 처리)
    # 5,000개 정도는 한 번에 들어가지만, 진행률을 보기 위해 1000개씩 나눠 넣습니다.
    batch_size = 1000
    print(f"\n💾 총 {len(documents)}개의 청크를 Vector DB에 임베딩하여 저장합니다.")
    print("이 작업은 컴퓨터의 CPU/GPU 성능에 따라 수 분이 걸릴 수 있습니다.")
    
    for i in tqdm(range(0, len(documents), batch_size)):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

    print("\n🎉 Vector DB 생성이 완벽하게 끝났습니다!")
    print(f"데이터베이스 저장 위치: {DB_DIR}")

if __name__ == "__main__":
    build_vector_db()