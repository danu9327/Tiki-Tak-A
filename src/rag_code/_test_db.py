import os
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DIR = os.path.join(BASE_DIR, "data/rag/vector_db")

def test_search(query_text, n_results=3):
    print(f"\n🔍 질문의 의미를 분석하여 데이터베이스를 뒤지는 중...")
    
    # 임베딩 모델 로드 (캐시된 모델을 바로 불러옵니다)
    ko_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jhgan/ko-sbert-nli"
    )
    
    # DB 연결
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name="youth_statistics", embedding_function=ko_embedding)
    
    # Vector DB에 유사도 검색 요청 (핵심 로직!)
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # 검색 결과 예쁘게 출력
    print("\n=== 🎯 가장 유사한 문서 Top 3 ===")
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i] # Vector Distance (숫자가 작을수록 의미가 비슷함)
        
        print(f"\n🥇 [{i+1}순위] 출처: {meta['source']} (📄 {meta['page']}페이지)")
        print(f"   - 관련성 지수 (Distance): {dist:.4f}")
        print(f"   - 추출된 내용:\n{doc}\n")
        print("-" * 50)

if __name__ == "__main__":
    print("🤖 RAG 지식 창고 검색 테스트를 시작합니다!")
    while True:
        user_input = input("\n💡 청소년 관련 질문을 던져보세요! (종료: q)\n >> ")
        if user_input.lower() in ['q', 'ㅂ', 'exit']:
            print("테스트를 종료합니다.")
            break
        test_search(user_input)