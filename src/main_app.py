import os
import torch
import json
import chromadb # FAISS 대신 ChromaDB 사용
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAONE_PATH = os.path.join(BASE_DIR, "models/base/EXAONE")
EXAONE_LORA_PATH = os.path.join(BASE_DIR, "models/tuned/exaone_sft_lora/final")
RISK_MODEL_PATH = os.path.join(BASE_DIR, "models/tuned/risk_model_final")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data/rag/vector_db")
LOCATION_DATA_PATH = os.path.join(BASE_DIR, "data/rag/location/unified_centers_lite.json")

class TikiTakaSystem:
    def __init__(self):
        print("🚀 [1/4] 위험도 판별 모델(RoBERTa) 로드 중...")
        self.risk_tokenizer = AutoTokenizer.from_pretrained(RISK_MODEL_PATH, local_files_only=True)
        self.risk_model = AutoModelForSequenceClassification.from_pretrained(
            RISK_MODEL_PATH, local_files_only=True
        ).to("cuda")
        
        print("🚀 [2/4] RAG 시스템(ChromaDB) 연결 중...")
        # 구축 시 사용했던 임베딩 함수 설정
        self.ko_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jhgan/ko-sbert-nli"
        )
        # ChromaDB 클라이언트 연결
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.client.get_collection(
            name="youth_statistics", 
            embedding_function=self.ko_embedding
        )
            
        print("🚀 [3/4] 위치 정보 데이터베이스 로드 중...")
        with open(LOCATION_DATA_PATH, "r", encoding="utf-8") as f:
            self.location_db = json.load(f)

        print("🚀 [4/4] EXAONE 3.5 SFT 모델 로드 중... (VRAM 집중 투하)")
        self.exa_tokenizer = AutoTokenizer.from_pretrained(EXAONE_PATH, trust_remote_code=True)
        
        # 1. 베이스 모델부터 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            EXAONE_PATH, 
            dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 🚨 [중요: 순서 변경] 
        # PEFT 어댑터를 붙이기 '전'에 입구와 출구 이름표를 먼저 달아줍니다.
        base_model.get_input_embeddings = lambda: base_model.transformer.wte
        base_model.get_output_embeddings = lambda: base_model.lm_head
        
        # 2. 그 다음에 LoRA 어댑터 결합
        print("🔗 LoRA 어댑터 결합 중...")
        self.exa_model = PeftModel.from_pretrained(base_model, EXAONE_LORA_PATH)
        self.exa_model.eval() # 추론 모드 전환
        
        print("\n✨ 모든 모듈 로드 완료! Tiki-Tak-A 상담을 시작합니다. ✨")

    def get_risk_score(self, text):
        inputs = self.risk_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            outputs = self.risk_model(**inputs)
        # Regression 모델이므로 logits이 바로 점수입니다.
        return outputs.logits.item()

    def search_rag(self, text, k=2):
        # ChromaDB 쿼리 실행
        results = self.collection.query(
            query_texts=[text],
            n_results=k
        )
        # 결과 텍스트들만 합쳐서 반환
        return "\n".join(results['documents'][0])

    def find_centers(self, region_keyword):
        return [c for c in self.location_db if region_keyword in c['region']][:2]

    def chat(self, user_input, region_keyword=None):
        risk_score = self.get_risk_score(user_input)
        context_str = self.search_rag(user_input)

        location_info = ""
        if risk_score > 75 and region_keyword:
            centers = self.find_centers(region_keyword)
            if centers:
                location_info = "\n[근처 도움받을 곳]\n" + "\n".join([f"- {c['name']} ({c['phone']}): {c['address']}" for c in centers])

        system_prompt = (
            "너는 따뜻하고 친근한 SNS 상담 친구야. 아래 제공된 [전문 지식]을 참고해서 답변해줘. "
            f"현재 내담자의 위험도 점수는 {risk_score:.1f}/100점이야. "
        )
        if risk_score > 75:
            system_prompt += "내담자가 매우 위험한 상태니까 정서적 지지와 함께 반드시 전문 센터 방문을 권유해줘."
        
        full_prompt = f"[|system|]{system_prompt}\n[전문 지식]: {context_str}\n{location_info}[|user|]{user_input}[|assistant|]"
        
        inputs = self.exa_tokenizer(full_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_tokens = self.exa_model.generate(
                **inputs, max_new_tokens=256, do_sample=True, temperature=0.7, repetition_penalty=1.2
            )
        
        response = self.exa_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return response.split("[|assistant|]")[-1].strip(), risk_score

if __name__ == "__main__":
    tikitaka = TikiTakaSystem()
    while True:
        user_msg = input("\n👤 유저: ")
        if user_msg.lower() in ['exit', 'quit', '종료']: break
        region = input("📍 (선택) 지역은? (없으면 엔터): ")
        ans, score = tikitaka.chat(user_msg, region if region else None)
        print(f"\n🚨 위험도 점수: {score:.2f}")
        print(f"🤖 Tiki-Tak-A: {ans}")