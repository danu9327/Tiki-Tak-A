import os
import json
import uuid
import torch
import chromadb
from flask import Flask, request, jsonify, send_from_directory
from enum import Enum
from chromadb.utils import embedding_functions
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

EXAONE_BASE_PATH   = os.path.join(BASE_DIR, "models/base/EXAONE")
EXAONE_LORA_PATH   = os.path.join(BASE_DIR, "models/tuned/exaone_sft_lora/final")
RISK_MODEL_PATH    = os.path.join(BASE_DIR, "models/tuned/risk_model_final")
VECTOR_DB_PATH     = os.path.join(BASE_DIR, "data/rag/vector_db")
LOCATION_DATA_PATH = os.path.join(BASE_DIR, "data/rag/location/unified_centers_lite.json")

LABEL_NAMES = ["안전", "주의", "위험"]
RAG_DISTANCE_THRESHOLD = 1.5

SYSTEM_PROMPT = (
    "너는 청소년 또래 친구처럼 편하게 대화하면서도, "
    "상대방의 고민을 진심으로 들어주고 도움을 줄 수 있는 상담 챗봇이야. "
    "예의바르게 반말을 사용하고, 아래 원칙을 꼭 지켜줘.\n"
    "상대방의 외모, 성격, 행동을 탓하거나 문제의 원인을 상대방에게 돌리지 마. "
    "'네가 뭘 잘못한 거 아니야?', '네 행동에 문제가 있는 건 아닌지' 같은 말은 절대 하지 마.\n"
    "상대방 편을 들어주고 조언은 부드럽게 짧게 답해줘"

)

RAG_TRIGGER_KEYWORDS = [
    "힘들", "우울", "불안", "스트레스", "자살", "죽고", "외로",
    "왕따", "따돌림", "폭력", "학대", "가출", "중독", "게임",
    "성적", "시험", "진로", "취업", "학교", "친구", "부모",
    "걱정", "무서", "두려", "슬프", "화나", "짜증", "고민",
    "상담", "도움", "지원", "센터", "쉼터", "통계", "현황",
]

DANGER_KEYWORDS = [
    "죽고 싶", "죽을래", "죽어버리", "죽을 거", "죽었으면", "죽고싶",
    "자살", "자해", "손목", "옥상", "뛰어내리", "목매", "약 먹",
    "유서", "마지막", "끝내고 싶", "사라지고 싶", "없어지고 싶",
    "살고 싶지 않", "살기 싫", "태어나지 말", "안 태어났으면",
    "때려", "때리", "맞아", "맞았", "폭력", "학대", "구타",
    "성폭력", "성추행", "강간", "성폭행", "몰카", "불법촬영",
    "강제로", "억지로", "만져", "만졌", "신체 접촉",
    "가출", "도망", "집 나가", "집 나왔", "쫓겨났",
    "버림받", "방치", "방임", "굶겨", "밥을 안 줘",
    "무서워", "두려워", "겁나", "위협", "협박", "죽이겠",
    "칼", "흉기", "찌르", "피가 나", "피가 났",
    "왕따", "따돌림", "괴롭힘", "집단폭행", "린치",
    "돈 빼앗", "돈 뺏", "셔틀", "빵셔틀",
]

CAUTION_KEYWORDS = [
    "힘들어", "힘든", "우울", "불안", "외로워", "외롭", "슬퍼", "슬프",
    "짜증", "화가 나", "화나", "분노", "답답", "막막", "절망",
    "무기력", "의욕", "아무것도 하기 싫", "귀찮", "지쳐", "지치",
    "눈물", "울었", "울고", "멘붕", "멘탈",
    "잠이 안", "잠을 못", "불면", "악몽", "가위눌",
    "밥을 안", "먹기 싫", "식욕", "토하", "구토",
    "아파", "다쳤", "멍이", "상처",
    "학교 가기 싫", "등교 거부", "결석", "자퇴", "전학",
    "친구가 없", "혼자", "외톨이", "놀림", "무시",
    "선생님이", "선생님한테",
    "부모님", "엄마", "아빠", "싸워", "싸움", "이혼", "별거",
    "집에 가기 싫", "집이 싫", "가족이 싫",
    "고민", "걱정", "스트레스", "압박", "부담",
    "중독", "게임 중독", "도박", "음주", "술",
    "진로", "성적", "시험", "수능", "입시",
]

SUPPORT_SOURCE_KEYWORDS = ["상담", "청소년", "쉼터", "복지", "지원", "보호"]


class AppState(Enum):
    CHAT = "chat"
    AWAITING_CONSENT = "awaiting_consent"      # 상담 제안 수락/거절 대기
    AWAITING_LOCATION = "awaiting_location"    # 동네 입력 대기


# ============================================================
# 세션 관리 (사용자별 대화 상태)
# ============================================================
sessions = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "state": AppState.CHAT,
            "offered_support": False,
        }
    return sessions[session_id]


# ============================================================
# 모델 로드 (서버 시작 시 1회)
# ============================================================
def load_models():
    print("📥 모델 로딩 중... (최초 실행 시 1~2분 소요)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   디바이스: {device}")

    print("  [1/4] EXAONE SFT LoRA 모델...")
    tokenizer = AutoTokenizer.from_pretrained(EXAONE_BASE_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        EXAONE_BASE_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    try:
        base_model.get_input_embeddings()
    except NotImplementedError:
        base_model.get_input_embeddings = lambda: base_model.transformer.wte
    try:
        base_model.get_output_embeddings()
    except NotImplementedError:
        base_model.get_output_embeddings = lambda: base_model.lm_head

    exaone = PeftModel.from_pretrained(base_model, EXAONE_LORA_PATH)
    exaone.eval()
    exaone_device = next(exaone.parameters()).device

    print("  [2/4] RoBERTa 위험도 분류 모델...")
    risk_tokenizer = AutoTokenizer.from_pretrained(RISK_MODEL_PATH)
    risk_model = AutoModelForSequenceClassification.from_pretrained(RISK_MODEL_PATH)
    risk_model.to(device)
    risk_model.eval()

    print("  [3/4] ChromaDB 청소년 통계 DB...")
    ko_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jhgan/ko-sbert-nli"
    )
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    stats_collection = chroma_client.get_collection(
        name="youth_statistics",
        embedding_function=ko_embedding,
    )

    print("  [4/4] 지원시설 위치 데이터...")
    with open(LOCATION_DATA_PATH, "r", encoding="utf-8") as f:
        location_data = json.load(f)

    print("✅ 모든 모델 로드 완료!\n")
    return {
        "exaone": exaone,
        "exaone_device": exaone_device,
        "tokenizer": tokenizer,
        "risk_model": risk_model,
        "risk_tokenizer": risk_tokenizer,
        "risk_device": device,
        "stats_collection": stats_collection,
        "location_data": location_data,
    }


# ============================================================
# 위험도 분류 (키워드 + 모델 하이브리드)
# ============================================================
def classify_risk(user_input, history, models):
    recent_texts = [t["user"] for t in history[-3:]]
    recent_texts.append(user_input)
    combined = " ".join(recent_texts)

    if any(kw in combined for kw in DANGER_KEYWORDS):
        return 2

    inputs = models["risk_tokenizer"](
        combined, return_tensors="pt", max_length=512,
        truncation=True, padding=True,
    ).to(models["risk_device"])

    with torch.no_grad():
        logits = models["risk_model"](**inputs).logits
    model_pred = torch.argmax(logits, dim=1).item()

    if model_pred == 0 and any(kw in combined for kw in CAUTION_KEYWORDS):
        return 1
    return model_pred


# ============================================================
# RAG 검색
# ============================================================
def should_search_rag(user_input, risk_level):
    if risk_level >= 1:
        return True
    return any(kw in user_input for kw in RAG_TRIGGER_KEYWORDS)

def search_stats(query, models, n_results=2):
    results = models["stats_collection"].query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    distances = results["distances"][0]
    return [doc for doc, dist in zip(docs, distances) if dist < RAG_DISTANCE_THRESHOLD]
def retrieve_candidates_from_list(user_input, location_data, top_k=8):
    # 검색에 불필요한 단어들 제거
    stopwords = ["살아", "살아요", "옆에", "근처", "어디", "있어", "알려줘", "동네"]
    keywords = [k for k in user_input.split() if len(k) >= 2 and k not in stopwords]
    
    if not keywords:
        return []
    
    scored_results = []
    for item in location_data:
        score = 0
        addr = item.get('address', '')
        name = item.get('name', '')
        
        for kw in keywords:
            # 주소에 포함되면 높은 점수 (지역 기반이므로)
            if kw in addr: score += 10
            # 시설 이름에 포함되면 중간 점수
            if kw in name: score += 5
            
        if score > 0:
            scored_results.append((score, item))
    
    # 점수 높은 순으로 정렬
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [res[1] for res in scored_results[:top_k]]

# ============================================================
# 응답 생성 함수들
# ============================================================
def build_prompt(history, user_msg, risk_level, rag_context=None):
    system = SYSTEM_PROMPT
    if rag_context:
        stats_block = "\n".join(f"- {doc}" for doc in rag_context)
        user_msg = f"[참고 통계]\n{stats_block}\n\n{user_msg}"
    if risk_level == 2:
        system += "\n지금 상대방이 많이 힘든 상황임을 인지해. 예의바르게 걱정 어린 말투로 먼저 괜찮은지 확인해줘."
    elif risk_level == 1:
        system += "\n상대방이 조금 힘든 것 같아. 공감해주고 조심스럽게 물어봐줘."

    prompt = f"[|system|]{system}[|endofturn|]\n"
    for turn in history[-3:]:
        prompt += f"[|user|]{turn['user']}\n[|assistant|]{turn['assistant']}[|endofturn|]\n"
    prompt += f"[|user|]{user_msg}\n[|assistant|]"
    return prompt


def _generate(models, prompt, max_new_tokens=150, temperature=0.7):
    tokenizer = models["tokenizer"]
    inputs = tokenizer(prompt, return_tensors="pt").to(models["exaone_device"])
    with torch.no_grad():
        output_ids = models["exaone"].generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.9, repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.split("[|endofturn|]")[0].strip()


def generate_response(user_msg, history, models, risk_level, rag_context=None):
    prompt = build_prompt(history, user_msg, risk_level, rag_context)
    result = _generate(models, prompt)
    return result if result else "응, 계속 말해줘."


def generate_support_offer(user_msg, history, models):
    system = (
        "전문 상담을 받아보는 게 어떨지 자연스럽게 제안해줘. "
        "상담이나 지원시설의 도움을 받는게 어떤지 자연스럽게 제안해줘"
        "동네를 알려주면 근처 지원시설을 찾아줄 수 있다고 말해줘. "
        "반말로, 짧고 예의바르게, 부담 안 주는 톤으로."
    )
    prompt = f"[|system|]{system}[|endofturn|]\n"
    for turn in history[-2:]:
        prompt += f"[|user|]{turn['user']}\n[|assistant|]{turn['assistant']}[|endofturn|]\n"
    prompt += f"[|user|]{user_msg}\n[|assistant|]"
    result = _generate(models, prompt, max_new_tokens=80)
    return result if result else "혹시 전문 상담 한번 받아볼 생각 있어? 동네 알려주면 근처 센터 찾아줄게."


def generate_comfort_after_search(models):
    system = (
        "너는 청소년 또래 친구야. 방금 상대방에게 상담 지원시설 정보를 알려줬어. "
        "혼자 감당하지 않아도 된다고 따뜻하게 응원해줘. 반말로, 짧고 예의바르게, 진심 어린 톤으로."
    )
    prompt = f"[|system|]{system}[|endofturn|]\n[|user|]응원해줘\n[|assistant|]"
    result = _generate(models, prompt, max_new_tokens=60, temperature=0.8)
    return result if result else "혼자 감당 안 해도 돼. 나도 응원할게!"


def generate_decline_response(user_msg, models):
    system = (
        "너는 청소년 또래 친구야. 상대방에게 상담 지원시설을 제안했는데 거절했어. "
        "괜찮다고 하면서, 언제든 필요하면 말하라고 해줘. "
        "청소년 전화 1388은 24시간 무료라는 것도 자연스럽게 알려줘. "
        "짧게, 예의바르게 부담 안 주는 톤으로."
    )
    prompt = f"[|system|]{system}[|endofturn|]\n[|user|]{user_msg}\n[|assistant|]"
    result = _generate(models, prompt, max_new_tokens=80)
    return result if result else "알겠어! 언제든 필요하면 말해줘. 1388은 24시간이야!"


# ============================================================
# 시설 검색
# ============================================================
def recommend_centers_with_llm(user_location_input, models):
    """
    LLM이 위치 데이터(JSON)를 직접 보고 가장 적합한 시설을 추천함
    """
    candidates = retrieve_candidates_from_list(user_location_input, models["location_data"])
    print(f"DEBUG: 후보군 개수 -> {len(candidates)}")
    candidates_text = ""
    for i, c in enumerate(candidates):
        candidates_text += f"[{i+1}] {c['name']} | 주소: {c['address']} | 전화: {c.get('phone', '1388')}\n"

    system_msg = (
        "너는 위치 정보 전문가야. 사용자의 현재 위치를 바탕으로 가장 적절한 청소년 지원시설(상담센터, 쉼터 등)을 추천해줘.\n"
        "아래 제공된 [시설 목록]에서만 선택해야 하며, 사용자가 말한 위치와 지리적으로 가장 가까운 곳 2~3곳을 골라줘.\n"
        "시설 이름, 주소, 전화번호를 친절하게 안내하고, 왜 그곳을 추천했는지 짧게 설명해줘.\n"
        "만약 적절한 곳이 없다면 솔직하게 말하고 1388을 안내해."
    )
    
    user_msg = f"사용자 위치: {user_location_input}\n\n[시설 목록]\n{candidates_text}"
    
    prompt = f"[|system|]{system_msg}[|endofturn|]\n[|user|]{user_msg}\n[|assistant|]"

    # 3. LLM 추론
    tokenizer = models["tokenizer"]
    exaone = models["exaone"]
    inputs = tokenizer(prompt, return_tensors="pt").to(models["exaone_device"])

    with torch.no_grad():
        output_ids = exaone.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    recommendation = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return recommendation.split("[|endofturn|]")[0].strip()


def is_declining(text):
    declines = ["아니", "괜찮아", "됐어", "안해", "싫어", "필요없어", "나중에"]
    return any(d in text for d in declines)


# ============================================================
# Flask 앱
# ============================================================
app = Flask(__name__, static_folder=STATIC_DIR)
models = None


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/session", methods=["POST"])
def create_session():
    """새 세션 생성"""
    session_id = str(uuid.uuid4())
    get_session(session_id)
    return jsonify({"session_id": session_id})


@app.route("/api/chat", methods=["POST"])
def chat():
    """메인 채팅 API"""
    data = request.json
    session_id = data.get("session_id", "")
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "빈 메시지"}), 400

    session = get_session(session_id)
    history = session["history"]
    state = session["state"]
    messages = []  # 이번 턴에 보낼 응답들

    # ── 상담 제안 수락/거절 대기 중 ──
    if state == AppState.AWAITING_CONSENT:
        if is_declining(user_input):
            reply = generate_decline_response(user_input, models)
            messages.append({"type": "text", "content": reply})
            history.append({"user": user_input, "assistant": reply})
            session["state"] = AppState.CHAT
        else:
            # 수락 → 동네 질문
            reply = "그러면 너 사는 동네가 어디야? (예: 강남구, 수원시, 부산 해운대)"
            messages.append({"type": "text", "content": reply})
            history.append({"user": user_input, "assistant": reply})
            session["state"] = AppState.AWAITING_LOCATION

        return jsonify({"messages": messages, "risk_level": 0})

    # ── 동네 입력 대기 중 ──
    if state == AppState.AWAITING_LOCATION:
        if is_declining(user_input):
            reply = generate_decline_response(user_input, models)
            messages.append({"type": "text", "content": reply})
            history.append({"user": user_input, "assistant": reply})
            session["state"] = AppState.CHAT
        else:
            # 1. 여기서 이미 LLM이 추천 멘트를 다 만들어옴 (약 3~4초 소요)
            recommendation = recommend_centers_with_llm(user_input, models)
            
            if recommendation:
                # 2. 여기서 '또' generate_comfort_after_search를 부르지 말고, 
                # 그냥 추천 결과만 보내거나 고정된 응원 멘트를 뒤에 붙여줘.
                # 정 응원하고 싶으면 recommend_centers_with_llm 프롬프트에 "마지막에 응원도 해줘"라고 넣는게 효율적이야!
                
                messages.append({
                    "type": "centers", 
                    "location": user_input, 
                    "centers": recommendation  # LLM이 만든 추천 텍스트
                })
                
                # 추가 LLM 호출 대신 고정 멘트 활용 (속도 향상)
                comfort_fixed = "너를 항상 응원해. 혼자 고민하지 마! 😊"
                messages.append({"type": "text", "content": comfort_fixed})
                
                history.append({"user": user_input, "assistant": f"{recommendation}\n{comfort_fixed}"})
            else:
                reply = f"미안해, {user_input} 근처에서 시설을 못 찾았어. 1388로 전화해보는 건 어때?"
                messages.append({"type": "text", "content": reply})
                history.append({"user": user_input, "assistant": reply})
            
            session["state"] = AppState.CHAT

        return jsonify({"messages": messages, "risk_level": 0})

    # ── 일반 대화 ──
    risk_level = classify_risk(user_input, history, models)

    # 사용자가 직접 도움/상담을 요청하는지 감지
    HELP_REQUEST_KEYWORDS = [
        "도움", "도와줘", "상담 받고", "상담받고", "상담 어디", "상담센터",
        "신고", "어디에 말해", "누구한테 말해", "알려줘", "센터 찾아",
        "쉼터 어디", "지원시설", "연락처", "전화번호",
    ]
    user_requests_help = any(kw in user_input for kw in HELP_REQUEST_KEYWORDS)

    rag_context = None
    if should_search_rag(user_input, risk_level):
        rag_context = search_stats(user_input, models)

    response = generate_response(user_input, history, models, risk_level, rag_context)
    messages.append({"type": "text", "content": response})

    # 위험 감지 또는 직접 도움 요청 → 상담 제안
    should_offer = (risk_level == 2 or user_requests_help) and not session["offered_support"]
    if should_offer:
        offer = generate_support_offer(user_input, history, models)
        messages.append({"type": "text", "content": offer})
        history.append({"user": user_input, "assistant": f"{response}\n{offer}"})
        session["state"] = AppState.AWAITING_CONSENT
        session["offered_support"] = True
    else:
        history.append({"user": user_input, "assistant": response})

    return jsonify({
        "messages": messages,
        "risk_level": risk_level,
        "risk_label": LABEL_NAMES[risk_level],
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """대화 초기화"""
    data = request.json
    session_id = data.get("session_id", "")
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "ok"})


# ============================================================
# 서버 시작
# ============================================================
if __name__ == "__main__":
    models = load_models()
    
    # 0.0.0.0으로 바인딩해야 같은 네트워크의 다른 기기에서 접속 가능
    print("=" * 55)
    print("  🌐 Tiki 웹앱 서버 시작!")
    print("  로컬 접속: http://localhost:5000")
    print("  외부 접속: http://<내 IP>:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)