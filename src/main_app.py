import os
import json
import torch
import chromadb
from enum import Enum
from chromadb.utils import embedding_functions
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# RAG 검색을 트리거하는 상담 관련 키워드
RAG_TRIGGER_KEYWORDS = [
    "힘들", "우울", "불안", "스트레스", "자살", "죽고", "외로",
    "왕따", "따돌림", "폭력", "학대", "가출", "중독", "게임",
    "성적", "시험", "진로", "취업", "학교", "친구", "부모",
    "걱정", "무서", "두려", "슬프", "화나", "짜증", "고민",
    "상담", "도움", "지원", "센터", "쉼터", "통계", "현황",
]


class AppState(Enum):
    CHAT = "chat"
    AWAITING_CONSENT = "awaiting_consent"
    AWAITING_LOCATION = "awaiting_location"


# ──────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────

def load_models():
    print("📥 모델 로딩 중... (최초 실행 시 1~2분 소요)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   디바이스: {device}")

    # 1. EXAONE SFT LoRA
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

    # 2. RoBERTa 위험도 분류
    print("  [2/4] RoBERTa 위험도 분류 모델...")
    risk_tokenizer = AutoTokenizer.from_pretrained(RISK_MODEL_PATH)
    risk_model = AutoModelForSequenceClassification.from_pretrained(RISK_MODEL_PATH)
    risk_model.to(device)
    risk_model.eval()

    # 3. ChromaDB 청소년 통계 RAG
    print("  [3/4] ChromaDB 청소년 통계 DB...")
    ko_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jhgan/ko-sbert-nli"
    )
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    stats_collection = chroma_client.get_collection(
        name="youth_statistics",
        embedding_function=ko_embedding,
    )

    # 4. 위치 데이터 사전 로드
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


# ──────────────────────────────────────────
# 위험도 분류 (키워드 룰 + 모델 하이브리드)
# ──────────────────────────────────────────

# 즉시 '위험' 판정 키워드
DANGER_KEYWORDS = [
    # 자살/자해
    "죽고 싶", "죽을래", "죽어버리", "죽을 거", "죽었으면", "죽고싶",
    "자살", "자해", "손목", "옥상", "뛰어내리", "목매", "약 먹",
    "유서", "마지막", "끝내고 싶", "사라지고 싶", "없어지고 싶",
    "살고 싶지 않", "살기 싫", "태어나지 말", "안 태어났으면",
    # 폭력/학대
    "때려", "때리", "맞아", "맞았", "폭력", "학대", "구타",
    "성폭력", "성추행", "강간", "성폭행", "몰카", "불법촬영",
    "강제로", "억지로", "만져", "만졌", "신체 접촉",
    # 가정 위기
    "가출", "도망", "집 나가", "집 나왔", "쫓겨났",
    "버림받", "방치", "방임", "굶겨", "밥을 안 줘",
    # 공포/위협
    "무서워", "두려워", "겁나", "위협", "협박", "죽이겠",
    "칼", "흉기", "찌르", "피가 나", "피가 났",
    # 따돌림/괴롭힘
    "왕따", "따돌림", "괴롭힘", "집단폭행", "린치",
    "돈 빼앗", "돈 뺏", "셔틀", "빵셔틀",
]

# '주의' 판정 키워드 (모델이 안전으로 예측해도 최소 주의로 올림)
CAUTION_KEYWORDS = [
    # 감정/심리
    "힘들어", "힘든", "우울", "불안", "외로워", "외롭", "슬퍼", "슬프",
    "짜증", "화가 나", "화나", "분노", "답답", "막막", "절망",
    "무기력", "의욕", "아무것도 하기 싫", "귀찮", "지쳐", "지치",
    "눈물", "울었", "울고", "멘붕", "멘탈",
    # 수면/식사/신체
    "잠이 안", "잠을 못", "불면", "악몽", "가위눌",
    "밥을 안", "먹기 싫", "식욕", "토하", "구토",
    "아파", "다쳤", "멍이", "상처",
    # 학교/대인관계
    "학교 가기 싫", "등교 거부", "결석", "자퇴", "전학",
    "친구가 없", "혼자", "외톨이", "놀림", "무시",
    "선생님이", "선생님한테",
    # 가정
    "부모님", "엄마", "아빠", "싸워", "싸움", "이혼", "별거",
    "집에 가기 싫", "집이 싫", "가족이 싫",
    # 기타
    "고민", "걱정", "스트레스", "압박", "부담",
    "중독", "게임 중독", "도박", "음주", "술",
    "진로", "성적", "시험", "수능", "입시",
]

def classify_risk(user_input, history, models):
    """키워드 룰 + 모델 예측 하이브리드 판단"""
    recent_texts = [t["user"] for t in history[-3:]]
    recent_texts.append(user_input)
    combined = " ".join(recent_texts)

    # 1단계: 키워드 룰 (최우선) — 위험 키워드가 있으면 무조건 위험
    if any(kw in combined for kw in DANGER_KEYWORDS):
        return 2

    # 2단계: 모델 예측
    inputs = models["risk_tokenizer"](
        combined,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(models["risk_device"])

    with torch.no_grad():
        logits = models["risk_model"](**inputs).logits
    model_pred = torch.argmax(logits, dim=1).item()

    # 3단계: 키워드 보정 — 모델이 안전이라 해도 주의 키워드가 있으면 올림
    if model_pred == 0 and any(kw in combined for kw in CAUTION_KEYWORDS):
        return 1

    return model_pred


# ──────────────────────────────────────────
# 통계 RAG 검색 (조건부 실행)
# ──────────────────────────────────────────
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

def should_search_rag(user_input, risk_level):
    """RAG 검색이 필요한지 판단"""
    if risk_level >= 1:
        return True
    return any(kw in user_input for kw in RAG_TRIGGER_KEYWORDS)

def search_stats(query, models, n_results=2):
    results = models["stats_collection"].query(
        query_texts=[query],
        n_results=n_results,
    )
    docs      = results["documents"][0]
    distances = results["distances"][0]
    return [doc for doc, dist in zip(docs, distances) if dist < RAG_DISTANCE_THRESHOLD]


# ──────────────────────────────────────────
# 응답 생성
# ──────────────────────────────────────────

def build_prompt(history, user_msg, risk_level, rag_context=None):
    system = SYSTEM_PROMPT

    if rag_context:
        stats_block = "\n".join(f"- {doc}" for doc in rag_context)
        user_msg = f"[참고 통계]\n{stats_block}\n\n{user_msg}"

    if risk_level == 2:
        system += (
            "\n지금 상대방이 많이 힘든 상황임을 인지해. "
            "걱정 어린 말투로 먼저 괜찮은지 확인해줘."
        )
    elif risk_level == 1:
        system += "\n상대방이 조금 힘든 것 같아. 공감해주고 조심스럽게 물어봐줘."

    prompt = f"[|system|]{system}[|endofturn|]\n"
    for turn in history[-3:]:
        prompt += (
            f"[|user|]{turn['user']}\n"
            f"[|assistant|]{turn['assistant']}[|endofturn|]\n"
        )
    prompt += f"[|user|]{user_msg}\n[|assistant|]"
    return prompt


def generate_response(user_msg, history, models, risk_level, rag_context=None):
    prompt = build_prompt(history, user_msg, risk_level, rag_context)
    tokenizer = models["tokenizer"]
    exaone = models["exaone"]

    inputs = tokenizer(prompt, return_tensors="pt").to(models["exaone_device"])

    with torch.no_grad():
        output_ids = exaone.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response = response.split("[|endofturn|]")[0].strip()
    return response if response else "응, 계속 말해줘."


def generate_support_offer(user_msg, history, models):
    """위험 감지 시 지원시설 제안 멘트를 모델이 생성"""
    system = (
        "전문 상담을 받아보는 게 어떨지 자연스럽게 제안해줘. "
        "상담이나 지원시설의 도움을 받는게 어떤지 자연스럽게 제안해줘"
        "동네를 알려주면 근처 지원시설을 찾아줄 수 있다고 말해줘. "
        "반말로, 짧고 예의바르게, 부담 안 주는 톤으로."
    )
    prompt = f"[|system|]{system}[|endofturn|]\n"
    for turn in history[-2:]:
        prompt += (
            f"[|user|]{turn['user']}\n"
            f"[|assistant|]{turn['assistant']}[|endofturn|]\n"
        )
    prompt += f"[|user|]{user_msg}\n[|assistant|]"

    tokenizer = models["tokenizer"]
    inputs = tokenizer(prompt, return_tensors="pt").to(models["exaone_device"])

    with torch.no_grad():
        output_ids = models["exaone"].generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    offer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    offer = offer.split("[|endofturn|]")[0].strip()
    return offer if offer else "혹시 전문 상담 한번 받아볼 생각 있어? 동네 알려주면 근처 센터 찾아줄게."


def generate_comfort_after_search(models):
    """시설 안내 후 응원 멘트를 모델이 생성"""
    system = (
        "너는 청소년 또래 친구야. "
        "방금 상대방에게 상담 지원시설 정보를 알려줬어. "
        "혼자 감당하지 않아도 된다고 따뜻하게 응원해줘. "
        "짧게, 예의바르게 진심 어린 톤으로."
    )
    prompt = f"[|system|]{system}[|endofturn|]\n[|user|]응원해줘\n[|assistant|]"

    tokenizer = models["tokenizer"]
    inputs = tokenizer(prompt, return_tensors="pt").to(models["exaone_device"])

    with torch.no_grad():
        output_ids = models["exaone"].generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    comfort = tokenizer.decode(new_tokens, skip_special_tokens=True)
    comfort = comfort.split("[|endofturn|]")[0].strip()
    return comfort if comfort else "혼자 감당 안 해도 돼. 나도 응원할게!"


def generate_decline_response(user_msg, models):
    """시설 안내 거절 시 응답을 모델이 생성"""
    system = (
        "너는 청소년 또래 친구야. "
        "상대방에게 상담 지원시설을 제안했는데 거절했어. "
        "괜찮다고 하면서, 언제든 필요하면 말하라고 해줘. "
        "청소년 전화 1388은 24시간 무료라는 것도 자연스럽게 알려줘. "
        "반말로, 짧고 예의바르게, 부담 안 주는 톤으로."
    )
    prompt = f"[|system|]{system}[|endofturn|]\n[|user|]{user_msg}\n[|assistant|]"

    tokenizer = models["tokenizer"]
    inputs = tokenizer(prompt, return_tensors="pt").to(models["exaone_device"])

    with torch.no_grad():
        output_ids = models["exaone"].generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    reply = reply.split("[|endofturn|]")[0].strip()
    return reply if reply else "알겠어! 언제든 필요하면 말해줘. 1388은 24시간이야!"


# ──────────────────────────────────────────
# 위치 기반 지원시설 검색
# ──────────────────────────────────────────
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


# ──────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🤝 또래 상담 챗봇  (Tiki-Tak-A)")
    print("=" * 55)

    models = load_models()
    history: list[dict] = []
    state = AppState.CHAT
    offered_support = False

    # 첫 인사도 모델이 생성
    greeting_prompt = (
        f"[|system|]{SYSTEM_PROMPT}\n처음 만난 상대방에게 편하게 인사해줘. "
        f"짧고 친근하게.[|endofturn|]\n[|user|]안녕[|endofturn|]\n[|assistant|]"
    )
    tokenizer = models["tokenizer"]
    inputs = tokenizer(greeting_prompt, return_tensors="pt").to(models["exaone_device"])
    with torch.no_grad():
        output_ids = models["exaone"].generate(
            **inputs, max_new_tokens=50, do_sample=True,
            temperature=0.8, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    greeting = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    greeting = greeting.split("[|endofturn|]")[0].strip()
    if not greeting:
        greeting = "안녕! 편하게 얘기해 😊"

    print(f"챗봇: {greeting}")
    print("      (종료: q)\n")

    while True:
        try:
            user_input = input("너: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n챗봇: 얘기해줘서 고마워!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["q", "quit", "종료"]:
            print("챗봇: 얘기해줘서 고마워!")
            break

        # ── 상담 제안 수락/거절 대기 중 ──────────────────
        if state == AppState.AWAITING_CONSENT:
            if is_declining(user_input):
                reply = generate_decline_response(user_input, models)
                print(f"챗봇: {reply}\n")
                history.append({"user": user_input, "assistant": reply})
                state = AppState.CHAT
                continue
            else:
                # 수락 → 동네 질문
                reply = "그러면 너 사는 동네가 어디야? (예: 강남구, 수원시, 부산 해운대)"
                print(f"챗봇: {reply}\n")
                history.append({"user": user_input, "assistant": reply})
                state = AppState.AWAITING_LOCATION
                continue

        # ── 동네 입력 대기 중 ─────────────────────────
        if state == AppState.AWAITING_LOCATION:
            if is_declining(user_input):
                reply = generate_decline_response(user_input, models)
                print(f"챗봇: {reply}\n")
                history.append({"user": user_input, "assistant": reply})
                state = AppState.CHAT
                continue

            print("챗봇: 근처에 도움이 될 만한 곳이 있는지 찾아볼게... 🔍", end="\r")
            
            # LLM이 데이터를 보고 직접 답변 생성
            recommendation = recommend_centers_with_llm(user_input, models)
            
            if recommendation:
                # 추가 LLM 호출 대신 고정 멘트 활용 (속도 향상)
                comfort_fixed = "너를 항상 응원해. 혼자 고민하지 마! 😊"
                print(f"챗봇: {recommendation}\n{comfort_fixed}\n")
                history.append({"user": user_input, "assistant": f"{recommendation}\n{comfort_fixed}"})
            
            else:
                reply = (
                    f"미안해, {user_input} 근처에서 적당한 시설을 못 찾았어. ㅠㅠ "
                    "대신 24시간 열려있는 1388 청소년 전화로 바로 연락해보는 건 어때?"
                )
                print(f"챗봇: {reply}\n")
            
            history.append({"user": user_input, "assistant": reply})
            state = AppState.CHAT
            continue

        # ── 위험도 분류 (누적 판단) ────────────────────
        risk_level = classify_risk(user_input, history, models)

        # 사용자가 직접 도움/상담을 요청하는지 감지
        HELP_REQUEST_KEYWORDS = [
            "도움", "도와줘", "상담 받고", "상담받고", "상담 어디", "상담센터",
            "신고", "어디에 말해", "누구한테 말해", "알려줘", "센터 찾아",
            "쉼터 어디", "지원시설", "연락처", "전화번호",
        ]
        user_requests_help = any(kw in user_input for kw in HELP_REQUEST_KEYWORDS)

        # ── 통계 RAG 검색 (조건부 실행) ────────────────
        rag_context = None
        if should_search_rag(user_input, risk_level):
            rag_context = search_stats(user_input, models)

        # ── 응답 생성 ──────────────────────────────────
        print("챗봇: ...", end="\r")
        response = generate_response(user_input, history, models, risk_level, rag_context)
        print(f"챗봇: {response}")

        # ── 위험 감지 또는 직접 도움 요청 → 상담 제안 ────
        should_offer = (risk_level == 2 or user_requests_help) and not offered_support
        if should_offer:
            offer = generate_support_offer(user_input, history, models)
            print(f"챗봇: {offer}")
            history.append({"user": user_input, "assistant": f"{response}\n{offer}"})
            state = AppState.AWAITING_CONSENT
            offered_support = True
        else:
            history.append({"user": user_input, "assistant": response})

        print()


if __name__ == "__main__":
    main()