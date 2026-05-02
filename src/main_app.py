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
    "너의 이름은 Tiki야. "
    "너는 청소년 또래 친구처럼 편하게 대화하면서도, "
    "상대방의 고민을 진심으로 들어주고 도움을 줄 수 있는 상담 챗봇이야. "
    "반말을 사용하고, 아래 원칙을 꼭 지켜줘.\n"
    "1. 공감 먼저: 상대방의 감정을 먼저 알아주고 인정해줘. '그랬구나', '힘들었겠다' 같은 말을 먼저 해줘.\n"
    "2. 절대 하면 안 되는 것: 상대방의 외모, 성격, 행동을 탓하거나 문제의 원인을 상대방에게 돌리지 마. "
    "'네가 뭘 잘못한 거 아니야?', '네 행동에 문제가 있는 건 아닌지' 같은 말은 절대 하지 마.\n"
    "3. 편 들어주기: 상대방이 괴롭힘이나 폭력을 당했다고 하면, 무조건 상대방 편에 서줘. "
    "'그건 네 잘못이 아니야', '그런 행동은 잘못된 거야' 같은 말을 해줘.\n"
    "4. 조언은 부드럽게: 해결책을 제시할 때는 강요하지 말고 '이런 방법도 있어' 식으로 제안해줘.\n"
    "5. 짧게 답해: 한 번에 너무 길게 말하지 말고, 2~3문장 정도로 답해줘."
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
    "성범죄", "성피해", "성적 수치", "강제 추행",
    "강제로", "억지로", "만져", "만졌", "신체 접촉",
    "당했", "당한 적",
    # 가정 위기
    "가출", "도망", "집 나가", "집 나왔", "집 나온", "쫓겨났",
    "밖에서 자", "갈 데가 없", "갈 곳이 없", "노숙",
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
        "너는 청소년 또래 친구야. "
        "상대방이 많이 힘든 상황인 것 같아. "
        "전문 상담을 받아보는 게 어떨지 자연스럽게 제안해줘. "
        "동네를 알려주면 근처 지원시설을 찾아줄 수 있다고 말해줘. "
        "반말로, 짧게, 부담 안 주는 톤으로."
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
        "반말로, 짧게, 진심 어린 톤으로."
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
        "반말로, 짧게, 부담 안 주는 톤으로."
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

SUPPORT_SOURCE_KEYWORDS = ["상담", "청소년", "쉼터", "복지", "지원", "보호"]

LOCATION_STOPWORDS = ["쪽", "근처", "근방", "동네", "살아", "살고", "있어", "인데", "이야", "나", "나는", "나도", "저는", "난"]

LOCATION_PARTICLES = ["에서", "으로", "에도", "에는", "에", "은", "는", "이", "가", "을", "를", "도", "로", "의"]

def strip_particles(word):
    for p in LOCATION_PARTICLES:
        if word.endswith(p) and len(word) > len(p) + 1:
            return word[:-len(p)]
    return word

def extract_location_keywords(text):
    cleaned = text
    for sw in LOCATION_STOPWORDS:
        cleaned = cleaned.replace(sw, " ")
    
    raw_keywords = [k.strip() for k in cleaned.split() if len(k.strip()) >= 2]
    keywords = [strip_particles(kw) for kw in raw_keywords]
    keywords = [kw for kw in keywords if len(kw) >= 2]
    
    specific = []
    general = []
    
    for kw in keywords:
        if any(kw.endswith(suffix) for suffix in ["구", "군", "시", "동", "읍", "면"]):
            specific.append(kw)
        elif kw in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                     "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]:
            general.append(kw)
        else:
            specific.append(kw)
    
    return specific, general


def search_centers(location_text, models, max_results=5):
    """지역 키워드로 지원시설 검색. 구체적 지역(구/군) 우선 매칭."""
    specific, general = extract_location_keywords(location_text)
    
    if not specific and not general:
        return []
    
    all_centers = models["location_data"]
    
    SUPPORT_SOURCES = [
        "청소년상담복지센터",
        "청소년쉼터",
        "청소년자립지원관",
        "청소년디딤센터",
        "청소년성문화센터",
        "청소년복지시설관심지점정보",
        "청소년지원시설관심지점",
        "여성·가족·청소년·권익시설정보",
    ]
    
    EXCLUDE_NAME_KEYWORDS = [
        "학원", "교습소", "어린이집", "유치원", "학교", "태권도",
        "피아노", "미술", "음악", "영어", "수학", "보습",
        "입시", "코딩", "체육관", "수영장",
    ]
    
    def is_real_support(center):
        source = center.get("source", "")
        name = center.get("name", "")
        if not any(s in source for s in SUPPORT_SOURCES):
            return False
        if any(ex in name for ex in EXCLUDE_NAME_KEYWORDS):
            return False
        return True
    
    tier1, tier2 = [], []
    
    for center in all_centers:
        if not is_real_support(center):
            continue
        
        addr = center.get("address", "")
        region = center.get("region", "")
        name = center.get("name", "")
        searchable = f"{addr} {region} {name}"
        
        if specific and any(kw in searchable for kw in specific):
            tier1.append(center)
        elif general and any(kw in searchable for kw in general):
            tier2.append(center)
    
    results = tier1 if tier1 else tier2
    return results[:max_results]


def is_declining(text):
    declines = ["괜찮아", "됐어", "안 해", "안해", "필요없", "나중에", "싫어", "안 할"]
    if text.strip() in ["아니", "아니요", "아니야", "아닌데"]:
        return True
    return any(d in text for d in declines)


def is_accepting(text):
    accepts = [
        "응", "어", "ㅇㅇ", "ㅇ", "좋아", "해줘", "알려줘", "부탁",
        "해볼래", "해볼게", "받고 싶", "받고싶", "그래", "원해",
        "찾아줘", "네", "당연", "물론", "해봐", "해 줘",
        "갈게", "가볼게", "가고 싶", "가볼래",
    ]
    if is_declining(text):
        return False
    return any(a in text for a in accepts)


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

            if is_accepting(user_input):
                reply = "그러면 너 사는 동네가 어디야? (예: 강남구, 수원시, 부산 해운대)"
                print(f"챗봇: {reply}\n")
                history.append({"user": user_input, "assistant": reply})
                state = AppState.AWAITING_LOCATION
                continue

            # 수락도 거절도 아닌 일반 대화 → 정상 응답 + 상태 유지
            risk_level = classify_risk(user_input, history, models)
            rag_context = None
            if should_search_rag(user_input, risk_level):
                rag_context = search_stats(user_input, models)
            response = generate_response(user_input, history, models, risk_level, rag_context)
            print(f"챗봇: {response}")
            print(f"챗봇: 그리고 아까 얘기인데, 근처 상담센터 찾아볼까? 원하면 말해줘!\n")
            history.append({"user": user_input, "assistant": response})
            continue

        # ── 동네 입력 대기 중 ─────────────────────────
        if state == AppState.AWAITING_LOCATION:
            if is_declining(user_input):
                reply = generate_decline_response(user_input, models)
                print(f"챗봇: {reply}\n")
                history.append({"user": user_input, "assistant": reply})
                state = AppState.CHAT
                continue

            # 지역명인지 판단: 시설 검색 결과가 있으면 지역명
            centers = search_centers(user_input, models)
            if centers:
                print(f"\n챗봇: {user_input} 근처 지원시설 찾았어 👇\n")
                for i, c in enumerate(centers, 1):
                    print(f"  [{i}] {c.get('name', '이름 없음')}")
                    print(f"       📍 {c.get('address', '주소 없음')}")
                    print(f"       📞 {c.get('phone', '1388')}\n")
                comfort = generate_comfort_after_search(models)
                print(f"챗봇: {comfort}\n")
                history.append({"user": user_input, "assistant": comfort})
                state = AppState.CHAT
                continue

            # 시설 검색 결과 없음 → 일반 대화로 응답 + 동네 다시 질문
            risk_level = classify_risk(user_input, history, models)
            rag_context = None
            if should_search_rag(user_input, risk_level):
                rag_context = search_stats(user_input, models)
            response = generate_response(user_input, history, models, risk_level, rag_context)
            print(f"챗봇: {response}")
            print(f"챗봇: 참, 아까 동네 알려주면 근처 지원시설 찾아줄 수 있어! 사는 곳이 어디야?\n")
            history.append({"user": user_input, "assistant": response})
            continue

        # ── 위험도 분류 (누적 판단) ────────────────────
        risk_level = classify_risk(user_input, history, models)

        # 사용자가 직접 도움/상담을 요청하는지 감지
        HELP_REQUEST_KEYWORDS = [
            "도움", "도와줘", "도와줄", "도와주",
            "상담 받고", "상담받고", "상담 어디", "상담센터", "상담하고",
            "신고", "어디에 말해", "누구한테 말해", "누구한테 말할",
            "알려줘", "센터 찾아", "찾아줘",
            "쉼터 어디", "쉼터 있어", "쉼터 갈", "쉼터 같은",
            "지원시설", "연락처", "전화번호",
            "어떻게 해야", "어떡해", "어떻게 해",
            "어디 가야", "어디로 가", "갈 수 있는 데",
            "도와줄 수 있", "도움받", "도움 받",
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