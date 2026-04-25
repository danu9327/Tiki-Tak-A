import os
import json
from tqdm import tqdm

# 1. 경로 설정 (리눅스 환경 최적화)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RISK_ROOT = os.path.join(BASE_DIR, "data", "risk")
RISK_OUTPUT = os.path.join(RISK_ROOT, "total_risk_data.jsonl")
SFT_OUTPUT = os.path.join(BASE_DIR, "data", "sft", "sft_from_risk_data.jsonl")

# 분류 레이블 정의 (3단계)
LABEL_MAP = {0: "안전", 1: "주의", 2: "위험"}

def score_to_class(score):
    """0~100 정규화 점수를 3단계 분류 레이블로 변환"""
    if score < 30:
        return 0   # 안전
    elif score < 60:
        return 1   # 주의
    else:
        return 2   # 위험

def merge_logic():
    os.makedirs(os.path.dirname(SFT_OUTPUT), exist_ok=True)
    risk_entries = []
    sft_entries = []
    
    # os.walk로 모든 하위 폴더를 샅샅이 뒤집니다.
    all_files = []
    for root, dirs, files in os.walk(RISK_ROOT):
        for f in files:
            if f.endswith(".json") and f != "total_risk_data.json":
                all_files.append(os.path.join(root, f))

    print(f"🔄 총 {len(all_files)}개의 JSON 파일을 발견했습니다. 분석을 시작합니다...")

    for file_path in tqdm(all_files, desc="Merging"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # --- 포맷 판별 및 추출 ---
            
            # [CASE 1] Youth 데이터 (info와 list가 있는 경우)
            if isinstance(data, dict) and "info" in data and "list" in data:
                score = float(data["info"].get("합계점수", 0))
                user_texts = []
                temp_dialogue = []
                for cat in data.get("list", []):
                    for item in cat.get("list", []):
                        for audio in item.get("audio", []):
                            txt, role = audio.get("text", ""), audio.get("type")
                            if role == "A": user_texts.append(txt)
                            temp_dialogue.append({"role": role, "text": txt})
                
                if user_texts:
                    # source 표시: 나중에 정규화할 때 구분용
                    risk_entries.append({
                        "text": " ".join(user_texts),
                        "raw_score": score,
                        "source": "youth"
                    })
                for i in range(len(temp_dialogue)-1):
                    if temp_dialogue[i]["role"] == "A" and temp_dialogue[i+1]["role"] == "Q":
                        sft_entries.append({"instruction": temp_dialogue[i]["text"], "output": temp_dialogue[i+1]["text"]})

            # [CASE 2] Psych 데이터 (paragraph가 있는 경우) — 이미 0~100 스케일
            elif isinstance(data, dict) and "paragraph" in data:
                risk_factors = ["depressive_mood", "worthlessness", "guilt", "suicidal", "anhedonia", "sleep_disturbance", "fatigue"]
                client_texts = []
                max_scaled_score = 0
                
                paragraphs = data.get("paragraph", [])
                for i, p in enumerate(paragraphs):
                    if p.get("paragraph_speaker") == "내담자":
                        txt = p.get("paragraph_text", "").strip()
                        client_texts.append(txt)
                        total = sum(p.get(f, 0) for f in risk_factors)
                        scaled = (total / (len(risk_factors) * 2.0)) * 100.0
                        max_scaled_score = max(max_scaled_score, scaled)
                        
                        if i < len(paragraphs)-1 and paragraphs[i+1].get("paragraph_speaker") == "상담사":
                            sft_entries.append({"instruction": txt, "output": paragraphs[i+1].get("paragraph_text", "")})
                
                if client_texts:
                    risk_entries.append({
                        "text": " ".join(client_texts),
                        "raw_score": max_scaled_score,
                        "source": "psych"
                    })

        except Exception:
            continue

    # ============================================================
    # 레이블 스케일 통합: Youth 점수를 0~100으로 정규화
    # ============================================================
    youth_scores = [e["raw_score"] for e in risk_entries if e["source"] == "youth"]
    
    if youth_scores:
        y_min = min(youth_scores)
        y_max = max(youth_scores)
        print(f"\n📊 Youth 원본 점수 범위: {y_min} ~ {y_max}")
        
        for e in risk_entries:
            if e["source"] == "youth":
                if y_max > y_min:
                    e["raw_score"] = (e["raw_score"] - y_min) / (y_max - y_min) * 100.0
                else:
                    e["raw_score"] = 0.0

    # Psych 점수 범위도 출력 (확인용)
    psych_scores = [e["raw_score"] for e in risk_entries if e["source"] == "psych"]
    if psych_scores:
        print(f"📊 Psych 점수 범위: {min(psych_scores):.1f} ~ {max(psych_scores):.1f} (이미 0~100)")

    # ============================================================
    # 3단계 분류 레이블 변환
    # ============================================================
    label_counts = {0: 0, 1: 0, 2: 0}
    
    for e in risk_entries:
        cls_label = score_to_class(e["raw_score"])
        e["label"] = cls_label
        label_counts[cls_label] += 1
        # 저장 시 불필요한 필드 제거
        del e["raw_score"]
        del e["source"]

    # 분포 출력
    print(f"\n📊 분류 레이블 분포:")
    for cls_id, count in label_counts.items():
        print(f"   {LABEL_MAP[cls_id]}({cls_id}): {count}건")

    # --- 저장 ---
    with open(RISK_OUTPUT, "w", encoding="utf-8") as f:
        for e in risk_entries: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(SFT_OUTPUT, "w", encoding="utf-8") as f:
        for e in sft_entries: f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n✨ 병합 완료!")
    print(f"📊 Risk Data: {len(risk_entries)}건")
    print(f"📊 SFT Data: {len(sft_entries)}건")

if __name__ == "__main__":
    merge_logic()