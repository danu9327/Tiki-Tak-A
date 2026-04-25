import os
import json
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COUNSELING_SFT = os.path.join(BASE_DIR, "data/sft/sft_counseling_peer_tone.jsonl")
SNS_SFT = os.path.join(BASE_DIR, "data/sft/sft_from_sns_data.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/sft/sft_total.jsonl")

SEED = 42
SNS_CAP = 30000
random.seed(SEED)

def load_jsonl(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

def main():
    # 1. 데이터 로드
    counseling = load_jsonl(COUNSELING_SFT)
    sns = load_jsonl(SNS_SFT)
    
    print(f"📊 상담 SFT: {len(counseling)}건")
    print(f"📊 SNS SFT 원본: {len(sns)}건")

    # SNS 데이터 캡 적용 (1:1 비율)
    if len(sns) > SNS_CAP:
        sns = random.sample(sns, SNS_CAP)
        print(f"📊 SNS SFT 샘플링 후: {len(sns)}건 ({SNS_CAP}건으로 제한)")
    
    # 2. 출처 태그 추가 (나중에 분석/디버깅용)
    for entry in counseling:
        entry["source"] = "counseling"
    for entry in sns:
        entry["source"] = "sns"
    
    # 3. 통합 + 셔플
    total = counseling + sns
    random.shuffle(total)
    
    # 4. 빈 텍스트 필터링
    filtered = []
    for entry in total:
        inst = entry.get("instruction", "").strip()
        out = entry.get("output", "").strip()
        if inst and out:
            filtered.append(entry)
    
    removed = len(total) - len(filtered)
    
    # 5. 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in filtered:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 통합 완료!")
    print(f"📊 총 {len(filtered)}건 (빈 텍스트 제거: {removed}건)")
    print(f"📊 비율 — 상담: {len(counseling)}건 / SNS: {len(sns)}건")
    print(f"💾 저장: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()