# 한국어 sns데이터 json이 176,605개 너무너무너무 많음
# 토픽별(건강및식음료,경제및사회,과학기술,문화생활및여가,미용과패션,스포츠및e스포츠,여행관광및명소,정치,콘텐츠소비)
# 균등하게 11,000건만 선택해서 통합하는 코드
# 176,605건 -> 11,000건(sft_from_sns_data.jsonl)

import os
import json
import random
from collections import defaultdict
from tqdm import tqdm

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_DIR = os.path.join(BASE_DIR, "data/jsons")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/sft/sft_from_sns_data.jsonl")
SAMPLE_SIZE = 11000
SEED = 42

random.seed(SEED)

def is_two_speaker(data):
    """실제 발화(utterances)에서 등장한 화자가 정확히 2명인지 확인"""
    utterances = data.get("utterances", [])
    speakers = set(utt.get("speaker", "") for utt in utterances)
    return len(speakers) == 2

def convert_to_sft(data):
    """2인 대화를 instruction-output 쌍으로 변환
    
    speakerA → user (청소년)
    speakerB → assistant (또래 친구 = 챗봇 페르소나)
    
    연속된 같은 화자 발화는 하나로 합침
    """
    utterances = data.get("utterances", [])
    if not utterances:
        return []
    
    # 연속 발화 병합 (같은 화자가 연속으로 말한 경우)
    merged = []
    for utt in utterances:
        speaker = utt["speaker"]
        text = utt["text"].strip()
        if not text:
            continue
        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] += " " + text
        else:
            merged.append({"speaker": speaker, "text": text})
    
    # speakerA 발화 → instruction, speakerB 응답 → output
    sft_pairs = []
    for i in range(len(merged) - 1):
        if merged[i]["speaker"] == "speakerA" and merged[i + 1]["speaker"] == "speakerB":
            sft_pairs.append({
                "instruction": merged[i]["text"],
                "output": merged[i + 1]["text"]
            })
    
    return sft_pairs

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 1단계: 2인 대화만 수집 (토픽별 분류)
    print("🔍 2인 대화 필터링 중...")
    topic_files = defaultdict(list)  # {토픽: [파일경로, ...]}
    total_files = 0
    skipped = 0
    
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".json")]
    
    for filename in tqdm(all_files, desc="Scanning"):
        file_path = os.path.join(SOURCE_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            total_files += 1
            
            if not is_two_speaker(data):
                skipped += 1
                continue
            
            topic = data.get("info", {}).get("topic", "기타")
            topic_files[topic].append(file_path)
            
        except Exception:
            continue
    
    print(f"\n📊 전체 {total_files}개 중 2인 대화: {sum(len(v) for v in topic_files.values())}개 (3인 이상 제외: {skipped}개)")
    print(f"📊 토픽별 분포:")
    for topic, files in sorted(topic_files.items()):
        print(f"   {topic}: {len(files)}개")
    
    # 2단계: 토픽별 균등 샘플링
    print(f"\n🎲 토픽별 균등 샘플링 ({SAMPLE_SIZE}건 목표)...")
    num_topics = len(topic_files)
    per_topic = SAMPLE_SIZE // num_topics
    remainder = SAMPLE_SIZE % num_topics
    
    sampled_files = []
    for i, (topic, files) in enumerate(sorted(topic_files.items())):
        # 나머지를 앞쪽 토픽에 1개씩 추가 배분
        n = per_topic + (1 if i < remainder else 0)
        n = min(n, len(files))  # 부족하면 있는 만큼만
        sampled = random.sample(files, n)
        sampled_files.extend(sampled)
        print(f"   {topic}: {n}건 샘플링")
    
    # 3단계: SFT 포맷 변환
    print(f"\n🔄 SFT 포맷 변환 중...")
    sft_entries = []
    
    for file_path in tqdm(sampled_files, desc="Converting"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pairs = convert_to_sft(data)
            sft_entries.extend(pairs)
        except Exception:
            continue
    
    # 4단계: 저장
    random.shuffle(sft_entries)  # 토픽 편향 방지를 위해 셔플
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in sft_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 완료!")
    print(f"📊 샘플링된 대화: {len(sampled_files)}개")
    print(f"📊 생성된 SFT 쌍: {len(sft_entries)}건")
    print(f"💾 저장 위치: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()