# 고품질 상담 데이터 30,000건은 너무 딱딱한 말투
# 이 Tiki Tak-a 챗봇의 의도인 "또래 페르소나"를 위한 말투톤 변환 필요
# EXAONE 로컬 모델을 활용하여 변환 ->(sft_counseling_peer_tone.jsonl)

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = "/home/user/Tiki-Tak-A"
MODEL_PATH = os.path.join(BASE_DIR, "models/base/EXAONE")
INPUT_PATH = os.path.join(BASE_DIR, "data/sft/sft_counseling_filtered.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/sft/sft_counseling_peer_tone.jsonl")
PROGRESS_PATH = os.path.join(BASE_DIR, "data/sft/convert_progress.json")

# ============================================================
# 생성 설정
# ============================================================
MAX_NEW_TOKENS = 256
BATCH_LOG_INTERVAL = 100   # N건마다 진행 상황 저장

# ============================================================
# 변환 프롬프트
# ============================================================
CONVERT_PROMPT = """다음은 상담사가 내담자에게 한 말이야. 이걸 청소년 또래 친구가 말하는 것처럼 자연스럽게 바꿔줘.

규칙:
1. 반말로 바꿔 (존댓말 금지)
2. 공감의 내용은 유지해
3. 조언이 있으면 부드럽게 유지해
4. 너무 길게 늘리지 마
5. "ㅠㅠ", "ㅋㅋ", "ㄹㅇ", "진짜" 같은 표현을 자연스럽게 써도 돼
6. 변환된 문장만 출력해. 설명이나 부연은 하지 마.

원문: {original}

변환:"""

def load_model():
    """EXAONE 베이스 모델 로드"""
    print("📥 EXAONE 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()
    
    print(f"✅ 모델 로드 완료!")
    return model, tokenizer

def convert_tone(model, tokenizer, original_text):
    """상담사 톤을 또래 톤으로 변환"""
    prompt = CONVERT_PROMPT.format(original=original_text)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 입력 부분 제거하고 생성된 부분만 추출
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # 후처리: 첫 줄만 사용 (모델이 여러 줄 생성할 수 있음)
    result = result.split("\n")[0].strip()
    
    # 따옴표 제거
    result = result.strip('"').strip("'").strip()
    
    return result

def load_progress():
    """이전 진행 상황 로드 (중단 후 재개용)"""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"completed": 0, "results": []}

def save_progress(progress):
    """진행 상황 저장"""
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

def main():
    # 1. 데이터 로드
    entries = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    
    print(f"📂 필터링된 데이터: {len(entries)}건")
    
    # 2. 이전 진행 상황 확인
    progress = load_progress()
    start_idx = progress["completed"]
    results = progress["results"]
    
    if start_idx > 0:
        print(f"🔄 이전 진행분 {start_idx}건 발견. 이어서 변환합니다.")
    
    # 3. 모델 로드
    model, tokenizer = load_model()
    
    # 4. 변환 시작
    remaining = entries[start_idx:]
    print(f"\n🔄 말투 변환 시작! ({len(remaining)}건 남음)")
    
    start_time = time.time()
    fail_count = 0
    
    for i, entry in enumerate(tqdm(remaining, desc="Converting")):
        original_output = entry["output"]
        
        try:
            converted = convert_tone(model, tokenizer, original_output)
            
            # 변환 실패 체크 (너무 짧거나 원문과 동일)
            if len(converted) < 5 or converted == original_output:
                converted = original_output  # 실패 시 원문 유지
                fail_count += 1
            
            results.append({
                "instruction": entry["instruction"],
                "output": converted,
                "original_output": original_output  # 원문 보존 (디버깅용)
            })
            
        except Exception as e:
            # 에러 시 원문 유지
            results.append({
                "instruction": entry["instruction"],
                "output": original_output,
                "original_output": original_output
            })
            fail_count += 1
        
        # 주기적 저장 (중단 대비)
        current_idx = start_idx + i + 1
        if current_idx % BATCH_LOG_INTERVAL == 0:
            progress = {"completed": current_idx, "results": results}
            save_progress(progress)
            
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            remaining_time = (len(remaining) - i - 1) / speed if speed > 0 else 0
            
            print(f"\n💾 {current_idx}건 저장 | 속도: {speed:.1f}건/초 | 남은 시간: {remaining_time/3600:.1f}시간")
    
    # 5. 최종 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in results:
            # original_output은 저장하지 않음 (디버깅 끝나면)
            save_entry = {
                "instruction": entry["instruction"],
                "output": entry["output"]
            }
            f.write(json.dumps(save_entry, ensure_ascii=False) + "\n")
    
    # 진행 파일 정리
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
    
    total_time = time.time() - start_time
    print(f"\n✅ 말투 변환 완료!")
    print(f"📊 총 {len(results)}건 변환 (실패/원문유지: {fail_count}건)")
    print(f"⏱️  소요 시간: {total_time/3600:.1f}시간")
    print(f"💾 저장: {OUTPUT_PATH}")
    
    # 변환 샘플 출력
    print(f"\n📋 변환 샘플 5개:")
    for entry in results[:5]:
        print(f"   [원문] {entry['original_output'][:60]}...")
        print(f"   [변환] {entry['output'][:60]}...")
        print()

if __name__ == "__main__":
    main()