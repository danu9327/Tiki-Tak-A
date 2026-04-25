# 심리 상담 데이터에서 추출한 json이 362,107건 너무너무너무 많음
# read_me에 작성한 기준으로 고품질 상담 데이터 30,000건만 추출
# 362,107건(sft_from_risk_data.jsonl) -> 30,000건(sft_counceling_filtered.jsonl)


import os
import json
import re
import random
from tqdm import tqdm

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = "/home/user/Tiki-Tak-A"
INPUT_PATH = os.path.join(BASE_DIR, "data/sft/sft_from_risk_data.jsonl")
FILTERED_PATH = os.path.join(BASE_DIR, "data/sft/sft_counseling_filtered.jsonl")

# ============================================================
# 필터링 기준 (강화)
# ============================================================
TARGET_COUNT = 30000        # 최종 목표 건수
MIN_INSTRUCTION_LEN = 15   # 내담자 발화 최소 길이 (10→15)
MIN_OUTPUT_LEN = 30         # 상담사 응답 최소 길이 (15→30)
MAX_OUTPUT_LEN = 500        # 너무 긴 응답도 제거 (변환 품질 저하)
SEED = 42

random.seed(SEED)

# 척도/설문/행정 패턴 (output에서 제거)
SCALE_PATTERNS = [
    r"0부터\s*\d+까지",
    r"\d+점\s*만점",
    r"전혀\s*(없음|아니다)",
    r"거의\s*매일",
    r"며칠\s*동안",
    r"사이에\s*말씀",
    r"체크.*해\s*볼",
    r"척도",
    r"설문",
    r"검사를?\s*(시작|진행)",
    r"다음\s*(문항|질문|항목)",
    r"점수.*매기",
    r"몇\s*점",
    r"평가.*해\s*(보|줘|주)",
]

# 비상담 패턴 (output에서 제거)
NON_COUNSELING_PATTERNS = [
    r"다음\s*시간",
    r"오늘\s*상담.*마무리",
    r"다음\s*주.*뵐",
    r"예약",
    r"접수",
    r"동의서",
    r"녹음",
    r"비밀\s*보장",
]

# 단순 되묻기/추임새 패턴
ECHO_PATTERNS = [
    r"^.{0,10}\?$",                     # 10자 이하 + 물음표만
    r"^(네|응|아|음|그래요|맞아요)\s*[\.\?]*$",
    r"^(그렇군요|그랬구나|아 그래요)\s*[\.\?]*$",
]

# 공감/조언 키워드 (이게 있으면 가산점 → 우선 선별)
EMPATHY_KEYWORDS = [
    "힘들", "괜찮", "걱정", "마음", "감정", "느낌", "이해",
    "공감", "응원", "함께", "도움", "노력", "용기", "잘했",
    "스트레스", "불안", "우울", "외로", "슬프", "화가",
    "고민", "상처", "아프", "지치", "무섭", "두렵",
]

def is_scale_or_admin(text):
    for pattern in SCALE_PATTERNS:
        if re.search(pattern, text):
            return True
    for pattern in NON_COUNSELING_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

def is_echo_only(text):
    for pattern in ECHO_PATTERNS:
        if re.match(pattern, text.strip()):
            return True
    return False

def has_empathy(text):
    """공감/조언 키워드 포함 여부"""
    return any(kw in text for kw in EMPATHY_KEYWORDS)

def is_quality_response(instruction, output):
    """양질의 상담 응답인지 종합 판단"""
    inst = str(instruction).strip() if instruction is not None else ""
    out = str(output).strip() if output is not None else ""
    
    # 길이 기준
    if len(inst) < MIN_INSTRUCTION_LEN:
        return False, "instruction_too_short", 0
    if len(out) < MIN_OUTPUT_LEN:
        return False, "output_too_short", 0
    if len(out) > MAX_OUTPUT_LEN:
        return False, "output_too_long", 0
    
    # 척도/행정/비상담 발화
    if is_scale_or_admin(out):
        return False, "scale_or_admin", 0
    
    # 단순 되묻기
    if is_echo_only(out):
        return False, "echo_only", 0
    
    # output이 물음표로만 끝나고 30자 미만이면 단순 질문
    if out.endswith("?") and len(out) < 30:
        return False, "short_question_only", 0
    
    # 품질 점수 계산 (공감 키워드 기반)
    score = 0
    if has_empathy(out):
        score += 2  # 공감 키워드가 output에 있으면 우선
    if has_empathy(inst):
        score += 1  # instruction에 감정 표현이 있으면 가산
    if len(out) >= 50:
        score += 1  # 충분한 길이의 응답
    
    return True, "pass", score

def main():
    # 데이터 로드
    entries = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    
    print(f"📂 원본 데이터: {len(entries)}건")
    
    # 1단계: 필터링
    passed_entries = []
    reject_reasons = {}
    
    for entry in tqdm(entries, desc="Filtering"):
        inst = entry.get("instruction", "")
        out = entry.get("output", "")
        
        passed, reason, score = is_quality_response(inst, out)
        
        if passed:
            entry["_quality_score"] = score
            passed_entries.append(entry)
        else:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    
    print(f"\n📊 1차 필터링: {len(entries)}건 → {len(passed_entries)}건")
    print(f"📊 제거 사유:")
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"   {reason}: {count}건")
    
    # 2단계: 품질 점수 기반 상위 30,000건 선별
    if len(passed_entries) > TARGET_COUNT:
        # 점수 높은 순으로 정렬 후 상위 선택, 같은 점수 내에서는 랜덤
        passed_entries.sort(key=lambda x: (-x["_quality_score"], random.random()))
        selected = passed_entries[:TARGET_COUNT]
        print(f"\n📊 2차 선별: {len(passed_entries)}건 → {TARGET_COUNT}건 (품질 점수 상위)")
    else:
        selected = passed_entries
        print(f"\n📊 필터 후 {len(selected)}건 (목표 {TARGET_COUNT}건 미만, 전체 사용)")
    
    # 품질 점수 분포 출력
    score_dist = {}
    for entry in selected:
        s = entry["_quality_score"]
        score_dist[s] = score_dist.get(s, 0) + 1
    print(f"📊 선별된 데이터 품질 점수 분포:")
    for s in sorted(score_dist.keys()):
        print(f"   점수 {s}: {score_dist[s]}건")
    
    # 저장 (품질 점수 필드 제거)
    random.shuffle(selected)  # 셔플
    os.makedirs(os.path.dirname(FILTERED_PATH), exist_ok=True)
    with open(FILTERED_PATH, "w", encoding="utf-8") as f:
        for entry in selected:
            save_entry = {
                "instruction": entry["instruction"],
                "output": entry["output"]
            }
            f.write(json.dumps(save_entry, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 최종 {len(selected)}건 저장 완료!")
    print(f"💾 저장: {FILTERED_PATH}")
    
    # 샘플 출력
    print(f"\n📋 선별 샘플 5개:")
    for entry in selected[:5]:
        print(f"   [내담자] {str(entry['instruction'])[:60]}...")
        print(f"   [상담사] {str(entry['output'])[:60]}...")
        print()

if __name__ == "__main__":
    main()