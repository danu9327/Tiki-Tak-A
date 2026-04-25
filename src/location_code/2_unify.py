import json
import os
import re

SOURCE_DIR = "/home/user/Tiki-Tak-A/data/rag/location"
OUTPUT_PATH = "/home/user/Tiki-Tak-A/data/rag/location/unified_centers_lite.json"

def normalize_phone(phone):
    """전화번호를 일관된 하이픈 형식으로 정규화"""
    if not phone or phone == "1388":
        return "1388"
    
    # 숫자만 추출
    digits = re.sub(r"\D", "", phone)
    
    # 지역번호별 포맷팅
    if digits.startswith("02"):
        if len(digits) == 9:    # 02-XXX-XXXX
            return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
        elif len(digits) == 10: # 02-XXXX-XXXX
            return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
    elif len(digits) == 11:     # 0XX-XXXX-XXXX
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    elif len(digits) == 10:     # 0XX-XXX-XXXX
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    
    return phone  # 포맷 불가 시 원본 반환

def build_rag_text(item, filename):
    """검색용 텍스트 필드 생성: 원본 데이터의 유용한 정보를 모두 포함"""
    
    # 파일별로 RAG에 유용한 추가 필드 정의
    extra_fields = {
        "아동청소년보호기관정보_위치현황.json": [
            ("기관구분값", "기관유형"),
        ],
        "여성·가족·청소년·권익시설정보_위치현황.json": [
            ("facType", "시설유형"),
        ],
        "청소년디딤센터_위치현황.json": [
            ("시설종류", "시설종류"),
            ("대상 이용자", "대상 이용자"),
            ("주요 프로그램", "주요 프로그램"),
            ("입소자현황", "입소자현황"),
            ("서비스 이용자 구분", "서비스 이용자 구분"),
        ],
        "청소년복지시설관심지점정보_위치현황.json": [
            ("fcltClsfNm", "시설분류"),
            ("fcltTypeNm", "시설유형"),
            ("fcltExpln", "시설설명"),
        ],
        "청소년상담복지센터_위치현황.json": [
            ("홈페이지", "홈페이지"),
        ],
        "청소년성문화센터_위치현황.json": [
            ("hmpgAddr", "홈페이지"),
        ],
        "청소년쉼터_위치현황.json": [
            ("시설유형", "시설유형"),
        ],
        "청소년자립지원관_위치현황.json": [
            ("시설유형", "시설유형"),
        ],
        "청소년지원시설관심지점_위치현황.json": [
            ("fcltClsfNm", "시설분류"),
            ("fcltTypeNm", "시설유형"),
            ("fcltExpln", "시설설명"),
        ],
    }
    
    parts = []
    for orig_key, display_label in extra_fields.get(filename, []):
        value = item.get(orig_key, "").strip()
        if value:
            parts.append(f"{display_label}: {value}")
    
    return " | ".join(parts) if parts else ""

def unify_to_text():
    unified_list = []
    
    # 9개 파일별 핵심 필드 매핑 (이름, 주소, 지역, 연락처 키)
    mapping = {
        "아동청소년보호기관정보_위치현황.json": ("아동청소년보호기관명", "기관주소", "주소지시군구", "기관전화번호"),
        "여성·가족·청소년·권익시설정보_위치현황.json": ("facname", "address", "", "phone"),
        "청소년디딤센터_위치현황.json": ("시설명", "시설주소", "시군구", "대표전화"),
        "청소년복지시설관심지점정보_위치현황.json": ("fcltNm", "daddr", "sggNm", "rprsTelno"),
        "청소년상담복지센터_위치현황.json": ("센터명", "주소", "시군구명", "전화번호_1"),
        "청소년성문화센터_위치현황.json": ("teenGdctCntrNm", "addr", "areaDvsnNm", "telno"),
        "청소년쉼터_위치현황.json": ("시설명", "시설주소", "시군구", "대표전화"),
        "청소년자립지원관_위치현황.json": ("시설명", "시설주소", "시군구", "대표전화"),
        "청소년지원시설관심지점_위치현황.json": ("fcltNm", "daddr", "sggNm", "rprsTelno"),
    }

    for filename, (name_k, addr_k, region_k, phone_k) in mapping.items():
        file_path = os.path.join(SOURCE_DIR, filename)
        if not os.path.exists(file_path):
            print(f"⏩ 파일을 찾을 수 없어 건너뜁니다: {filename}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                # 시군구 정보 처리
                region = item.get(region_k, "")
                address = item.get(addr_k, "")
                if not region and address:
                    region = " ".join(address.split()[:2])

                # 연락처 처리 + 정규화
                raw_phone = item.get(phone_k) or "1388"
                phone = normalize_phone(raw_phone)

                # RAG용 추가 정보 텍스트 생성
                extra_text = build_rag_text(item, filename)

                name = item.get(name_k, "").strip()
                if not name:
                    continue  # 이름 없는 항목 제외

                unified_list.append({
                    "name": name,
                    "address": address,
                    "phone": phone,
                    "region": region,
                    "source": filename.replace("_위치현황.json", ""),
                    "extra": extra_text,
                })

    # ============================================================
    # 중복 제거: 이름 + 주소 기준
    # ============================================================
    before_count = len(unified_list)
    seen = set()
    deduped_list = []
    
    for entry in unified_list:
        # 이름과 주소에서 공백 제거 후 비교 (표기 차이 흡수)
        dedup_key = (
            re.sub(r"\s+", "", entry["name"]),
            re.sub(r"\s+", "", entry["address"]),
        )
        if dedup_key not in seen:
            seen.add(dedup_key)
            deduped_list.append(entry)
    
    removed = before_count - len(deduped_list)

    # ============================================================
    # RAG 검색용 text 필드 조합
    # ============================================================
    for entry in deduped_list:
        # 검색 시 매칭될 통합 텍스트
        text_parts = [
            entry["name"],
            entry["address"],
            entry["region"],
        ]
        if entry["extra"]:
            text_parts.append(entry["extra"])
        
        entry["text"] = " | ".join(text_parts)

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped_list, f, ensure_ascii=False, indent=4)

    print(f"✅ 통합 완료! 총 {len(deduped_list)}개 기관 확보.")
    print(f"🔄 중복 제거: {removed}건 제거됨 ({before_count} → {len(deduped_list)})")
    
    # source별 건수 출력
    from collections import Counter
    source_counts = Counter(e["source"] for e in deduped_list)
    print(f"\n📊 데이터셋별 건수:")
    for src, cnt in source_counts.most_common():
        print(f"   {src}: {cnt}건")

if __name__ == "__main__":
    unify_to_text()