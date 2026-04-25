import os
import json
import re

BASE_DIR = "/home/user/Tiki-Tak-A/"
PROCESSED_DIR = os.path.join(BASE_DIR, "data/rag/statistics/processed")
RAG_DB_INPUT_DIR = os.path.join(BASE_DIR, "data/rag/statistics/db_input")

os.makedirs(RAG_DB_INPUT_DIR, exist_ok=True)

# 청킹 설정값
MAX_CHUNK_SIZE = 500   # 청크 최대 글자 수
MIN_CHUNK_SIZE = 100   # 이보다 짧은 청크는 이전 청크에 병합
OVERLAP_SENTENCES = 1  # 이전 청크의 마지막 N문장을 다음 청크 앞에 붙임

def clean_text(text):
    """지저분한 줄바꿈과 페이지 노이즈를 정리하는 함수"""
    # 반복되는 줄바꿈 정리
    text = re.sub(r'\n\s*\n', '\n', text)
    # 중간에 있는 페이지 번호 제거
    text = re.sub(r'\n\d+\n', '\n', text)
    return text.strip()

def split_into_sentences(text):
    """텍스트를 문장 단위로 분리"""
    # 한국어 문장 종결 패턴: 다. 요. 음. 임. 됨. 등
    # 숫자 뒤의 마침표(예: 3.5%, 제2조.)는 분리하지 않도록 처리
    sentences = re.split(r'(?<=[다요음임됨함까죠니지세라])\.?\s+|(?<=[.!?])\s+', text)
    # 빈 문장 제거
    return [s.strip() for s in sentences if s.strip()]

def split_long_sentence(sent, max_size):
    """문장 분리가 안 된 긴 텍스트를 줄바꿈/쉼표/세미콜론 기준으로 재분할"""
    if len(sent) <= max_size:
        return [sent]
    
    # 줄바꿈 → 쉼표/세미콜론 순으로 분할 시도
    for delimiter in ['\n', ', ', '; ', ' - ']:
        parts = sent.split(delimiter)
        if len(parts) > 1:
            result = []
            current = ""
            for p in parts:
                if current and len(current) + len(p) > max_size:
                    result.append(current.strip())
                    current = p
                else:
                    current = current + delimiter + p if current else p
            if current.strip():
                result.append(current.strip())
            # 분할이 효과적이었으면 반환
            if all(len(r) <= max_size * 1.5 for r in result):
                return result
    
    # 어떤 구분자로도 안 되면 max_size 글자 단위로 강제 분할
    return [sent[i:i+max_size] for i in range(0, len(sent), max_size)]

def create_sentence_chunks(text, max_size, min_size, overlap_n):
    """문장 경계를 존중하는 청킹 알고리즘"""
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    
    for sent in sentences:
        # 문장 자체가 max_size보다 길면 재분할
        sub_parts = split_long_sentence(sent, max_size)
        
        for part in sub_parts:
            part_len = len(part)
            
            # 현재 청크에 추가하면 max_size 초과하는 경우
            if current_length + part_len > max_size and current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                
                # 오버랩: 이전 청크의 마지막 N문장을 가져옴
                overlap = current_chunk_sentences[-overlap_n:] if overlap_n > 0 else []
                current_chunk_sentences = overlap.copy()
                current_length = sum(len(s) for s in current_chunk_sentences)
            
            current_chunk_sentences.append(part)
            current_length += part_len
    
    # 마지막 남은 문장들 처리
    if current_chunk_sentences:
        last_chunk = " ".join(current_chunk_sentences)
        
        # 마지막 청크가 너무 짧으면 이전 청크에 병합
        if len(last_chunk) < min_size and chunks:
            chunks[-1] = chunks[-1] + " " + last_chunk
        else:
            chunks.append(last_chunk)
    
    # 후처리: min_size 미만인 청크를 인접 청크에 병합
    if len(chunks) > 1:
        merged = [chunks[0]]
        for c in chunks[1:]:
            if len(c) < min_size:
                merged[-1] = merged[-1] + " " + c
            else:
                merged.append(c)
        chunks = merged
    
    return chunks

if __name__ == "__main__":
    final_chunks = []
    
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(PROCESSED_DIR, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                pages = json.load(f)
                
                print(f"[{filename}] 청킹 시작...")
                
                for page_data in pages:
                    raw_text = page_data["content"]
                    
                    if len(raw_text) < 100:
                        continue
                        
                    clean_txt = clean_text(raw_text)
                    
                    # 문장 경계 기반 청킹
                    page_chunks = create_sentence_chunks(
                        clean_txt, 
                        MAX_CHUNK_SIZE, 
                        MIN_CHUNK_SIZE, 
                        OVERLAP_SENTENCES
                    )
                    
                    for i, chunk_text in enumerate(page_chunks):
                        final_chunks.append({
                            "source": page_data["source"],
                            "page": page_data["page"],
                            "chunk_index": i + 1,
                            "content": chunk_text
                        })

    # 최종 결과물 저장
    output_path = os.path.join(RAG_DB_INPUT_DIR, "final_rag_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=4)
    
    # 청크 길이 통계 출력
    lengths = [len(c["content"]) for c in final_chunks]
    import numpy as np
    print(f"\n🎉 청킹 완료! 총 {len(final_chunks)}개의 청크가 생성되었습니다.")
    print(f"📊 청크 길이 통계: 평균 {np.mean(lengths):.0f}자 | "
          f"중앙값 {np.median(lengths):.0f}자 | "
          f"최소 {min(lengths)}자 | 최대 {max(lengths)}자")
    print(f"결과: '{output_path}'")