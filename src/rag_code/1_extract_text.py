import os
import fitz  # PyMuPDF (일반 텍스트 추출)
import pdfplumber  # 표 추출 전용
from tqdm import tqdm
import json

BASE_DIR = "/home/user/Tiki-Tak-A/"
SOURCE_DIR = os.path.join(BASE_DIR, "data/rag/statistics/source")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/rag/statistics/processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def table_to_text(table):
    """추출된 표를 RAG 검색에 적합한 텍스트로 변환
    
    예시 출력:
    [표] 지역 | 비율 | 전년대비
    서울 | 12.3% | +1.2%p
    부산 | 8.7% | -0.5%p
    """
    if not table or len(table) < 2:
        return ""
    
    rows = []
    for row in table:
        # None 셀을 빈 문자열로, 줄바꿈 제거
        cells = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
        # 모든 셀이 비어있는 행 건너뛰기
        if not any(cells):
            continue
        rows.append(" | ".join(cells))
    
    if not rows:
        return ""
    
    # 첫 행을 헤더로 표시
    return "[표] " + "\n".join(rows)

def extract_tables_from_page(plumber_page):
    """pdfplumber로 페이지의 모든 표를 추출하여 텍스트로 변환"""
    tables = plumber_page.extract_tables()
    table_texts = []
    
    for table in tables:
        text = table_to_text(table)
        if text and len(text) > 20:  # 너무 짧은 표 제외
            table_texts.append(text)
    
    return table_texts

def extract_text_from_pdf(pdf_path, filename):
    # 두 라이브러리로 동시에 열기
    fitz_doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)
    extracted_data = []
    
    print(f"\n📄 문서 분석 중: {filename} (총 {len(fitz_doc)}페이지)")
    
    table_count = 0
    
    for page_num in tqdm(range(len(fitz_doc))):
        # 1. 일반 텍스트 추출 (PyMuPDF)
        fitz_page = fitz_doc.load_page(page_num)
        text = fitz_page.get_text("text").strip()
        
        # 2. 표 추출 (pdfplumber)
        plumber_page = plumber_doc.pages[page_num]
        table_texts = extract_tables_from_page(plumber_page)
        table_count += len(table_texts)
        
        # 3. 텍스트 + 표 결합
        combined_parts = []
        if text and len(text) >= 50:
            combined_parts.append(text)
        if table_texts:
            combined_parts.extend(table_texts)
        
        if not combined_parts:
            continue
        
        combined = "\n\n".join(combined_parts)
            
        extracted_data.append({
            "source": filename,
            "page": page_num + 1,
            "content": combined,
            "has_table": len(table_texts) > 0
        })
    
    plumber_doc.close()
    fitz_doc.close()
    
    print(f"   → 표 {table_count}개 추출됨")
    return extracted_data

if __name__ == "__main__":
    all_documents = []
    
    for filename in os.listdir(SOURCE_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(SOURCE_DIR, filename)
            
            doc_data = extract_text_from_pdf(pdf_path, filename)
            all_documents.extend(doc_data)
            
            save_name = filename.replace(".pdf", ".json")
            save_path = os.path.join(PROCESSED_DIR, save_name)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=4)
                
    # 통계 출력
    total_tables = sum(1 for d in all_documents if d.get("has_table"))
    print(f"\n🎉 텍스트 추출 완료!")
    print(f"📊 총 {len(all_documents)}페이지 (표 포함 페이지: {total_tables}개)")
    print(f"결과물은 {PROCESSED_DIR} 폴더를 확인해 주세요.")