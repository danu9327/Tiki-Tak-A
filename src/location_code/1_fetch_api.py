import os
import json
import requests
import time
from dotenv import load_dotenv
import math

load_dotenv()
API_KEY = os.getenv("PUBLIC_DATA_API_KEY")
BASE_DIR = "/home/user/Tiki-Tak-A/"

API_CONFIGS = [
    ###########위치 관련 9개 데이터##################
    #{ # 1
    #    "name": "청소년디딤센터_위치현황",
    #    "category": "location",
    #    # 쿼리 스트링(?page=...)을 제외한 '순수 API 주소'만 넣습니다.
    #    "base_url": "https://api.odcloud.kr/api/15100268/v1/uddi:2b0c0abb-fa41-44f1-b8df-87ec27f402f8"
    #},
    #{ # 2
    #    "name": "아동청소년보호기관정보_위치현황",
    #    "category": "location",
    #    "base_url": "https://api.odcloud.kr/api/15059846/v1/uddi:b95cc85f-d5a6-4169-b0ae-9b1669d83449"
    #},
    #{ # 3
    #    "name": "청소년상담복지센터_위치현황",
    #    "category": "location",
    #    "base_url": "https://api.odcloud.kr/api/3084537/v1/uddi:09e173fc-9c56-4b0d-9c57-f08e593a486b"
    #},
    #{ # 4
    #    "name": "청소년쉼터_위치현황",
    #    "category": "location",
    #    "base_url": "https://api.odcloud.kr/api/3084536/v1/uddi:97a8b1dd-de6f-46df-8aba-cf41a1cb3b94"
    #},
    #{ # 5
    #    "name": "청소년자립지원관_위치현황",
    #    "category": "location",
    #    "base_url": "https://api.odcloud.kr/api/15100267/v1/uddi:b09b965e-7a69-4c5f-a100-8c14d6932b67"
    #},
    #{ # 6 미친 1-5까지 다른 유형
    #    "name": "청소년복지시설관심지점정보_위치현황",
    #    "category": "location",
    #    "base_url": "https://apis.data.go.kr/1383000/yhis/YouthLvgFcltPoiService/getYouthLvgFcltPoiList"
    #},
    { # 7 1-5, 6이랑 다른 유형 홀리몰리
        "name": "여성·가족·청소년·권익시설정보_위치현황", 
        "category": "location",
        "base_url": "http://apis.data.go.kr/1383000/facility/selectList",
        "extra_params": {
            "type": "json" 
        }
    },
    #{ # 8
    #    "name": "청소년성문화센터_위치현황", 
    #    "category": "location",
    #    "base_url": "http://apis.data.go.kr/1383000/spis/teenagerCenterService/getTeenagerCenterList",
    #    "extra_params": {
    #        "type": "json" 
    #    }
    #},
    #{ # 9
    #    "name": "청소년지원시설관심지점_위치현황", 
    #    "category": "location",
    #    "base_url": "https://apis.data.go.kr/1383000/yhis/YouthUseFcltPoiService/getYouthUseFcltPoiList",
    #    "extra_params": {
    #        "type": "json",
    #        "fcltNm": "",
    #        "ctpvNm": "",
    #        "sggNm": ""
    #    }
    #},
]

def fetch_all_pages(config):
    target_dir = os.path.join(BASE_DIR, f"data/rag/{config['category']}")
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, f"{config['name']}.json")

    base_url = config['base_url']
    is_odcloud = "odcloud.kr" in base_url
    
    p_page = "page" if is_odcloud else "pageNo"
    p_size = "perPage" if is_odcloud else "numOfRows"
    per_page = 1000 
    
    if is_odcloud:
        request_url = f"{base_url}?serviceKey={API_KEY}"
        params = {p_page: 1, p_size: 1}
    else:
        request_url = base_url
        params = {"serviceKey": API_KEY, p_page: 1, p_size: 1}
    
    if "extra_params" in config:
        params.update(config["extra_params"])

    all_data = []
    print(f"\n[{config['name']}] 수집 시작...")

    try:
        # Step 1: 전체 개수 파악
        res = requests.get(request_url, params=params, verify=False, timeout=15)
        
        try:
            first_res = res.json()
        except ValueError:
            print(f"  -> [서버 에러] 서버 응답: {res.text[:250]}")
            return
            
        # [업그레이드] 돌연변이 구조(body가 리스트인 경우)까지 감지하는 totalCount 파서
        if is_odcloud:
            total_count = first_res.get('totalCount', 0)
        else:
            # 1. 정상 구형 (response > body)
            if 'response' in first_res and 'body' in first_res['response']:
                total_count = int(first_res['response']['body'].get('totalCount', 0))
            # 2. 돌연변이 구형 (body가 리스트 [ ] 로 감싸진 경우 - 7번 API)
            elif 'body' in first_res and isinstance(first_res['body'], list) and len(first_res['body']) > 0:
                total_count = int(first_res['body'][0].get('totalCount', 0))
            # 3. 초단순 구형 (totalCount가 밖으로 나와있는 경우)
            elif 'totalCount' in first_res:
                total_count = int(first_res['totalCount'])
            else:
                total_count = 0

        if total_count == 0:
            print("  -> 데이터를 찾을 수 없습니다.")
            return

        total_pages = math.ceil(total_count / per_page)
        print(f"  -> 총 데이터: {total_count}건 / 예상 페이지: {total_pages}개")

        # Step 2: 루프 수집
        for page in range(1, total_pages + 1):
            params[p_page] = page
            params[p_size] = per_page
            
            for attempt in range(3):
                try:
                    response = requests.get(request_url, params=params, verify=False, timeout=20)
                    page_json = response.json()
                    
                    # [업그레이드] 돌연변이 구조용 데이터 추출 파서
                    if is_odcloud:
                        page_data = page_json.get('data', [])
                    else:
                        if 'response' in page_json and 'body' in page_json['response']:
                            items_wrapper = page_json['response']['body'].get('items', {})
                        elif 'body' in page_json and isinstance(page_json['body'], list) and len(page_json['body']) > 0:
                            items_wrapper = page_json['body'][0].get('items', {})
                        else:
                            items_wrapper = {}

                        if items_wrapper and 'item' in items_wrapper:
                            page_data = items_wrapper['item']
                            if isinstance(page_data, dict): page_data = [page_data]
                        else:
                            # 최후의 수단: 아무 이름의 리스트나 다 긁어오기
                            page_data = []
                            for key, value in page_json.items():
                                if isinstance(value, list) and key not in ['header', 'body']:
                                    page_data = value
                                    break

                    all_data.extend(page_data)
                    print(f"  -> {page}/{total_pages} 완료 ({len(all_data)}건)")
                    break 
                except Exception as e:
                    print(f"  -> {page}p 실패({attempt+1}/3): {e}")
                    time.sleep(2)
            time.sleep(0.1)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"  -> [완료] {len(all_data)}건 저장 완료!")

    except Exception as e:
        print(f"  -> 치명적 에러 발생: {e}")
                
if __name__ == "__main__":
    for config in API_CONFIGS:
        fetch_all_pages(config)