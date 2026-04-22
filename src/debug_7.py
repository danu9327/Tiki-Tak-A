import os
import requests
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
API_KEY = os.getenv("PUBLIC_DATA_API_KEY")

url = "http://apis.data.go.kr/1383000/facility/selectList"
params = {
    "serviceKey": API_KEY,
    "pageNo": "1",
    "numOfRows": "2", # 테스트용이라 2개만
    "type": "json"    # 명세서대로 type 파라미터 사용
}

print("=== 7번 API 서버 응답 원문 까보기 ===")
res = requests.get(url, params=params, verify=False)

print(f"상태 코드: {res.status_code}")
print(f"서버 응답 원문:\n{res.text[:1000]}") # 넉넉하게 1000자 출력