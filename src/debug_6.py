import os
import requests
from dotenv import load_dotenv

# 1. 환경변수 불러오기
load_dotenv()
API_KEY = os.getenv("PUBLIC_DATA_API_KEY")

# 2. 에러가 나는 6번 API 주소
url = "https://apis.data.go.kr/1383000/yhis/YouthLvgFcltPoiService/getYouthLvgFcltPoiList"

# 3. 파라미터 세팅 (명세서 기준 필수값 포함)
params = {
    "serviceKey": API_KEY, # requests가 알아서 인코딩/디코딩 처리하게 맡깁니다.
    "pageNo": "1",
    "numOfRows": "10",
    "type": "json",
    "fcltNm": "",
    "ctpvNm": "",
    "sggNm": ""
}

print("=== API 서버에 직접 요청을 보냅니다 ===")
# 4. API 요청
response = requests.get(url, params=params, verify=False) # SSL 인증 무시 옵션 추가

# 5. [핵심] JSON 파싱을 시도하지 않고, 서버가 보낸 날것의(Raw) 텍스트를 그대로 출력합니다!
print(f"HTTP 상태 코드: {response.status_code}")
print("서버 응답 원문:\n")
print(response.text)