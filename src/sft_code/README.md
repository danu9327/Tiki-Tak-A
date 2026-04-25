
### 또래 페르소나 챗봇 모델 학습 코드들

##### 심리상담데이터(risk), sns대화데이터(sft)를 정제 및 변환하고 EXAONE-3.5-7.8B를 LoRA로 파인튜닝하여 "또래 페르소나" 챗봇을 만들 코드들

## 실행 순서
```bash
python src/sft_code/0_SNSdata_select.py      # SNS 데이터 샘플링
```
```bash
python src/sft_code/1_filter_counseling.py   # 상담 데이터 필터링
```
```bash
python src/sft_code/2_convert_tone.py        # 또래 말투 변환 (약 6시간)
```
```bash
python src/sft_code/3_merge_data.py          # 데이터 통합
```
```bash
python src/sft_code/4_train_exaone.py        # LoRA SFT 학습 (약 22시간....ㅎ)
```
|단계|입력|출력|
|---|----|----|
|0_SNSdata_select.py|data/jsons/ (SNS 원본)|sft_from_sns_data.jsonl|
|1_filter_counseling.py|sft_from_risk_data.jsonl (362,107건)|sft_counseling_filtered.jsonl (30,000건)|
|2_convert_tone.py|sft_counseling_filtered.jsonl|sft_counseling_peer_tone.jsonl|
|3_merge_data.py|상담 + SNS jsonl|sft_total.jsonl|
|4_train_exaone.py|sft_total.jsonl|models/tuned/exaone_sft_lora/|

## 0_SNSdata_select.py
##### 데이터에 2~4인 대화가 있어서 2인대화만 추출
##### 데이터 설명처럼 총 9개 토픽별 균등하게 데이터 추출
##### 한명이 연속으로 계속 떠든 경우->하나의 발화로 처리
###### 9개 토픽: 건강및식음료 / 경제및사회 / 과학기술 / 문화생활및여가 / 미용과패션 / 스포츠및e스포츠 / 여행관광및명소 / 정치 / 콘텐츠소비

## 1_filter_counseling.py
##### sft_from_risk_data.jsonl가 362,107건으로 너무너무 많아서 고품질 상담 데이터 30,000건을 추출
|추출 시 기준|
|-----------|
|내담자 발화 최소 15자, 상담사 응답 최소 30자|
|응답 상한 500자 추가|
|단순 질문 기준 최소 30자|
|비상담 패턴("다음 상담은", "예약", "동의서" 등) 탈락|
|공감 키워드 포함된 응답 추출|
###### 공감 키워드: 힘들 / 괜찮 / 마음 / 감정 / 이해 / 공감 / 불안 / 우울 / 외로 / 고민 등 26개

## 2_convert_tone.py
##### 고품질 상담 데이터 30,000건이 너무 딱딱한 어른 말투
##### 추출한 30,000건을 "또래 페르소나"를 위해 src/sft_code/convert_tone.py로 말투 변환(EXAONE_7.8B 로컬 모델 활용)->무려 6시간....
|변환 프롬프트 규칙|
|-----------------|
|반말로 변환 (존댓말 금지)|
|공감의 내용 유지|
|조언이 있으면 부드럽게 유지|
|과도하게 늘리지 않음|
|ㅠㅠ, ㅋㅋ, ㄹㅇ, 진짜 같은 표현 허용|

|변환예시_1|
|----------|
|[원문] 그래요. 그래요. 주변 사람들의 도움. 그다음에 아예 담배를 피울 수 없는 상황.|
|[변환] 진짜? 주변 사람들이 도와주고, 담배 못 피우면 해결되나 봐. 그럴 때마다 힘내!|

|변환예시_2|
|---------|
|[원문] 최근 일주일 동안 크게 짜증이 났거나 화가 난 적이 있어?...|
|[변환] 최근 일주일간 짜증 났던 거 진짜 많았니?...|

## 3_merge_data.py
##### 또래 말투로 변환된 상담 데이터와 SNS 데이터를 하나로 통합
##### 최종 결과->data/sft/sft_total.jsonl

## 4_train_exaone.py
##### 진짜 간당간당한 학습 파라미터들 공유 아래 참고
###### 32077MiB /  32607MiB <- 거의 터지기 직전
##### 하이퍼파라미터

|항목|값|
|---|--|
|베이스 모델|EXAONE-3.5-7.8B|
|Epochs|3|
|Learning Rate|1e-4|
|실효 배치 크기|16 (per_device=1 × gradient_accumulation=16)|
|Max Length|512|
|Optimizer|paged_adamw_8bit|
|Precision|bf16|
|Warmup Ratio|5%|

##### LoRA 설정
|항목|값|
|---|--|
|r|16|
|alpha|32|
|dropout|0.05|
|target modules|q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj|

###### 프롬프트 구조 및 마스킹
###### EXAONE 채팅 템플릿을 사용하고 프롬프트 부분은 labels=-100으로 마스킹하여 응답 부분만 손실에 반영