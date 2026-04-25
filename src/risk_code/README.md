### risk(psych+youth)데이터를 정규화, 병합하고 위험도 3등급 분류모델을 학습하는 코드들

## 실행 순서
```bash
src/risk_code/1_merge_data.py
```
##### 위 코드로 data/risk에 있는 모든 데이터를 정규화 및 위험도 분류 3등급 정의 후 통합->`total_risk_data.jsonl`, 상담 페르소나를 위한 `sft_from_risk_data.jsonl`도 생성
```bash
src/risk_code/2_train_roberta.py    #로 `total_risk_data.jsonl`를 모델 학습
```
###### 사용하는 모델: RoBERTa-large (330M)
###### 모델 주요 학습 파라미터: max_length=512, batch_size=32, EPOCHS = 15,LEARNING_RATE = 2e-5
###### 사용 VRAM: 31299MiB /  32607MiB 이 이상은 불가능함 천장 찍음
###### 최종 결과 `models/tuned/risk_model_final`