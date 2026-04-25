import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSONL_PATH = os.path.join(BASE_DIR, "data/risk/total_risk_data.jsonl")

class TotalRiskDataset(Dataset):
    """병합된 .jsonl 파일을 읽어서 모델용 텐서로 변환하는 클래스"""
    def __init__(self, jsonl_path, tokenizer_name, max_length=512):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {jsonl_path}\n먼저 데이터 병합 스크립트를 실행해주세요.")
            
        # JSONL 파일 한 줄씩 읽기 (메모리 효율적)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"✅ 총 {len(self.data)}개의 데이터를 성공적으로 로드했습니다.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)  # 분류: long 타입
        }

"""
def create_dataloaders(batch_size=16):
    tokenizer_name = "klue/roberta-large"
    dataset = TotalRiskDataset(JSONL_PATH, tokenizer_name)
    
    # 8:2 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"📊 Train: {len(train_dataset)}개 | Val: {len(val_dataset)}개")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
"""

def create_dataloaders(batch_size=16):
    """모든 데이터를 학습용으로 사용하도록 수정된 버전"""
    tokenizer_name = "klue/roberta-large"
    dataset = TotalRiskDataset(JSONL_PATH, tokenizer_name)
    
    train_dataset = dataset
    val_dataset = dataset 
    
    print(f"📊 전수 학습 모드: 총 {len(train_dataset)}개 샘플을 모두 학습에 사용합니다.")

    # max_length=512로 늘었으므로 배치 16으로 조정 (5090 VRAM 기준 안전)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader