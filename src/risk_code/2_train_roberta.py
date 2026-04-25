import torch
from torch.optim import AdamW
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from dataloader import create_dataloaders
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import f1_score, classification_report

# 하드웨어 세팅
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 현재 사용 중인 디바이스: {device}")

# 분류 레이블 정의 (3단계)
LABEL_NAMES = ["안전", "주의", "위험"]
NUM_LABELS = len(LABEL_NAMES)

# 1. 모델 및 토크나이저 세팅
MODEL_NAME = "/home/user/Tiki-Tak-A/models/base/roberta-large"
SAVE_PATH = "/home/user/Tiki-Tak-A/models/tuned/risk_model_final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)

# 2. 하이퍼파라미터 세팅
EPOCHS = 15
LEARNING_RATE = 2e-5
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

def compute_class_weights(dataloader):
    """데이터로더에서 클래스별 샘플 수를 세어 역빈도 가중치를 계산"""
    from collections import Counter
    label_counts = Counter()
    for batch in dataloader:
        label_counts.update(batch['labels'].numpy().tolist())
    
    total = sum(label_counts.values())
    weights = []
    for i in range(NUM_LABELS):
        count = label_counts.get(i, 1)  # 0 방지
        weights.append(total / (NUM_LABELS * count))
    
    print(f"⚖️  클래스 가중치: {[f'{w:.2f}' for w in weights]}")
    print(f"   (안전={label_counts[0]}, 주의={label_counts[1]}, 위험={label_counts[2]})")
    return torch.tensor(weights, dtype=torch.float).to(device)

def train():
    # max_length=512이므로 배치 16으로 조정
    train_loader, val_loader = create_dataloaders(batch_size=32)
    
    # 클래스 불균형 보정을 위한 가중치 계산
    class_weights = compute_class_weights(train_loader)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # --- 학습(Train) 단계 ---
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # long 타입
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)  # [batch, num_labels] vs [batch]
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # 학습 정확도 계산
            preds = torch.argmax(outputs.logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = train_correct / train_total * 100
        
        # --- 검증(Validation) 단계 ---
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_losses.append(loss.item())
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        
        print(f"📈 Epoch {epoch+1} 완료 | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"📊 Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}% | Macro F1: {val_f1:.4f}")

        # --- 최적 모델 저장 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(SAVE_PATH, exist_ok=True)
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"🌟 Best Model Saved! (Loss: {best_val_loss:.4f})")
        print("-" * 50)

    # 최종 클래스별 성능 리포트
    print("\n📋 최종 클래스별 성능:")
    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0))
    print("🎉 모든 학습이 완료되었습니다!")

if __name__ == "__main__":
    train()