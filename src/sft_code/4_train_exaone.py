import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = "/home/user/Tiki-Tak-A"
MODEL_PATH = os.path.join(BASE_DIR, "models/base/EXAONE")
SFT_DATA_PATH = os.path.join(BASE_DIR, "data/sft/sft_total.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "models/tuned/exaone_sft_lora")

# ============================================================
# 하이퍼파라미터
# ============================================================
MAX_LENGTH = 384
EPOCHS = 3
LEARNING_RATE = 1e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ============================================================
# 시스템 프롬프트
# ============================================================
SYSTEM_PROMPT = (
    "너는 청소년 또래 친구처럼 편하게 대화하면서도, "
    "상대방의 고민을 진심으로 들어주고 도움을 줄 수 있는 상담 챗봇이야. "
    "반말을 사용하고, 공감을 먼저 해준 뒤에 조언을 해줘."
)

# ============================================================
# 데이터셋 (프롬프트 마스킹 포함)
# ============================================================
class SFTDataset(Dataset):
    """instruction/output을 EXAONE 채팅 템플릿으로 변환하고,
    프롬프트 부분은 labels=-100으로 마스킹하여 응답만 학습"""

    def __init__(self, jsonl_path, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        print(f"✅ SFT 데이터 {len(self.data)}건 로드 완료")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        output = item["output"]

        # EXAONE 채팅 템플릿
        prompt = f"[|system|]{SYSTEM_PROMPT}[|endofturn|]\n[|user|]{instruction}[|endofturn|]\n[|assistant|]"
        full_text = prompt + output + "[|endofturn|]"

        # 전체 토크나이징
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()

        # 프롬프트 길이 측정 (마스킹용)
        prompt_tokenized = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        prompt_len = prompt_tokenized["input_ids"].shape[1]

        # 라벨: 프롬프트 → -100, 응답 → 실제 토큰, 패딩 → -100
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# ============================================================
# 커스텀 Data Collator
# ============================================================
class SFTDataCollator:
    def __call__(self, features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }

# ============================================================
# 학습
# ============================================================
def train():
    # 1. 토크나이저
    print("📥 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 모델 로드
    print("📥 EXAONE 모델을 bf16으로 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

    # 3. Gradient Checkpointing
    model.gradient_checkpointing_enable()

    # 4. EXAONE 전용 몽키 패치
    try:
        model.get_input_embeddings()
    except NotImplementedError:
        model.get_input_embeddings = lambda: model.transformer.wte
    try:
        model.get_output_embeddings()
    except NotImplementedError:
        model.get_output_embeddings = lambda: model.lm_head

    # 5. kbit training 준비 + LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. 데이터셋
    dataset = SFTDataset(SFT_DATA_PATH, tokenizer, MAX_LENGTH)

    # 7. Trainer 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=LEARNING_RATE,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SFTDataCollator(),
    )

    print(f"\n{'='*60}")
    print(f"📋 학습 설정")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch: 1 (실효: 16)")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"   Target Modules: {LORA_TARGET_MODULES}")
    print(f"   데이터: {len(dataset)}건")
    print(f"{'='*60}")

    print("\n🔥 학습 시작!")
    trainer.train()

    # 8. 최종 저장
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n🎉 학습 완료!")
    print(f"💾 최종 어댑터: {final_path}")
    print(f"💾 체크포인트: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
