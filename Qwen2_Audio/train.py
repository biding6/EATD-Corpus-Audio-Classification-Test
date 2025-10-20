import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoProcessor, get_scheduler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

from dataset import EATDCorpusDataset
from model import DepressionClassifier

# --- 1. 配置参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "D:/Study/PycharmProject/EATD-Corpus_Test/HF_Models/Qwen2-Audio-7B"
DATA_ROOT = "./data/EATD-Corpus"
EPOCHS = 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
VALIDATION_SPLIT = 0.487
RANDOM_STATE = 42

TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_SAMPLES = TARGET_SAMPLE_RATE * MAX_AUDIO_SECONDS


def collate_fn(batch):
    processed_waveforms = []
    labels = []
    for waveform, sample_rate, label in batch:
        waveform_np = waveform.numpy()
        if waveform_np.ndim > 1: waveform_np = waveform_np[0, :]
        if len(waveform_np) > MAX_SAMPLES: waveform_np = waveform_np[:MAX_SAMPLES]
        if len(waveform_np) < MAX_SAMPLES:
            pad_width = MAX_SAMPLES - len(waveform_np)
            waveform_np = np.pad(waveform_np, (0, pad_width), mode='constant', constant_values=0)
        processed_waveforms.append(waveform_np)
        labels.append(label)
    sampling_rate = batch[0][1] if batch else TARGET_SAMPLE_RATE
    labels = torch.tensor(labels)
    return processed_waveforms, sampling_rate, labels


def train_one_epoch(model, data_loader, loss_fn, optimizer, scheduler, processor, device, scaler):
    model.train()
    total_loss = 0
    for i, (waveforms, sample_rate, labels) in enumerate(tqdm(data_loader, desc="Training")):
        labels = labels.to(device)
        processed_inputs = processor.feature_extractor(waveforms, sampling_rate=sample_rate, return_tensors="pt").to(
            device)

        with autocast(device_type=device, dtype=torch.float16):
            predictions = model(processed_inputs.input_features)
            loss = loss_fn(predictions, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    return total_loss / len(data_loader)


def evaluate(model, data_loader, loss_fn, processor, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for waveforms, sample_rate, labels in tqdm(data_loader, desc="Evaluating"):
            labels = labels.to(device)
            processed_inputs = processor.feature_extractor(waveforms, sampling_rate=sample_rate,
                                                           return_tensors="pt").to(device)

            with autocast(device_type=device, dtype=torch.float16):
                predictions = model(processed_inputs.input_features)
                loss = loss_fn(predictions, labels)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()

            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, accuracy, f1


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    model = DepressionClassifier(model_name=MODEL_ID, num_labels=2)

    print("Model created. Verifying trainable parameters...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - Trainable: {name}")

    all_volunteer_ids = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    train_ids, val_ids = train_test_split(all_volunteer_ids, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)

    train_dataset = EATDCorpusDataset(data_root=DATA_ROOT, volunteer_ids=train_ids,
                                      target_sample_rate=TARGET_SAMPLE_RATE)
    val_dataset = EATDCorpusDataset(data_root=DATA_ROOT, volunteer_ids=val_ids, target_sample_rate=TARGET_SAMPLE_RATE)

    train_labels = [sample[1] for sample in train_dataset.samples]
    label_counts = Counter(train_labels)
    weight_for_class_1 = label_counts[0] / label_counts[1] if label_counts[1] > 0 else 1.0
    class_weights = torch.tensor([1.0, weight_for_class_1], dtype=torch.float32).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_training_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                              num_training_steps=num_training_steps)

    # --- 解决方案：移除不正确的关键字参数 ---
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, processor, DEVICE, scaler)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, processor, DEVICE)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F-Score: {val_f1:.4f}")

    print("\nTraining finished!")