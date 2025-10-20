import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ClapProcessor, get_scheduler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

from dataset import EATDCorpusDataset
from model import DepressionClassifier

# --- 1. 配置参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "laion/clap-htsat-unfused"
DATA_ROOT = "./data/EATD-Corpus"
# --- 增加Epochs给微调更多时间，降低学习率防止破坏预训练权重 ---
EPOCHS = 15
BATCH_SIZE = 16  # 微调时梯度会占用更多显存，可能需要降低Batch Size
LEARNING_RATE = 2e-5  # 微调时必须使用更小的学习率！
VALIDATION_SPLIT = 0.487
RANDOM_STATE = 42

TARGET_SAMPLE_RATE = 48000
MAX_AUDIO_SECONDS = 10
MAX_SAMPLES = TARGET_SAMPLE_RATE * MAX_AUDIO_SECONDS


# collate_fn, train_one_epoch, evaluate 函数保持不变 (此处省略以保持简洁)
def collate_fn(batch):
    processed_waveforms = []
    labels = []
    for waveform, sample_rate, label in batch:
        waveform_np = waveform.numpy()
        if waveform_np.ndim > 1:
            waveform_np = waveform_np[0, :]
        if len(waveform_np) > MAX_SAMPLES:
            waveform_np = waveform_np[:MAX_SAMPLES]
        if len(waveform_np) < MAX_SAMPLES:
            pad_width = MAX_SAMPLES - len(waveform_np)
            waveform_np = np.pad(waveform_np, (0, pad_width), mode='constant', constant_values=0)
        processed_waveforms.append(waveform_np)
        labels.append(label)
    sampling_rate = batch[0][1] if batch else TARGET_SAMPLE_RATE
    labels = torch.tensor(labels)
    return processed_waveforms, sampling_rate, labels


def train_one_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for waveforms, sample_rate, labels in tqdm(data_loader, desc="Training"):
        labels = labels.to(device)
        inputs = processor(audios=waveforms, sampling_rate=sample_rate, return_tensors="pt").to(device)
        predictions = model(inputs['input_features'])
        loss = loss_fn(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, loss_fn, processor, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for waveforms, sample_rate, labels in tqdm(data_loader, desc="Evaluating"):
            labels = labels.to(device)
            inputs = processor(audios=waveforms, sampling_rate=sample_rate, return_tensors="pt").to(device)
            predictions = model(inputs['input_features'])
            loss = loss_fn(predictions, labels)
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

    all_volunteer_ids = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    train_ids, val_ids = train_test_split(all_volunteer_ids, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)

    train_dataset = EATDCorpusDataset(data_root=DATA_ROOT, volunteer_ids=train_ids,
                                      target_sample_rate=TARGET_SAMPLE_RATE)
    val_dataset = EATDCorpusDataset(data_root=DATA_ROOT, volunteer_ids=val_ids, target_sample_rate=TARGET_SAMPLE_RATE)

    print("Calculating class weights for training set...")
    train_labels = [sample[1] for sample in train_dataset.samples]
    label_counts = Counter(train_labels)
    if label_counts[1] > 0:
        weight_for_class_1 = label_counts[0] / label_counts[1]
    else:
        weight_for_class_1 = 1.0
    class_weights = torch.tensor([1.0, weight_for_class_1], dtype=torch.float32).to(DEVICE)
    print(f"Class counts: {label_counts}")
    print(f"Calculated class weights: {class_weights}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = DepressionClassifier(clap_model_name=MODEL_ID).to(DEVICE)

    # --- 核心修改：将所有可训练的参数分组交给优化器 ---
    # 筛选出所有需要计算梯度的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    # --- 核心修改：加入 weight_decay ---
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    num_training_steps = EPOCHS * len(train_loader)
    num_warmup_steps = int(num_training_steps * 0.1)

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, processor, DEVICE)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F1-Score: {val_f1:.4f}")

    print("\nTraining finished!")
