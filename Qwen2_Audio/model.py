import torch
import torch.nn as nn
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder


class DepressionClassifier(nn.Module):
    def __init__(self, model_name="D:/Study/PycharmProject/EATD-Corpus_Test/HF_Models/Qwen2-Audio-7B", num_labels=2):
        super().__init__()

        print("Loading backbone model with CPU offloading enabled...")
        self.audio_model = Qwen2AudioEncoder.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print("Freezing the entire backbone...")
        for param in self.audio_model.parameters():
            param.requires_grad = False

        embedding_dim = self.audio_model.config.d_model

        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_labels)
        )

        # --- 解决方案：只移动设备，保持权重为默认的 FP32 ---
        self.classification_head.to(self.audio_model.device)
        # ----------------------------------------------------

        print("Trainable classification head initialized.")
        print(
            f"Backbone device: {self.audio_model.device}, Classifier device: {next(self.classification_head.parameters()).device}, Classifier dtype: {next(self.classification_head.parameters()).dtype}")

    def forward(self, input_features):
        # 输入依然是 FP16
        input_features = input_features.to(self.audio_model.device, dtype=torch.float16)

        with torch.no_grad():
            outputs = self.audio_model(input_features=input_features)

        # audio_embeds 是 FP16
        audio_embeds = outputs.last_hidden_state.mean(dim=1)

        # autocast 会自动处理 FP16 输入到 FP32 模型的转换
        logits = self.classification_head(audio_embeds)

        # 因为分类头是 FP32，所以输出 logits 自然也是 FP32，无需转换
        return logits