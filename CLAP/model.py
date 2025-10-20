import torch
import torch.nn as nn
from transformers import ClapModel


class DepressionClassifier(nn.Module):
    def __init__(self, clap_model_name="laion/clap-htsat-unfused", num_labels=2, num_unfrozen_layers=1):
        super().__init__()

        print("Loading CLAP model...")
        self.clap = ClapModel.from_pretrained(clap_model_name)

        print("Freezing CLAP backbone...")
        for param in self.clap.parameters():
            param.requires_grad = False

        # --- 核心修正：使用正确的属性名 `audio_model` ---
        if num_unfrozen_layers > 0:
            print(f"Unfreezing the last {num_unfrozen_layers} layers of the audio model...")
            # 根据报错提示，正确的路径是 .audio_model
            try:
                # 适用于 Transformer 结构的编码器 (如 HTSAT)
                layers = self.clap.audio_model.encoder.layers
                for i in range(len(layers) - num_unfrozen_layers, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                    print(f"  - Unfroze Layer {i}")
            except AttributeError:
                # 备用方案：如果模型结构不是预期的Transformer，则打印警告
                print("Warning: Could not find '.encoder.layers'. Fine-tuning might not be applied as expected.")
                print("         Attempting to unfreeze the entire 'audio_model' as a fallback.")
                for param in self.clap.audio_model.parameters():
                    param.requires_grad = True

        embedding_dim = self.clap.config.projection_dim

        # 依然使用增强的分类头
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_labels)
        )
        print(f"Enhanced classification head initialized.")

    def forward(self, input_features):
        # --- 核心修正：forward调用中也使用正确的名称 ---
        # 注意：get_audio_features 是 ClapModel 的顶层方法，这个调用本身是正确的，无需修改。
        # 它的内部会自动调用 self.clap.audio_model
        audio_embeds = self.clap.get_audio_features(input_features=input_features)
        logits = self.classification_head(audio_embeds)
        return logits