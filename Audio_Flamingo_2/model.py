import torch
import torch.nn as nn
from transformers import ClapModel


class DepressionClassifier(nn.Module):
    def __init__(self, clap_model_name="laion/clap-htsat-unfused", num_labels=2, num_unfrozen_layers=0):
        super().__init__()

        print("Loading Audio_Flamingo_2 model...")
        self.clap = ClapModel.from_pretrained(clap_model_name)

        print("Freezing Audio_Flamingo_2 backbone...")
        for param in self.clap.parameters():
            param.requires_grad = False


        if num_unfrozen_layers > 0:
            # print(f"Unfreezing the last {num_unfrozen_layers} layers of the audio model...")

            try:

                layers = self.clap.audio_model.encoder.layers
                for i in range(len(layers) - num_unfrozen_layers, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                    print(f"  - Unfroze Layer {i}")
            except AttributeError:

                print("Warning: Could not find '.encoder.layers'. Fine-tuning might not be applied as expected.")
                print("         Attempting to unfreeze the entire 'audio_model' as a fallback.")
                for param in self.clap.audio_model.parameters():
                    param.requires_grad = True

        embedding_dim = self.clap.config.projection_dim


        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_labels)
        )
        print(f"Enhanced classification head initialized.")

    def forward(self, input_features):

        audio_embeds = self.clap.get_audio_features(input_features=input_features)
        logits = self.classification_head(audio_embeds)
        return logits