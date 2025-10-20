import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

class EATDCorpusDataset(Dataset):
    # 确认这里的 target_sample_rate 默认值是 16000
    def __init__(self, data_root, volunteer_ids, target_sample_rate=16000, depression_threshold=53):
        self.data_root = data_root
        self.volunteer_ids = volunteer_ids
        self.threshold = depression_threshold
        self.target_sample_rate = target_sample_rate
        self.resamplers = {}
        self.samples = []
        self._load_samples()
        print(f"Dataset initialized with {len(self.samples)} audio samples.")

    def _load_samples(self):
        for volunteer_id in self.volunteer_ids:
            volunteer_folder = os.path.join(self.data_root, volunteer_id)
            if not os.path.isdir(volunteer_folder): continue
            label_file = os.path.join(volunteer_folder, 'new_label.txt')
            if not os.path.exists(label_file): continue
            with open(label_file, 'r') as f:
                sds_score = float(f.read().strip())
            label = 1 if sds_score >= self.threshold else 0
            for audio_type in ['positive', 'negative', 'neutral']:
                audio_file = f'{audio_type}_out.wav'
                audio_path = os.path.join(volunteer_folder, audio_file)
                if os.path.exists(audio_path):
                    try:
                        if os.path.getsize(audio_path) > 1024:
                            self.samples.append((audio_path, label))
                        else:
                            print(f"Warning: Skipping empty file: {audio_path}")
                    except OSError:
                        print(f"Warning: Cannot access file {audio_path}. Skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        waveform, original_sample_rate = torchaudio.load(audio_path)
        if original_sample_rate != self.target_sample_rate:
            if original_sample_rate not in self.resamplers:
                self.resamplers[original_sample_rate] = T.Resample(
                    orig_freq=original_sample_rate, new_freq=self.target_sample_rate
                )
            waveform = self.resamplers[original_sample_rate](waveform)
        return waveform, self.target_sample_rate, label