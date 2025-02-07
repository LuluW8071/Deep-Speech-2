import pytorch_lightning as pl
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from utils import TextTransform

class LogMelSpec(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, hop_length=160):
        super(LogMelSpec, self).__init__()
        self.transform = T.MelSpectrogram(sample_rate=sample_rate,
                     n_mels=n_mels,
                     hop_length=hop_length)

    def forward(self, x):
        return self.transform(x)


def get_featurizer(sample_rate=16000, n_mels=128, hop_length=160):
    return LogMelSpec(sample_rate=sample_rate, 
                      n_mels=n_mels,
                      hop_length=hop_length)


class CustomAudioDataset(Dataset):
    def __init__(self, dataset, log_ex=True, valid=False):
        self.dataset = dataset
        self.text_process = TextTransform()  # Initialize TextProcess for text processing
        self.log_ex = log_ex

        if valid:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec()
            )
        else:
            self.audio_transforms = nn.Sequential(
                LogMelSpec(),
                T.FrequencyMasking(freq_mask_param=15),
                T.TimeMasking(time_mask_param=35)
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            waveform, _, utterance, _, _, _ = self.dataset[idx]
            utterance = utterance.lower()
            label = self.text_process.text_to_int(utterance)

            spectrogram = self.audio_transforms(waveform)  # (channel, feature, time)

            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)

            return spectrogram, label, spec_len, label_len

        except Exception as e:
            if self.log_ex:
                print(f"{str(e)}\r", end='')
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)


class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_url, valid_url, test_url, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.train_url = train_url
        self.valid_url = valid_url
        self.test_url = test_url
        self.num_workers = num_workers
        self.text_process = TextTransform() 

    def setup(self, stage=None):
        # Load multiple training and test URLs
        train_dataset = [torchaudio.datasets.LIBRISPEECH("./data", url=url, download=True) for url in self.train_url]
        valid_dataset = [torchaudio.datasets.LIBRISPEECH("./data", url=url, download=True) for url in self.valid_url]
        test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=self.test_url, download=True)

        # Concatenate multiple datasets into one
        combined_train_dataset = ConcatDataset(train_dataset)
        combined_valid_dataset = ConcatDataset(valid_dataset)

        self.train_dataset = CustomAudioDataset(combined_train_dataset, valid=False)
        self.valid_dataset = CustomAudioDataset(combined_valid_dataset, valid=True)
        self.test_dataset = CustomAudioDataset(test_dataset, valid=True)

    def data_processing(self, data):
        spectrograms, labels, input_lengths, label_lengths = [], [], [], []
        for (spectrogram, label, input_length, label_length) in data:
            if spectrogram is None:
                continue
            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
            labels.append(torch.Tensor(label))
            input_lengths.append(input_length)
            label_lengths.append(label_length)

        # Pad the spectrograms to have the same width (time dimension)
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        # Convert input_lengths and label_lengths to tensors
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)

        return spectrograms, labels, input_lengths, label_lengths
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.data_processing,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self.data_processing,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self.data_processing,
                          num_workers=self.num_workers,
                          pin_memory=True)