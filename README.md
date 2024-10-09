# Deep Speech 2: End-to-End Speech Recognition

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-in_progress-yellow.svg) ![License](https://img.shields.io/github/license/LuluW8071/Deep-Speech-2) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Deep-Speech-2) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/Deep-Speech-2) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/Deep-Speech-2) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Deep-Speech-2) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Deep-Speech-2)

</div>

This repository contains an implementation of the paper [Deep Speech 2: End-to-End Speech Recognition](https://arxiv.org/abs/1512.02595) using [PyTorch Lightning](https://www.pytorchlightning.ai/). Deep Speech 2 was a state-of-the-art automatic speech recognition (ASR) model designed to transcribe speech into text with end-to-end training using deep learning techniques in 2015.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  <!-- - [Evaluation](#evaluation) -->
  <!-- - [Inference](#inference) -->
- [Model Architecture](#model-architecture)
<!-- - [Results](#results) -->
- [References](#references)

## Overview

Deep Speech 2 is an improved version of the original Deep Speech model, which employs deep neural networks for end-to-end speech recognition. It features:

- **Recurrent Neural Networks (RNNs)** to capture sequential information in speech.
- **Batch Normalization** and **Bidirectional RNNs** to improve convergence and performance.
- **Connectionist Temporal Classification (CTC)** loss to align speech and text outputs without the need for frame-wise alignment.

This implementation is built using PyTorch Lightning, enabling scalability and efficient model training on modern hardware.

## Installation

1. Clone the repository:
   ```bash
   git clone --recursive https://github.com/LuluW8071/Deep-Speech-2.git
   cd deep-speech-2
   ```

2. Install **[Pytorch](https://pytorch.org/)** and  required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have `PyTorch` and `Lightning AI` installed.

## Dataset

This implementation supports popular speech recognition datasets like [LibriSpeech](http://www.openslr.org/12/). The datasets are automatically downloaded and preprocessed during training.

## Usage

### Training

To train the **Deep Speech 2** model, use the following command:

```bash
python3 train.py
```

Customize the pytorch training parameters by passing arguments in `train.py` to suit your needs:

Refer to the provided table for various flags and their descriptions.
| Args                   | Description                                                           | Default Value      |
|------------------------|-----------------------------------------------------------------------|--------------------|
| `-g, --gpus`           | Number of GPUs per node                                               | 1  |
| `-g, --num_workers`           | Number of CPU workers                                               | 8  |
| `-db, --dist_backend`           | Distributed backend to use for training                             | ddp_find_unused_parameters_true  |
| `--epochs`             | Number of total epochs to run                                         | 50                 |
| `--batch_size`         | Size of the batch                                                     | 64                 |
| `--learning_rate`      | Learning rate                                                         | 1e-3  (0.001)      |
| `--checkpoint_path` | Checkpoint path to resume training from                                 | None |
| `--precision`        | Precision of the training                                              | 16-mixed |



<!-- ### Evaluation

This will output the Word Error Rate (WER) and Character Error Rate (CER). -->

<!-- ### Inference

For performing inference on new audio samples:

```bash
python inference.py --audio path_to_audio.wav --checkpoint path_to_checkpoint.ckpt
``` -->

<!-- This will transcribe the audio file and return the predicted text. -->

## Model Architecture

The Deep Speech 2 model is composed of the following components:

- **Input Features**: Log-mel spectrogram features are extracted from the raw audio.
- **Recurrent Layers**: Bidirectional RNNs (GRUs or LSTMs) are used to model sequential speech data.
- **Batch Normalization**: Applied between layers to improve training efficiency.
- **CTC Loss**: Connectionist Temporal Classification is used to calculate the loss between predicted and true transcriptions, allowing end-to-end training.

### Architecture

![Deep Speech 2 Architecture](https://velog.velcdn.com/images/pass120/post/5b167fc2-1d24-4b91-8d91-5baef1b6a541/image.png)

<!-- ## Results

Deep Speech 2 achieves competitive results on large-scale speech datasets such as LibriSpeech, with reported Word Error Rates (WER) and Character Error Rates (CER) as benchmarks.

Example:

| Dataset       | WER  | CER  |
|---------------|------|------|
| LibriSpeech   | 5.3% | 2.8% | -->

## References

- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)
- [Sequnce Modelling with CTC](https://distill.pub/2017/ctc/)
- [KenLM](https://kheafield.com/code/kenlm/)
- [PyTorch TorchAudio Documentation](https://pytorch.org/audio/stable/index.html)

