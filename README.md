# Deep Speech 2: End-to-End Speech Recognition

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-in_progress-yellow.svg) ![License](https://img.shields.io/github/license/LuluW8071/Deep-Speech-2) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Deep-Speech-2) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/Deep-Speech-2) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/Deep-Speech-2) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Deep-Speech-2) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Deep-Speech-2)

</div>

This repository contains an implementation of the paper [Deep Speech 2: End-to-End Speech Recognition](https://arxiv.org/abs/1512.02595) using [Lightning AI :zap:](https://www.pytorchlightning.ai/). Deep Speech 2 was a state-of-the-art automatic speech recognition (ASR) model designed to transcribe speech into text with end-to-end training using deep learning techniques in 2015.

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

This implementation supports [__LibriSpeech__](http://www.openslr.org/12/). The datasets are automatically downloaded and preprocessed during training.

## Usage

### Training

>[!IMPORTANT]
> Before training make sure you have placed __comet ml api key__ and __project name__ in the environment variable file `.env`.

To train the **Deep Speech 2** model, use the following command for default training configs:

```bash
python3 train.py
```

Customize the pytorch training parameters by passing arguments in `train.py` to suit your needs:

Refer to the provided table to change hyperparameters and train configurations.
| Args                   | Description                                                           | Default Value      |
|------------------------|-----------------------------------------------------------------------|--------------------|
| `-g, --gpus`           | Number of GPUs per node                                               | 1  |
| `-g, --num_workers`           | Number of CPU workers                                               | 8  |
| `-db, --dist_backend`           | Distributed backend to use for training                             | ddp_find_unused_parameters_true  |
| `--epochs`             | Number of total epochs to run                                         | 50                 |
| `--batch_size`         | Size of the batch                                                     | 32                |
| `-lr, --learning_rate`      | Learning rate                                                         | 1e-5  (0.00001)      |
| `--checkpoint_path` | Checkpoint path to resume training from                                 | None |
| `--precision`        | Precision of the training                                              | 16-mixed |


```bash
python3 train.py 
-g 4                   # Number of GPUs per node for parallel gpu training
-w 8                   # Number of CPU workers for parallel data loading
--epochs 10            # Number of total epochs to run
--batch_size 64        # Size of the batch
-lr 2e-5               # Learning rate
--precision 16-mixed   # Precision of the training
--checkpoint_path path_to_checkpoint.ckpt    # Checkpoint path to resume training from
```


<!-- ### Evaluation

This will output the Word Error Rate (WER) and Character Error Rate (CER). -->

<!-- ### Inference

For performing inference on new audio samples:

```bash
python inference.py --audio path_to_audio.wav --checkpoint path_to_checkpoint.ckpt
``` -->

<!-- This will transcribe the audio file and return the predicted text. -->

## Model Architecture

![Deep Speech 2 Architecture](https://velog.velcdn.com/images/pass120/post/5b167fc2-1d24-4b91-8d91-5baef1b6a541/image.png)

<!-- ## Results

| Dataset       | WER  | CER  |
|---------------|------|------|
| LibriSpeech   | 5.3% | 2.8% | -->

## References

- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)
<!-- - [Sequnce Modelling with CTC](https://distill.pub/2017/ctc/) -->
- [KenLM](https://kheafield.com/code/kenlm/)
- [PyTorch TorchAudio Documentation](https://pytorch.org/audio/stable/index.html)

