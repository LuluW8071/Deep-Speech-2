## Deep Speech 2 with Parallel MinGRU Implementation

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-in_progress-yellow.svg) ![License](https://img.shields.io/github/license/LuluW8071/Deep-Speech-2) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Deep-Speech-2) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/Deep-Speech-2) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/Deep-Speech-2) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Deep-Speech-2) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Deep-Speech-2)

</div>

This repository contains an implementation of the paper __Deep Speech 2: End-to-End Speech Recognition__, a state-of-the-art ASR model designed to transcribe speech into text with end-to-end training using deep learning techniques in 2015. using __Lightning AI :zap:__. 

## 📜 Paper & Blogs Review 

- [x] [Gated Recurrent Neural Networks](https://arxiv.org/pdf/1412.3555)
- [x] [Deep Speech 2: End-to-End Speech Recognition](https://arxiv.org/abs/1512.02595)
- [x] [KenLM](https://kheafield.com/code/kenlm/)
- [x] [Boosting Sequence Generation Performance with Beam Search Language Model Decoding](https://towardsdatascience.com/boosting-your-sequence-generation-performance-with-beam-search-language-model-decoding-74ee64de435a)

--- 

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
| `-d, --device`         | Device to use for training                                            | `cuda`             |
| `-g, --gpus`           | Number of GPUs per node                                               | `1`                |
| `-w, --num_workers`    | Number of CPU workers for data loading                                | `8`                |
| `-db, --dist_backend`  | Distributed backend to use for aggregating multi-GPU training         | `ddp`              |
| `--train_json`         | JSON file to load training data                                       | `None` (Required)  |
| `--valid_json`         | JSON file to load validation data                                     | `None` (Required)  |
| `--epochs`             | Number of total epochs to run                                         | `50`               |
| `--batch_size`         | Size of the batch                                                     | `64`               |
| `-lr, --learning_rate` | Learning rate                                                         | `5e-5`             |
| `--precision`          | Precision for mixed precision training                                | `16-mixed`         |
| `--checkpoint_path`    | Path of checkpoint file to resume training                            | `None`             |
| `-gc, --grad_clip`     | Gradient norm clipping value                                          | `0.5`              |
| `-ag, --accumulate_grad` | Number of batches to accumulate gradients over                     | `4`                |



```bash
python3 train.py \
-d cuda \                        # Device to use for training (e.g., 'cuda' for GPU, 'cpu' for CPU)
-g 2 \                           # Number of GPUs per node for parallel GPU training
-w 4 \                           # Number of CPU workers for parallel data loading
--epochs 50 \                    # Number of total epochs to run
--batch_size 32 \                # Size of each training batch
-lr 2e-4 \                       # Learning rate for optimization
--precision 16-mixed \           # Precision of the training (e.g., '16-mixed' for mixed precision training)
--train_json path_to_training_data.json \   # Path to the training data JSON file
--valid_json path_to_validation_data.json \ # Path to the validation data JSON file
--checkpoint_path path_to_checkpoint.ckpt \ # Path to a checkpoint file to resume training
-gc 1 \                        # Gradient norm clipping value to prevent exploding gradients
-ag 4                            # Number of batches to accumulate gradients over
```

## Experiment Results

<!-- The model was trained on __LibriSpeech__ train set (100 + 360 + 500 hours) and validated on the __LibriSpeech__ test set ( ~ 10.5 hours).

| Dataset       | WER  |
|---------------|------|
| LibriSpeech   | 5.3% | -->

## Citations

```bibtex
@misc{amodei2015deepspeech2endtoend,
      title={Deep Speech 2: End-to-End Speech Recognition in English and Mandarin}, 
      author={Dario Amodei and Rishita Anubhai and Eric Battenberg and Carl Case and others,
      year={2015},
      url={https://arxiv.org/abs/1512.02595}, 
}
```