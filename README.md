## Deep Speech 2 with Parallel MinGRU Implementation

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-in_progress-yellow.svg) ![License](https://img.shields.io/github/license/LuluW8071/Deep-Speech-2) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Deep-Speech-2) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/Deep-Speech-2) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/Deep-Speech-2) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Deep-Speech-2) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Deep-Speech-2)

</div>

This repository contains an implementation of the paper __Deep Speech 2: End-to-End Speech Recognition__ and newly proposed __parallel minGRU__ architecture from __Were RNNs All We Needed?__ using __PyTorch :fire:__ and __Lightning AI :zap:__. 

## 📜 Paper & Blogs Review 

- [x] [Gated Recurrent Neural Networks](https://arxiv.org/pdf/1412.3555)
- [x] [Deep Speech 2: End-to-End Speech Recognition](https://arxiv.org/abs/1512.02595)
- [x] [Were RNNs All We Needed?](https://arxiv.org/pdf/2410.01201)
- [x] [KenLM](https://kheafield.com/code/kenlm/)
- [x] [Boosting Sequence Generation Performance with Beam Search Language Model Decoding](https://towardsdatascience.com/boosting-your-sequence-generation-performance-with-beam-search-language-model-decoding-74ee64de435a)


## 📖 Introduction

__Deep Speech 2__ was a state-of-the-art ASR model designed to transcribe speech into text with end-to-end training using deep learning techniques in 2015.

On the other hand, **Were RNNs All We Needed?** introduces a new RNN-based architecture with a __parallelized version__ of the __minGRU__ (Minimum Gated Recurrent Unit), aiming to enhance the efficiency of RNNs by reducing the dependency on sequential data processing. This architecture enables faster training and inference, making it potentially more suitable for ASR tasks and other real-time applications.

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
| `-g, --gpus`           | Number of GPUs per node                                               | 1  |
| `-g, --num_workers`           | Number of CPU workers                                               | 8  |
| `-db, --dist_backend`           | Distributed backend to use for training                             | ddp_find_unused_parameters_true  |
| `--epochs`             | Number of total epochs to run                                         | 50                 |
| `--batch_size`         | Size of the batch                                                     | 32                |
| `-lr, --learning_rate`      | Learning rate                                                         | 2e-4  (0.0002)      | 
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

## Results

The model was trained on __LibriSpeech__ train set (100 + 360 + 500 hours) and validated on the __LibriSpeech__ test set ( ~ 10.5 hours).

<!-- | Dataset       | WER  |
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

```bibtex
@inproceedings{Feng2024WereRA,
    title   = {Were RNNs All We Needed?},
    author  = {Leo Feng and Frederick Tung and Mohamed Osama Ahmed and Yoshua Bengio and Hossein Hajimirsadegh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273025630}
}
```

