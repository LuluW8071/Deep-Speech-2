# Deep Speech 2

<div align="center">

![Status](https://img.shields.io/badge/status-completed-green.svg) ![License](https://img.shields.io/github/license/LuluW8071/Deep-Speech-2) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Deep-Speech-2) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/Deep-Speech-2) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/Deep-Speech-2) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Deep-Speech-2) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Deep-Speech-2)

</div>

This repository contains an implementation of the paper **Deep Speech 2: End-to-End Speech Recognition**, a state-of-the-art ASR model designed for end-to-end speech-to-text transcription using deep learning techniques. The implementation leverages **Lightning AI ⚡** for efficient training and experimentation.

---

## 📜 Paper & Blog Reviews

- ✅ [Gated Recurrent Neural Networks](https://arxiv.org/pdf/1412.3555)
- ✅ [Deep Speech 2: End-to-End Speech Recognition](https://arxiv.org/abs/1512.02595)
- ✅ [KenLM](https://kheafield.com/code/kenlm/)
- ✅ [Boosting Sequence Generation Performance with Beam Search Language Model Decoding](https://towardsdatascience.com/boosting-your-sequence-generation-performance-with-beam-search-language-model-decoding-74ee64de435a)

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LuluW8071/Deep-Speech-2.git
   cd Deep-Speech-2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have `PyTorch` and `Lightning AI` installed.

---

## 📖 Usage

### 🔥 Training

> **Important:** Before training, make sure to set your **Comet ML API key** and **project name** in the `.env` file.

To train the **Deep Speech 2** model with default configurations:
```bash
python3 train.py
```

To customize the training parameters, modify `train.py` or pass arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `-g`, `--gpus` | Number of GPUs per node | `1` |
| `-w`, `--num_workers` | Number of data loading workers | `4` |
| `-db`, `--dist_backend` | Distributed backend | `'ddp_find_unused_parameters_true'` |
| `-m`, `--model_type` | Type of RNN (`lstm` or `gru`) | `'lstm'` |
| `-cl`, `--resnet_layers` | Number of residual CNN layers | `2` |
| `-nl`, `--rnn_layers` | Number of RNN layers | `3` |
| `-rd`, `--rnn_dim` | RNN hidden size | `512` |
| `--epochs` | Number of training epochs | `50` |
| `--batch_size` | Batch size | `32` |
| `-gc`, `--grad_clip` | Gradient clipping | `0.6` |
| `-lr`, `--learning_rate` | Learning rate | `2e-4` |
| `--precision` | Precision mode | `'16-mixed'` |
| `--checkpoint_path` | Path to checkpoint file | `None` |

---

### 🧊 Export TorchScript Model

```bash
python3 freeze.py --model_checkpoint saved_checkpoint/deepspeech2.ckpt
```

### 🎙️ Inference

To perform inference using a trained model:
```bash
python3 demo.py --model_path optimized_model.pt --share
```

---

## 📊 Experiment Results

The model was trained on **LibriSpeech train set** (100 + 360 + 500 hours) and validated on the **LibriSpeech test set** (~10.5 hours) using **16-bit mixed precision**.

🔗 **Download Checkpoint**: [Google Drive Link](https://drive.google.com/file/d/14J6HhN_Op4c0y-up096eY_6_6D5JLIHb/view?usp=sharing)

### Model Performance

| Model Type | ResCNN Layers | RNN Layers | RNN Dim | Epochs | Batch Size | Grad Clip | LR |
|------------|---------------|------------|---------|--------|------------|-----------|----|
| BiLSTM     | 2             | 3          | 512     | 25     | 64         | 0.6       | 2e-4 |

#### 📉 Loss Curves
![Loss Curves](assets/loss_curves.png)

#### 📝 WER & CER Metrics (Greedy Decoding)
![Greedy Metrics](assets/greedy_metrics.png)

#### 🔍 Beam Search Decoding
| Word Score | LM Weight | N-gram LM | Beam Size | Beam Threshold |
|------------|-----------|-----------|-----------|----------------|
| -0.26       | 0.3       | 4-gram    | 25        | 10             |

![Beam Search Metrics](assets/beam_search_metrics.png)

#### 🔎 Alignments Visualization
![Alignments](assets/plot_alignments.png)

---

## 🔗 Citations

```bibtex
@misc{amodei2015deepspeech2endtoend,
      title={Deep Speech 2: End-to-End Speech Recognition in English and Mandarin},
      author={Dario Amodei and Rishita Anubhai and Eric Battenberg and Carl Case and others},
      year={2015},
      url={https://arxiv.org/abs/1512.02595}
}
```