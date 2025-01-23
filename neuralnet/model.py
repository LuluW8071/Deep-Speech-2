import torch.nn as nn
import torch.nn.functional as F

from models.residual import ResidualCNN
from models.bi_lstm import BidirectionalLSTM
from models.bi_gru import BidirectionalGRU

# ==============================================================
""" Deep Speech 2: https://arxiv.org/abs/1512.02595 """
# with some modifications in RNN choices for faster convergence
# ==============================================================

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, rnn_type, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)  # cnn for extracting heirarchical features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)

        self.birnn_layers = nn.Sequential(*[
            (BidirectionalLSTM(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                            hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            if rnn_type == "lstm"
            else
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                            hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0))
            for i in range(n_rnn_layers)
        ])


        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, n_class)
        )

    def forward(self, x):
        # CNN and ResCNN layers
        x = self.cnn(x)
        x = self.rescnn_layers(x)

        # Flatten
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)

        # Linear layer
        x = x.transpose(1, 2)                                # (batch, time, feature)
        x = self.fully_connected(x)

        # BiRNN layers
        rnn_outputs = self.birnn_layers(x)  # (batch, time, rnn_dim*2)

        # Classification for each time step
        x = self.classifier(rnn_outputs)  # (batch, time, n_class)
        return x
