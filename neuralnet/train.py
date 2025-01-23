import comet_ml
import pytorch_lightning as pl
import os 
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from torchmetrics.text import WordErrorRate, CharErrorRate

# Load API
from dotenv import load_dotenv
load_dotenv()

from dataset import SpeechDataModule
from model import SpeechRecognitionModel
from utils import GreedyDecoder


class ASRTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(ASRTrainer, self).__init__()
        self.model = model
        self.args = args

        # Metrics
        self.losses = []
        self.val_wer, self.val_cer = [], []
        self.char_error_rate = CharErrorRate()
        self.word_error_rate = WordErrorRate()
        self.loss_fn = nn.CTCLoss(blank=28, zero_infinity=True)
        
        # Precompute sync_dist for distributed GPUs training
        self.sync_dist = True if args.gpus > 1 else False

        # Save the hyperparams of checkpoint
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate
        )

        # ReduceLROnPlateau with threshold for small changes
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.6,           # Reduce LR by multiplying it by 0.8
                patience=1,           # No. of epochs to wait before reducing LR
                threshold=3e-2,       # Minimum change in val_loss to qualify as improvement
                threshold_mode='rel', # Relative threshold (e.g., 0.1% change)
                min_lr=1e-5           # Minm. LR to stop reducing
            ),
            'monitor': 'val_loss',    # Metric to monitor
            'interval': 'epoch',      # Scheduler step every epoch
            'frequency': 1            # Apply scheduler after every epoch
        }

        return [optimizer], [scheduler]
    
    def _common_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths = batch
        output = self.model(spectrograms)        # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)         # (time, batch, n_class)
        loss = self.loss_fn(output, labels, input_lengths, label_lengths)
        return loss, output, labels, label_lengths
    
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._common_step(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        self.losses.append(loss)

        # Greedy decoding
        decoded_preds, decoded_targets = GreedyDecoder(y_pred.transpose(0, 1), labels, label_lengths)
        
        # Log final predictions
        if batch_idx % 25 == 0:
            self._text_logger(decoded_preds, decoded_targets)

        # Calculate metrics
        cer_batch = self.char_error_rate(decoded_preds, decoded_targets)
        wer_batch = self.word_error_rate(decoded_preds, decoded_targets)
        
        self.val_cer.append(cer_batch)
        self.val_wer.append(wer_batch)

        return {'val_loss': loss}


    def on_validation_epoch_end(self):
        # Calculate averages of metrics over the entire epoch
        avg_loss = torch.stack(self.losses).mean()
        avg_cer = torch.stack(self.val_cer).mean()
        avg_wer = torch.stack(self.val_wer).mean()

        # Log all metrics using log_dict
        metrics = {
            'val_loss': avg_loss,
            'val_cer': avg_cer,
            'val_wer': avg_wer
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)

        # Clear the lists for the next epoch
        self.losses.clear()
        self.val_wer.clear()
        self.val_cer.clear()

    def test_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        self.losses.append(loss)

        # Greedy decoding
        decoded_preds, decoded_targets = GreedyDecoder(y_pred.transpose(0, 1), labels, label_lengths)
        
        # Log final predictions
        if batch_idx % 4 == 0:
            self._text_logger(decoded_preds, decoded_targets)

        # Calculate metrics
        cer_batch = self.char_error_rate(decoded_preds, decoded_targets)
        wer_batch = self.word_error_rate(decoded_preds, decoded_targets)
        
        # Log all metrics using log_dict
        metrics = {
            'test_loss': loss,
            'test_cer': cer_batch,
            'test_wer': wer_batch
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)

        return metrics

    def _text_logger(self, decoded_preds, decoded_targets):
        formatted_log = []

        for i in range(len(decoded_targets)):
            formatted_log.append(f"{decoded_targets[i]}, {decoded_preds[i]}")
        log_text = "\n".join(formatted_log)
        self.logger.experiment.log_text(text=log_text)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    directory = "data"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Prepare dataset
    data_module = SpeechDataModule(batch_size=args.batch_size,
                                   train_url=[
                                    "train-clean-100", 
                                    "train-clean-360", 
                                    "train-other-500",
                                   ],
                                   valid_url=[
                                    "test-clean", 
                                    "test-other",
                                   ],
                                   test_url="dev-clean",
                                   num_workers=args.num_workers)
    data_module.setup()

    h_params = {
        "n_cnn_layers": args.resnet_layers,
        "rnn_type": args.model_type,
        "n_rnn_layers": args.rnn_layers,
        "rnn_dim": args.rnn_dim,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
    } 
    
    model = SpeechRecognitionModel(**h_params)
    speech_trainer = ASRTrainer(model=model, args=args)

    # NOTE: Comet Logger
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'), project_name=os.getenv('PROJECT_NAME'))

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./saved_checkpoint/",
        filename='deepspeech2-{epoch:02d}-{val_wer:.2f}',
        save_top_k=3,        # 3 Checkpoints
        mode='min'
    )

    # Trainer Instance
    trainer_args = {
        'accelerator': device,
        'devices': args.gpus,
        'min_epochs': 1,
        'max_epochs': args.epochs,
        'precision': args.precision,
        'check_val_every_n_epoch': 1, 
        'gradient_clip_val': args.grad_clip,
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),
                      EarlyStopping(monitor="val_loss", patience=5),
                      checkpoint_callback],
        'logger': comet_logger
    }
    
    if args.gpus > 1:
        trainer_args['strategy'] = args.dist_backend
        
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(speech_trainer, data_module, ckpt_path=args.checkpoint_path)
    trainer.validate(speech_trainer, data_module)
    trainer.test(speech_trainer, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=8, type=int, help='n data loading workers, default 0 = main process only')
    parser.add_argument('-db', '--dist_backend', default='ddp_find_unused_parameters_true', type=str,
                        help='which distributed backend to use for aggregating multi-gpu train')

    # Model Hyperparameters
    parser.add_argument('-m', '--model_type', default='lstm', type=str, help='rnn type: lstm or gru')
    parser.add_argument('-cl', '--resnet_layers', default=2, type=int, help='number of residual cnn layers')
    parser.add_argument('-nl', '--rnn_layers', default=3, type=int, help='number of rnn layers')
    parser.add_argument('-rd', '--rnn_dim', default=512, type=int, help='rnn dimension')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='size of batch')
    parser.add_argument('-gc', '--grad_clip', default=0.6, type=float, help='gradient clipping')
    parser.add_argument('-lr', '--learning_rate', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')

    # Checkpoint path
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path to a checkpoint file to load and resume training')

    args = parser.parse_args()
    main(args)
