"""Freezes and optimize the trained model checkpoint for inference. Use after training."""

import argparse
import torch
from model import SpeechRecognitionModel
from collections import OrderedDict
import os 

def trace(model):
    """
    Traces the model for optimization.

    Args:
        model (torch.nn.Module): Model to be traced.

    Returns:
        torch.jit.ScriptModule: Traced model.
    """
    model.eval()
    x = torch.rand(1, 1, 128, 300)
    traced = torch.jit.trace(model, (x))
    return traced

def main(args):
    """
    Main function to freeze and optimize the model.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    print("Loading model from", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    h_params = {
        "n_cnn_layers": 2,      # Residual CNN layer
        "model_type": "lstm",   # RNN Model
        "n_rnn_layers": 3,      # RNN Layer
        "rnn_dim": 512,         # RNN Hidden Layers
        "n_class": 29,          # Output classes
        "n_feats": 128,         # Spectrogram: n_mels 
        "stride": 2,            
        "dropout": 0.1,
    }
    model = SpeechRecognitionModel(**h_params)

    model_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    print("tracing model...")
    traced_model = trace(model)
        
    traced_model.save('optimized_model.pt')
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="freeze model checkpoint")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True,
                        help='Checkpoint of model to optimize')

    args = parser.parse_args()
    
    main(args)