import sys
sys.path.append('../')
import torch
import torch.nn as nn

import math, random, sys
import argparse
import logging
from fast_jtnn import *
import rdkit
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(vocab, model_path, hidden_size=450, latent_size=56, depthT=20, depthG=3):
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    dict_buffer = torch.load(model_path)
    
    # Check and resize embeddings if necessary
    def resize_embedding_weights(model, state_dict, param_name):
        model_param = model.state_dict()[param_name]
        state_dict_param = state_dict[param_name]
        if model_param.size() != state_dict_param.size():
            logger.info(f"Resizing {param_name} from {state_dict_param.size()} to {model_param.size()}")
            if model_param.size(0) > state_dict_param.size(0):
                # New vocab is larger, we need to pad the pretrained embedding
                new_param = torch.cat([state_dict_param.to(model_param.device), model_param[state_dict_param.size(0):]], dim=0)
            else:
                # New vocab is smaller, we need to trim the pretrained embedding
                new_param = state_dict_param[:model_param.size(0)].to(model_param.device)
            state_dict[param_name] = new_param
        return state_dict
    
    # Resize weights for specific layers
    dict_buffer = resize_embedding_weights(model, dict_buffer, 'jtnn.embedding.weight')
    dict_buffer = resize_embedding_weights(model, dict_buffer, 'decoder.embedding.weight')
    dict_buffer = resize_embedding_weights(model, dict_buffer, 'decoder.W_o.weight')
    dict_buffer = resize_embedding_weights(model, dict_buffer, 'decoder.W_o.bias')
    
    # Resize and re-initialize mismatched parameters
    for param_name in ['decoder.W.weight', 'decoder.U.weight', 'A_assm.weight', 'T_mean.weight', 'T_mean.bias', 
                       'T_var.weight', 'T_var.bias', 'G_mean.weight', 'G_mean.bias', 'G_var.weight', 'G_var.bias']:
        model_param = model.state_dict()[param_name]
        state_dict_param = dict_buffer[param_name]
        if model_param.size() != state_dict_param.size():
            logger.info(f"Resizing {param_name} from {state_dict_param.size()} to {model_param.size()}")
            if len(model_param.size()) == 2:
                # Handle 2D weights
                if model_param.size(1) > state_dict_param.size(1):
                    new_param = torch.cat([state_dict_param.to(model_param.device), torch.randn((model_param.size(0), model_param.size(1) - state_dict_param.size(1)), device=model_param.device)], dim=1)
                else:
                    new_param = state_dict_param[:, :model_param.size(1)].to(model_param.device)
                if model_param.size(0) > state_dict_param.size(0):
                    new_param = torch.cat([new_param, torch.randn((model_param.size(0) - state_dict_param.size(0), model_param.size(1)), device=model_param.device)], dim=0)
                else:
                    new_param = new_param[:model_param.size(0), :]
            else:
                # Handle 1D weights (biases)
                new_param = torch.cat([state_dict_param.to(model_param.device), torch.randn((model_param.size(0) - state_dict_param.size(0)), device=model_param.device)], dim=0)
            dict_buffer[param_name] = new_param

    model.load_state_dict(dict_buffer)
    model = model.cuda()

    torch.manual_seed(0)
    return model

def main_sample(vocab, output_file, model_path, nsample, hidden_size=450, latent_size=56, depthT=20, depthG=3):
    model = load_model(vocab, model_path, hidden_size, latent_size, depthT, depthG)

    torch.manual_seed(0)
    with open(output_file, 'w') as out_file:
        for i in tqdm(range(nsample), desc="Sampling molecules"):
            out_file.write(str(model.sample_prior())+'\n')
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1} samples")

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsample', type=int, required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    args = parser.parse_args()
    
    main_sample(args.vocab, args.output_file, args.model, args.nsample, args.hidden_size, int(args.latent_size), args.depthT, args.depthG)
