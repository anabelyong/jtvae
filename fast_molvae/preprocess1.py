import sys
sys.path.append('../')
import torch
import torch.nn as nn
from multiprocessing import Pool
import numpy as np
import os
from tqdm import tqdm

import math, random, sys
from optparse import OptionParser
import pickle

from fast_jtnn import *
import rdkit
import logging

#this preprocess.py works!
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using device: {device}')

def tensorize(smiles, assm=True):
    try:
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        if assm:
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
        return mol_tree
    except Exception as e:
        logger.error(f"Failed to process molecule with SMILES: {smiles} due to error: {str(e)}")
        return None  # Skip problematic molecule

def convert(train_path, pool, num_splits, output_path):
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    out_path = os.path.join(output_path, './')
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    with open(train_path) as f:
        data = [line.strip().split()[0] for line in f]
    logger.info('Input File read')
    
    logger.info('Tensorizing .....')
    total_count = 0  # Initialize the counter
    all_data = []
    
    for mol in tqdm(data):
        result = pool.apply_async(tensorize, (mol,))
        mol_tree = result.get()
        if mol_tree is not None:
            all_data.append(mol_tree)
            total_count += 1
            if total_count % 100 == 0:  # Log progress every 100 molecules
                logger.info(f"Processed {total_count} molecules")
    
    logger.info(f"Total molecules processed: {total_count}")
    
    all_data_split = np.array_split(all_data, num_splits)
    logger.info('Tensorizing Complete')
    
    for split_id in range(num_splits):
        with open(os.path.join(output_path, f'tensors-{split_id}.pkl'), 'wb') as f:
            pickle.dump(all_data_split[split_id], f)
    
    return True

def main_preprocess(train_path, output_path, num_splits=10, njobs=os.cpu_count()):
    pool = Pool(njobs)
    convert(train_path, pool, num_splits, output_path)
    return True

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    parser.add_option("-o", "--output", dest="output_path")
    
    opts, args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)
    convert(opts.train_path, pool, num_splits, opts.output_path)
