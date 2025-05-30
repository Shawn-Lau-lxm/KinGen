#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

def pretrain(restore_from=None):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc-chembl_31")

    # Create a Dataset from a SMILES file
    moldata = MolData("data/chembl_31_filtered.smi", voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True, num_workers=16,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    # for epoch in range(1, 8):
    for epoch in range(1, 6):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long() # torch.Size([batch_size, seq_length+1])

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs) # torch.Size([batch_size])
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), "data/Prior_chembl_31.ckpt")

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), "data/Prior_chembl_31.ckpt")

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
    #device_ids = [0,1]
    #dev = torch.device('cuda:1')
    torch.multiprocessing.set_start_method('spawn')
    pretrain("data/Prior_chembl_31.ckpt")
