#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import decrease_learning_rate, Variable, seq_to_smiles
from diversity_filters.conversions import Conversions
rdBase.DisableLog('rdApp.error')
import os
import random
import main.graph_ga.crossover as co
os.environ['CUDA_VISIBLE_DEVICES'] = f"0"


def pretrain(restore_from=None):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc")

    print('# Create a Dataset from a SMILES file')
    moldata = MolData("data/mols_filtered.smi", voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)
    print('build DataLoader')

    Prior = RNN(voc)
    print("build RNN")
    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    print("begin to learn")
    for epoch in range(1, 11):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch
            smiles = seq_to_smiles(seqs, voc)
            mating_mols = Conversions.smiles_to_mols(smiles)
            tokens = []
            tokens1 = []
            tokens2 = []

            num_mol = len(mating_mols)

            if num_mol % 2 == 1:
                mating_mols = mating_mols[:-1]
            parentsA = mating_mols[:num_mol//2]
            parentsB = mating_mols[num_mol//2:]

            for A, B in zip(parentsA, parentsB):

                non = co.crossover_return_fragment(A, B)
                if non is None:
                    continue
                frag1, frag2, new_mol = non
                smi = Chem.MolToSmiles(new_mol, canonical=False)
                f1 = Chem.MolToSmiles(co.remove_special_atoms(frag1), canonical=False)
                f2 = Chem.MolToSmiles(co.remove_special_atoms(frag2), canonical=False)
                tokens.append(voc.tokenize(smi))
                tokens1.append(voc.tokenize(f1))
                tokens2.append(voc.tokenize(f2))
            encs, encs1, encs2 = [], [], []
            for tok, f1, f2 in zip(tokens, tokens1, tokens2):
                try:
                    enc = Variable(voc.encode(tok))
                    enc1 = Variable(voc.encode(f1))
                    enc2 = Variable(voc.encode(f2))
                    encs.append(enc)
                    encs1.append(enc1)
                    encs2.append(enc2)
                except:
                    continue
            encoded = MolData.collate_fn(encs).long()
            encoded1 = MolData.collate_fn(encs1).long()
            encoded2 = MolData.collate_fn(encs2).long()

            h1 = Prior.likelihood_h_out(encoded1)
            h2 = Prior.likelihood_h_out(encoded2)

            log_p, _ = Prior.likelihood_given_h(encoded, (h1+h2)/2)
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
                with torch.inference_mode():
                    h1 = Prior.likelihood_h_out(encoded1)
                    h2 = Prior.likelihood_h_out(encoded2)
                    seqs, likelihood, _ = Prior.sample_from_h(len(encoded1), (h1+h2)/2)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    # if i < 5:
                    #     tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                loss_str = f"{loss.item():5.2f}".replace('.', '_')
                file_name = f"data/prior_co/Prior_CO_{epoch}_{step}_{loss_str}.ckpt"
                torch.save(Prior.rnn.state_dict(), file_name)


        # Save the Prior
        torch.save(Prior.rnn.state_dict(), "data/prior_co/Prior_CO.ckpt")


if __name__ == "__main__":
    pretrain(None)
