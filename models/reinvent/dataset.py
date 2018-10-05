# coding=utf-8

"""
Implementation of a SMILES dataset.
"""

import logging
from typing import List

import torch
import torch.utils.data as tud


from .utils import Variable
from .model import Model
from .vocabulary import Vocabulary


class Dataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, smiles: List[str], voc: Vocabulary):
        self.voc = voc
        self.smiles = self.voc.filter_smiles_list(smiles)
        diff = len(smiles) - len(self.smiles)
        if diff > 0:
            logging.info("Removed %d SMILES from the dataset, because the vocabulary can't encode them. The new size of "
                         "the dataset is %d", diff, len(self.smiles))

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        encoded = self.voc.encode(tokenized)
        return Variable(encoded)

    def __len__(self):
        return len(self.smiles)

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr

    @classmethod
    def for_model(cls, smiles: List[str], model: Model):
        """
        Creates a PyTorch dataset from list of SMILES for a specific model. It filters all smiles the model couldn't
        handle.
        :param smiles: List of smiles
        :param model: the model to use use
        :return: Dataset
        """
        return Dataset(smiles=smiles, voc=model.voc)
