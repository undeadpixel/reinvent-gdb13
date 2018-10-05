# coding=utf-8

"""
Vocabulary helper class
"""

import logging
import re

import numpy as np


class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ['$', '^']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file:
            self.init_from_voc_file(init_from_file)

    def __eq__(self, other):
        if isinstance(other, Vocabulary):
            if other.vocab == self.vocab:
                return True
        return False

    def encode(self, char_list):
        """
        Takes a list of characters and encodes to array of 1-hot vectors.
        :param char_list: A list of characters.
        :return: A matrix with the char list encoded in 1-hot vectors.
        """
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """
        Takes an array of indices and returns the corresponding SMILES.
        :param matrix: A 1-hot encoded matrix.
        :return: A SMILES.
        """
        chars = []
        for i in matrix:
            if i == self.vocab['$']:
                break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles, without_eos=False):
        """
        Takes a SMILES and return a list of characters/tokens.
        :param smiles: A SMILES string.
        :param without_eos: Do not add End of Sequence token (^).
        :return: A list of tokens.
        """
        regex = r"(\[[^\[\]]{1,6}\])"
        smiles = self._replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                for unit in char:
                    tokenized.append(unit)
        if not without_eos:
            tokenized.append('$')
        return tokenized

    def add_characters(self, chars):
        """
        Adds characters to the vocabulary.
        :param chars: Character list to add.
        """
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_voc_file(self, file_path):
        """
        Takes a file containing \n separated characters to initialize the vocabulary.
        :param file_path: File with the vocabulary entries.
        """
        with open(file_path, 'r') as voc_file:
            chars = voc_file.read().split()
        self.add_characters(chars)
        logging.debug("Initialized with vocabulary: %s", chars)

    def save_to_voc_file(self, file_path):
        """
        Saves the vocabulary to a file.
        :param file_path: File to store the vocabulary.
        """
        with open(file_path, 'w') as voc_file:
            for char in self.chars:
                voc_file.write(char + "\n")

    def init_from_smiles_list(self, smiles_list):
        """
        Adds all characters present in a list of SMILES.
        :param smiles_list: List of SMILES.
        """
        add_chars = set()
        for smiles in smiles_list:
            tokens = self.tokenize(smiles, without_eos=True)
            add_chars |= set(tokens)
        add_chars = list(add_chars)
        logging.debug("Add the following character to the vocabulary: %s", add_chars)
        self.add_characters(add_chars)

    @classmethod
    def _replace_halogen(cls, string):
        br_regexp = re.compile('Br')
        cl_regexp = re.compile('Cl')
        string = br_regexp.sub('R', string)
        string = cl_regexp.sub('L', string)

        return string

    def filter_smiles_list(self, smiles_list):
        """
        Filters SMILES by the characters they contain. Used to remove SMILES containing characters we cannot encode.
        :param smiles_list: A list with SMILES.
        :return: A list of valid SMILES.
        """
        smiles_list_valid = []
        for smiles in smiles_list:
            tokenized = self.tokenize(smiles)
            if all([char in self.chars for char in tokenized]):
                smiles_list_valid.append(smiles)
        return smiles_list_valid
