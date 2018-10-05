#!/usr/bin/env python
#  coding=utf-8

"""
Creates a new model from a set of options.
"""

import argparse
import logging

import models.reinvent.model as mm
import models.reinvent.vocabulary as mv
import models.reinvent.utils as mu


class CreateModelRunner:
    """Creates a new model from a set of given parameters."""

    def __init__(self, input_smiles_path, output_model_path, num_gru_layers=3, gru_layer_size=512,
                 embedding_layer_size=256):
        """
        Creates a CreateModelRunner.
        :param input_smiles_path: The input smiles string.
        :param output_model_path: The path to the newly created model.
        :param num_gru_layers: Number of GRU Layers.
        :param gru_layer_size: Size of each GRU layer.
        :param embedding_layer_size: Size of the embedding layer.
        :return:
        """
        self._smiles = mu.read_smi_file(input_smiles_path)
        self._output_model_path = output_model_path

        self._num_gru_layers = num_gru_layers
        self._gru_layer_size = gru_layer_size
        self._embedding_layer_size = embedding_layer_size

        self._already_run = False

    def run(self):
        """
        Performs the creation of the model.
        """
        if self._already_run:
            return

        logging.info("Building vocabulary")
        vocabulary = mv.Vocabulary()
        vocabulary.init_from_smiles_list(self._smiles)

        logging.info("Saving model at %s", self._output_model_path)
        rnn_params = {
            'num_gru_layers': self._num_gru_layers,
            'gru_layer_size': self._gru_layer_size,
            'embedding_layer_size': self._embedding_layer_size
        }
        model = mm.Model(voc=vocabulary, rnn_params=rnn_params)
        model.save(self._output_model_path)


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Create a model with the vocabulary extracted from a SMILES file.")

    parser.add_argument("--input-smiles-path", "-i",
                        help=(
                            "SMILES to calculate the vocabulary from. The SMILES are taken as-is, no processing is done."),
                        type=str, required=True)
    parser.add_argument("--output-model-path", "-o", help="Prefix to the output model.", type=str, required=True)
    parser.add_argument("--num-gru-layers", "-n", help="Number of GRU layers of the model [DEFAULT: 3]", type=int)
    parser.add_argument("--gru-layer-size", "-s", help="Size of each of the GRU layers [DEFAULT: 512]", type=int)
    parser.add_argument("--embedding-layer-size", "-e", help="Size of the embedding layer [DEFAULT: 256]", type=int)

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def run_main():
    """Main function"""
    args = parse_args()

    runner = CreateModelRunner(**args)
    runner.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s',
                        datefmt='%H:%M:%S')

    run_main()
