#!/usr/bin/env python
#  coding=utf-8

"""
Samples an existing RNN model.
"""

import argparse
import gzip
import logging
import sys

import torch
import tqdm

import models.reinvent.model as mm


class SampleFromModelRunner:
    """Samples an existing RNN model."""

    def __init__(self, model_path, output_smiles_path=None, use_gzip=False, num_smiles=1024, batch_size=128,
                 temperature=1.0,
                 with_likelihood=False, with_unique_id=False, clear_cache_after_n_iterations=0):
        """
        Creates a SampleFromModelRunner.
        :param model_path: The input model path.
        :param output_smiles_path: Path of the generated SMILES file.
        :param use_gzip: The output will be GZipped (and the .gz extension added) if True.
        :param num_smiles: Number of SMILES to sample.
        :param batch_size: Batch size (beware GPU memory usage).
        :param temperature: Temperature for the sequence sampling. Has to be larger than 0.
                            Values below 1 make the RNN more confident in it's generation, but also more conservative.
                            Values larger than 1 result in more random sequences.
        :param with_likelihood: Store the likelihood in a column after the SMILES.
        :param with_unique_id: Store a unique count ID in the first column (before the SMILES).
        :param clear_cache_after_n_iterations: Clear the GPU memory cache after N iteration (disabled = 0).
        :return:
        """
        self._model = mm.Model.load_from_file(model_path)

        if output_smiles_path:
            open_func = open
            path = output_smiles_path
            if use_gzip:
                open_func = gzip.open
                path += ".gz"
            self._output = open_func(path, "wt+")
        else:
            self._output = sys.stdout

        self._num_smiles = num_smiles
        self._batch_size = batch_size
        self._temperature = temperature

        self._with_likelihood = with_likelihood
        self._with_unique_id = with_unique_id

        self._clear_cache_iterations = clear_cache_after_n_iterations

    def __del__(self):
        if self._output != sys.stdout:
            self._output.close()

    def run(self):
        """
        Performs the sample.
        """
        current_id = 0
        num_iterations = 0
        molecules_left = self._num_smiles
        with tqdm.tqdm(total=self._num_smiles) as progress_bar:
            while molecules_left > 0:
                current_batch_size = min(self._batch_size, molecules_left)
                smiles, likelihoods = self._model.sample_smiles(
                    current_batch_size, batch_size=self._batch_size, temperature=self._temperature)

                for smi, log_likelihood in zip(smiles, likelihoods):
                    output_row = []
                    if self._with_unique_id:
                        output_row.append(str(current_id))
                    output_row.append(smi)
                    if self._with_likelihood:
                        output_row.append("{}".format(log_likelihood))
                    self._output.write("{}\n".format("\t".join(output_row)))
                    current_id += 1

                molecules_left -= current_batch_size

                if self._clear_cache_iterations > 0 and (num_iterations % self._clear_cache_iterations == 0):
                    torch.cuda.empty_cache()
                num_iterations += 1

                progress_bar.update(current_batch_size)


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Create a model with the vocabulary extracted from a SMILES file.")
    parser.add_argument("--model-path", "-m", help="Path to the model.", type=str, required=True)
    parser.add_argument("--output-smiles-path", "-o",
                        help="Path to the output file (if none given it will use stdout).", type=str)
    parser.add_argument("--num-smiles", "-n", help="Number of SMILES to sample [DEFAULT: 1024]", type=int)
    parser.add_argument("--with-likelihood", help="Store the likelihood in a column after the SMILES.",
                        action="store_true", default=False)
    parser.add_argument("--batch-size", "-b",
                        help="Batch size (beware GPU memory usage) [DEFAULT: 128]", type=int)
    parser.add_argument("--clear-cache",
                        help="Clear GPU cache after N iterations [DEFAULT: -1 (disabled)]", type=int)
    parser.add_argument("--temperature", "-t",
                        help=("Temperature for the sequence sampling. Has to be larger than 0. Values below 1 make "
                              "the RNN more confident in it's generation, but also more conservative. "
                              "Values larger than 1 result in more random sequences. [DEFAULT: 1.0]"), type=float)
    parser.add_argument("--use-gzip", help="Compress the output file (if set).", action="store_true", default=False)
    parser.add_argument("--with-unique-id", help="Store a unique count ID in the first column (before the SMILES)",
                        action="store_true", default=False)

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def run_main():
    """Main function."""
    args = parse_args()

    runner = SampleFromModelRunner(**args)
    runner.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s',
                        datefmt='%H:%M:%S')

    run_main()
